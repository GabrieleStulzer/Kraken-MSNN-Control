"""
combined_vehicle_forward_model.py

Combined longitudinal + lateral + yaw forward model in nnodely.

Goal
----
Given measured state and control histories, predict the next-step state:
    (vx, vy, r)_{k+1}

This is a "forward model" you can train on:
- Simulink+Unreal co-simulation logs
- Real car logs

Key features
------------
1) Separate longitudinal / lateral-yaw submodels (learned FIR structures).
2) Coupling via a friction-ellipse saturation block (time-optimal realism).
3) Next-state prediction using an explicit Euler integrator inside the model
   so training targets are simply vx.z(-1), vy.z(-1), r.z(-1).

Signals (minimal)
-----------------
Inputs:
    vx, vy, r           [m/s, m/s, rad/s]
    delta               [rad]
    throttle            [-] (0..1) or torque request normalized
    brake               [-] (0..1) or pressure normalized

Outputs:
    vx_hat_next, vy_hat_next, r_hat_next
"""

from __future__ import annotations

from dataclasses import dataclass

# ---- nnodely core building blocks (API style from official docs) ----
# The PyPI docs show these concepts: Input, Output, Fir, Modely, ParametricFunction, etc. :contentReference[oaicite:2]{index=2}
from nnodely import Modely
from nnodely import Input, Output
from nnodely import Fir
from nnodely import ParametricFunction


# ---------------------------- Configuration ----------------------------

@dataclass
class CombinedVehicleModelConfig:
    # Discretization
    Ts: float = 0.01  # 100 Hz (recommended for racing dynamics)

    # Time-window lengths (seconds)
    # These define how much "memory" each block sees.
    Tw_u: float = 0.10        # throttle/brake memory (actuator + driveline lag)
    Tw_delta: float = 0.10    # steering actuator + tire relaxation proxy
    Tw_state: float = 0.20    # state memory (captures transients)

    # Friction ellipse parameters (initial guesses)
    g: float = 9.81
    mu_min: float = 0.6
    mu_max: float = 2.0

    # Small numerical stability constant
    eps: float = 1e-6


# ----------------------- Custom PyTorch functions ----------------------
# Implemented as ParametricFunction so they are differentiable and trainable. :contentReference[oaicite:3]{index=3}

def _friction_ellipse_saturate(ax_raw, ay_raw, mu_eff, g=9.81, eps=1e-6):
    """
    Smooth-ish friction ellipse saturation:
        eta = sqrt((ax/(mu g))^2 + (ay/(mu g))^2)
        if eta > 1, scale down both ax, ay by 1/eta

    This is the key coupling block that prevents the model from predicting
    unrealistic simultaneous high ax and high ay (crucial for time-optimal).
    """
    import torch

    denom = (mu_eff * g) + eps
    nx = ax_raw / denom
    ny = ay_raw / denom
    eta = torch.sqrt(nx * nx + ny * ny + eps)
    scale = torch.where(eta > 1.0, 1.0 / eta, torch.ones_like(eta))
    return ax_raw * scale, ay_raw * scale


def _euler_next_state(vx, vy, r, ax, ay, rdot, Ts=0.01):
    """
    Minimal body-frame kinematics:
        vx_{k+1} = vx_k + Ts * ax
        vy_{k+1} = vy_k + Ts * (ay - r*vx)    (body-frame coupling)
        r_{k+1}  = r_k  + Ts * rdot

    Notes:
    - This is intentionally "simple". Your 11-14DoF Simulink model is the data generator.
    - The learned ax/ay/rdot blocks will absorb unmodelled effects.
    """
    import torch
    vx_next = vx + Ts * ax
    vy_next = vy + Ts * (ay - r * vx)
    r_next  = r  + Ts * rdot
    return vx_next, vy_next, r_next


# -------------------------- Model construction --------------------------

def build_combined_forward_model(cfg: CombinedVehicleModelConfig) -> Modely:
    """
    Build a combined forward model (structured MS-NN) in nnodely.

    Training objective:
        minimize vx.z(-1) - vx_hat_next
        minimize vy.z(-1) - vy_hat_next
        minimize r.z(-1)  - r_hat_next

    Returns:
        A neuralized Modely instance ready for loadData() and trainModel().
    """

    # -------------------- Define inputs (signals) --------------------
    vx = Input("vx")               # longitudinal velocity
    vy = Input("vy")               # lateral velocity
    r  = Input("r")                # yaw rate

    delta    = Input("delta")      # steering angle
    throttle = Input("throttle")   # throttle command
    brake    = Input("brake")      # brake command

    # -------------------- Longitudinal block: ax_raw --------------------
    # Structured, interpretable "FIR on windows" approach:
    # - throttle/brake window -> captures lag
    # - vx window -> captures drag / rolling resistance / speed dependencies
    # - delta/r/vy windows -> captures longitudinal reduction during cornering (combined slip proxy)
    ax_raw = (
        Fir(throttle.tw(cfg.Tw_u))
        - Fir(brake.tw(cfg.Tw_u))
        + Fir(vx.tw(cfg.Tw_state))
        + Fir(delta.tw(cfg.Tw_delta))
        + Fir(r.tw(cfg.Tw_state))
        + Fir(vy.tw(cfg.Tw_state))
    )

    # -------------------- Lateral block: ay_raw --------------------
    # ay depends on: steering + (vx, vy, r) + (throttle/brake) for load transfer/combined slip proxy
    ay_raw = (
        Fir(delta.tw(cfg.Tw_delta))
        + Fir(vx.tw(cfg.Tw_state))
        + Fir(vy.tw(cfg.Tw_state))
        + Fir(r.tw(cfg.Tw_state))
        + Fir(throttle.tw(cfg.Tw_u))
        - Fir(brake.tw(cfg.Tw_u))
    )

    # -------------------- Yaw block: rdot_raw --------------------
    # yaw acceleration depends strongly on steering and lateral state.
    rdot_raw = (
        Fir(delta.tw(cfg.Tw_delta))
        + Fir(vx.tw(cfg.Tw_state))
        + Fir(vy.tw(cfg.Tw_state))
        + Fir(r.tw(cfg.Tw_state))
        + Fir(throttle.tw(cfg.Tw_u))
        - Fir(brake.tw(cfg.Tw_u))
    )

    # -------------------- Grip / coupling: mu_eff --------------------
    # Minimal version: learn mu_eff as a function of "how aggressive you are".
    #
    # In later versions you can:
    # - add inputs like tire temps, normal loads, aero estimates, surface ID
    # - make mu_eff a LocalModel (speed bins / lateral-acc bins)
    mu_eff = ParametricFunction(
        name="mu_eff_estimator",
        fn=lambda vx_in, brk_in: cfg.mu_min + (cfg.mu_max - cfg.mu_min) * (1.0 / (1.0 + (-0.5 * vx_in - 2.0 * brk_in).exp())),
        inputs=[vx.last(), brake.last()],
    )

    # -------------------- Friction ellipse saturation --------------------
    ax_ay_sat = ParametricFunction(
        name="friction_ellipse",
        fn=lambda ax_in, ay_in, mu_in: _friction_ellipse_saturate(ax_in, ay_in, mu_in, g=cfg.g, eps=cfg.eps),
        inputs=[ax_raw, ay_raw, mu_eff],
    )
    # ParametricFunction returns a tuple; nnodely exposes outputs as "parts" depending on version.
    # We keep it explicit via two extra ParametricFunctions to be robust across versions.
    ax = ParametricFunction(
        name="ax_pick",
        fn=lambda pair: pair[0],
        inputs=[ax_ay_sat],
    )
    ay = ParametricFunction(
        name="ay_pick",
        fn=lambda pair: pair[1],
        inputs=[ax_ay_sat],
    )

    # -------------------- Euler integration to next-state --------------------
    next_state = ParametricFunction(
        name="euler_next_state",
        fn=lambda vx_in, vy_in, r_in, ax_in, ay_in, rdot_in: _euler_next_state(
            vx_in, vy_in, r_in, ax_in, ay_in, rdot_in, Ts=cfg.Ts
        ),
        inputs=[vx.last(), vy.last(), r.last(), ax, ay, rdot_raw],
    )

    vx_hat_next = ParametricFunction("vx_next_pick", fn=lambda s: s[0], inputs=[next_state])
    vy_hat_next = ParametricFunction("vy_next_pick", fn=lambda s: s[1], inputs=[next_state])
    r_hat_next  = ParametricFunction("r_next_pick",  fn=lambda s: s[2], inputs=[next_state])

    # -------------------- Define model outputs --------------------
    out_vx = Output("vx_hat_next", vx_hat_next)
    out_vy = Output("vy_hat_next", vy_hat_next)
    out_r  = Output("r_hat_next",  r_hat_next)

    # (Optional diagnostics; very useful for debugging and for time-optimal tuning)
    out_ax = Output("ax_hat", ax)
    out_ay = Output("ay_hat", ay)
    out_mu = Output("mu_eff", mu_eff)

    # -------------------- Compose Modely and losses --------------------
    model = Modely()
    model.addModel("vx_hat_next", out_vx)
    model.addModel("vy_hat_next", out_vy)
    model.addModel("r_hat_next",  out_r)

    # Optional outputs (still callable after neuralizeModel)
    model.addModel("ax_hat", out_ax)
    model.addModel("ay_hat", out_ay)
    model.addModel("mu_eff", out_mu)

    # Next-step supervision using z(-1) (from nnodely docs). :contentReference[oaicite:4]{index=4}
    model.addMinimize("loss_vx_next", vx.z(-1), out_vx, "mse")
    model.addMinimize("loss_vy_next", vy.z(-1), out_vy, "mse")
    model.addMinimize("loss_r_next",  r.z(-1),  out_r,  "mse")

    # Create the discrete-time MS-NN with sampling time Ts. :contentReference[oaicite:5]{index=5}
    model.neuralizeModel(cfg.Ts)

    return model


# -------------------------- Example training script --------------------------

def main():
    """
    Example usage:
    1) Build model
    2) Load dataset from folder
    3) Train

    Dataset format:
    - Put one or more files in a folder (CSV-like).
    - Provide 'format' array matching columns.
    nnodely loads all files in that folder. :contentReference[oaicite:6]{index=6}
    """

    cfg = CombinedVehicleModelConfig(
        Ts=0.01,
        Tw_u=0.10,
        Tw_delta=0.10,
        Tw_state=0.20,
    )

    model = build_combined_forward_model(cfg)

    # ---- You must adapt these to your logs ----
    data_folder = "./data/vehicle_logs/"  # folder containing csv/txt logs

    # Column order in your files (example).
    # Add more channels if you have them, but keep at least these.
    data_struct = [
        "time",
        "vx", "vy", "r",
        "delta",
        "throttle", "brake",
    ]

    # Load dataset (delimiter depends on your export)
    model.loadData(
        name="train_data",
        source=data_folder,
        format=data_struct,
        delimiter=",",
    )

    # Train with default hyperparameters (you can configure them in nnodely if needed)
    model.trainModel()

    # Quick inference example (needs enough history samples to cover max time window)
    # With Ts=0.01 and Tw_state=0.20, you need at least 20 samples for vx/vy/r.
    sample = {
        "vx": [10.0] * 25,
        "vy": [0.2] * 25,
        "r":  [0.05] * 25,
        "delta": [0.03] * 15,
        "throttle": [0.2] * 15,
        "brake": [0.0] * 15,
    }
    y = model(sample)
    print("Inference keys:", y.keys())
    print("vx_hat_next:", y["vx_hat_next"][-1])
    print("mu_eff:", y["mu_eff"][-1])


if __name__ == "__main__":
    main()