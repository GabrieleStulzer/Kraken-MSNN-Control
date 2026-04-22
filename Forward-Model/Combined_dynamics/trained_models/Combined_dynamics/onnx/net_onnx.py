import torch

def nnodely_basic_model_update_state(data_in, rel):
    data_out = data_in.clone()
    max_dim = min(rel.size(1), data_in.size(1))
    data_out[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return data_out

def nnodely_basic_model_timeshift(data_in):
    return torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)

def nnodely_layers_fuzzify_slicing(res, i, x):
    res[:, :, i:i+1] = x

def nnodely_layers_parametricfunction_long_corr_lat_local(accy,accy_0,  # inputs
                    k1,k2          # learnable parameter 
                  ):
  return k1+ k2*(accy - accy_0) # approsimazione accelerazione laterale linearizzata, scriviamo che l'accelerazione laterale è funzione lineare dell'accelerazione laterale di riferimento

def nnodely_layers_parametricfunction_acc_pos(T):
    mask_pos = torch.gt(T,0)
    return torch.mul(T,mask_pos)

def nnodely_layers_parametricfunction_acc_neg(T):
    mask_pos = torch.gt(T,0)
    return torch.mul(T,~mask_pos)

def nnodely_layers_parametricfunction_acc_model_based(Ty, v,F_y,delta,
                    r1,mass,Kd,Cv,Cr,Iw1):  # learnable parameter
    # function inputs:
    # Ty,v --> wheel torques and vehicle speed
    # F_y --> lateral tire forces
    # delta --> steering angle


    # learnable parameters:
    # r1 --> wheel radius
    # mass --> vehicle mass
    # Kd --> aero drag coefficient
    # Cv --> viscous coefficient
    # Cr --> rolling resistance coefficient
    # Iw1 --> wheel inertia

    # non-trainable parameters
    g_acc     = 9.81       # [m/s^2] gravity acceleration

    # function output: longitudinal acceleration, computed using the Newton's vehicle dynamics laws 
    return ((1.0/mass)*((Ty)/r1 - Kd * v**2 - Cv * v - F_y*torch.sin(delta)) - Cr*g_acc)/(1.0 + (2.0/mass)*(2*(Iw1/r1**2.0)))

def nnodely_layers_parametricfunction_understeer_corr_local(input,vx,input_0, vx_0,ax,ax_0,  # inputs
                    k1,k2,k3,k4          # learnable parameter 
                  ):
  return vx*(k1+ k2*(input - input_0) + k3* (vx-vx_0)+k4*(ax-ax_0)) # approssimazione handling diagram, superfici linearizzate. Non è una singola curva

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self._tensor_constant100 = torch.tensor(1.0)
        self._tensor_constant101 = torch.tensor(1)
        self._tensor_constant102 = torch.tensor(0.0)
        self._tensor_constant103 = torch.tensor(1.0)
        self._tensor_constant104 = torch.tensor(0)
        self._tensor_constant105 = torch.tensor(0.0)
        self._tensor_constant106 = torch.tensor(1.0)
        self._tensor_constant107 = torch.tensor(1)
        self._tensor_constant108 = torch.tensor(0.0)
        self._tensor_constant109 = torch.tensor(1.0)
        self._tensor_constant110 = torch.tensor(0)
        self._tensor_constant111 = torch.tensor(0.0)
        self._tensor_constant112 = torch.tensor(1.0)
        self._tensor_constant113 = torch.tensor(1)
        self._tensor_constant114 = torch.tensor(0.0)
        self._tensor_constant115 = torch.tensor(1.0)
        self._tensor_constant116 = torch.tensor(0)
        self._tensor_constant117 = torch.tensor(0.0)
        self._tensor_constant118 = torch.tensor(0.0)
        self._tensor_constant119 = torch.tensor(1)
        self._tensor_constant120 = torch.tensor(0.0)
        self._tensor_constant121 = torch.tensor(0.0)
        self._tensor_constant122 = torch.tensor(2)
        self._tensor_constant123 = torch.tensor(0.0)
        self._tensor_constant124 = torch.tensor(1.0)
        self._tensor_constant125 = torch.tensor(3)
        self._tensor_constant126 = torch.tensor(0.0)
        self._tensor_constant127 = torch.tensor(1.0)
        self._tensor_constant128 = torch.tensor(0)
        self._tensor_constant129 = torch.tensor(0.0)
        self._tensor_constant130 = torch.tensor(1.0)
        self._tensor_constant131 = torch.tensor(1)
        self._tensor_constant66 = torch.tensor(0.0)
        self._tensor_constant67 = torch.tensor(1.0)
        self._tensor_constant68 = torch.tensor(0)
        self._tensor_constant69 = torch.tensor(0.0)
        self._tensor_constant70 = torch.tensor(0.0)
        self._tensor_constant71 = torch.tensor(1)
        self._tensor_constant72 = torch.tensor(0.0)
        self._tensor_constant73 = torch.tensor(0.0)
        self._tensor_constant74 = torch.tensor(2)
        self._tensor_constant75 = torch.tensor(0.0)
        self._tensor_constant76 = torch.tensor(1.0)
        self._tensor_constant77 = torch.tensor(3)
        self._tensor_constant78 = torch.tensor(0.0)
        self._tensor_constant79 = torch.tensor(1.0)
        self._tensor_constant80 = torch.tensor(0)
        self._tensor_constant81 = torch.tensor(0.0)
        self._tensor_constant82 = torch.tensor(0.0)
        self._tensor_constant83 = torch.tensor(1)
        self._tensor_constant84 = torch.tensor(0.0)
        self._tensor_constant85 = torch.tensor(0.0)
        self._tensor_constant86 = torch.tensor(2)
        self._tensor_constant87 = torch.tensor(0.0)
        self._tensor_constant88 = torch.tensor(0.0)
        self._tensor_constant89 = torch.tensor(3)
        self._tensor_constant90 = torch.tensor(0.0)
        self._tensor_constant91 = torch.tensor(0.0)
        self._tensor_constant92 = torch.tensor(4)
        self._tensor_constant93 = torch.tensor(0.0)
        self._tensor_constant94 = torch.tensor(1.0)
        self._tensor_constant95 = torch.tensor(5)
        self._tensor_constant96 = torch.tensor(0.0)
        self._tensor_constant97 = torch.tensor(1.0)
        self._tensor_constant98 = torch.tensor(0)
        self._tensor_constant99 = torch.tensor(0.0)
        self.all_constants["ax_center_0"] = torch.tensor([-12.6427001953125], requires_grad=False)
        self.all_constants["ax_center_1"] = torch.tensor([8.793100357055664], requires_grad=False)
        self.all_constants["ay_center_0"] = torch.tensor([-16.053800582885742], requires_grad=False)
        self.all_constants["ay_center_1"] = torch.tensor([-9.961919784545898], requires_grad=False)
        self.all_constants["ay_center_2"] = torch.tensor([-3.870039939880371], requires_grad=False)
        self.all_constants["ay_center_3"] = torch.tensor([2.2218399047851562], requires_grad=False)
        self.all_constants["ay_center_4"] = torch.tensor([8.313719749450684], requires_grad=False)
        self.all_constants["ay_center_5"] = torch.tensor([14.405599594116211], requires_grad=False)
        self.all_constants["steer_center_0"] = torch.tensor([-1.6690000295639038], requires_grad=False)
        self.all_constants["steer_center_1"] = torch.tensor([-0.4763000011444092], requires_grad=False)
        self.all_constants["steer_center_2"] = torch.tensor([0.7164000272750854], requires_grad=False)
        self.all_constants["steer_center_3"] = torch.tensor([1.90910005569458], requires_grad=False)
        self.all_constants["vx_center_0"] = torch.tensor([6.065000057220459], requires_grad=False)
        self.all_constants["vx_center_1"] = torch.tensor([24.538999557495117], requires_grad=False)
        self.all_parameters["Cr"] = torch.nn.Parameter(torch.tensor([[0.014999999664723873]]), requires_grad=True)
        self.all_parameters["Cv"] = torch.nn.Parameter(torch.tensor([[2.0]]), requires_grad=True)
        self.all_parameters["Iw1"] = torch.nn.Parameter(torch.tensor([[0.8999999761581421]]), requires_grad=True)
        self.all_parameters["Kd"] = torch.nn.Parameter(torch.tensor([[1.0]]), requires_grad=True)
        self.all_parameters["k1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k1_3"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k1_4"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k1_5"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_3"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_4"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_5"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["mass"] = torch.nn.Parameter(torch.tensor([[200.0]]), requires_grad=True)
        self.all_parameters["r1"] = torch.nn.Parameter(torch.tensor([[0.20000000298023224]]), requires_grad=True)
        self.all_parameters["PFir136W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [8.766285759520542e-07], [1.1405219311200199e-06], [1.4838556126051117e-06], [1.930543930939166e-06], [2.511699676688295e-06], [3.2678019579179818e-06], [4.2515152927080635e-06], [5.531358056032332e-06], [7.196474598458735e-06], [9.362844139104709e-06], [1.21813609439414e-05], [1.5848341718083248e-05], [2.0619203496607952e-05], [2.682624472072348e-05], [3.490180824883282e-05], [4.5408371079247445e-05], [5.9077752666780725e-05], [7.686205208301544e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir138W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [8.766285759520542e-07], [1.1405219311200199e-06], [1.4838556126051117e-06], [1.930543930939166e-06], [2.511699676688295e-06], [3.2678019579179818e-06], [4.2515152927080635e-06], [5.531358056032332e-06], [7.196474598458735e-06], [9.362844139104709e-06], [1.21813609439414e-05], [1.5848341718083248e-05], [2.0619203496607952e-05], [2.682624472072348e-05], [3.490180824883282e-05], [4.5408371079247445e-05], [5.9077752666780725e-05], [7.686205208301544e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir140W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [8.766285759520542e-07], [1.1405219311200199e-06], [1.4838556126051117e-06], [1.930543930939166e-06], [2.511699676688295e-06], [3.2678019579179818e-06], [4.2515152927080635e-06], [5.531358056032332e-06], [7.196474598458735e-06], [9.362844139104709e-06], [1.21813609439414e-05], [1.5848341718083248e-05], [2.0619203496607952e-05], [2.682624472072348e-05], [3.490180824883282e-05], [4.5408371079247445e-05], [5.9077752666780725e-05], [7.686205208301544e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir142W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [8.766285759520542e-07], [1.1405219311200199e-06], [1.4838556126051117e-06], [1.930543930939166e-06], [2.511699676688295e-06], [3.2678019579179818e-06], [4.2515152927080635e-06], [5.531358056032332e-06], [7.196474598458735e-06], [9.362844139104709e-06], [1.21813609439414e-05], [1.5848341718083248e-05], [2.0619203496607952e-05], [2.682624472072348e-05], [3.490180824883282e-05], [4.5408371079247445e-05], [5.9077752666780725e-05], [7.686205208301544e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir15b"] = torch.nn.Parameter(torch.tensor([4.978706783731468e-06, 6.948345344426343e-06, 9.697197128843982e-06, 1.3533528544940054e-05, 1.8887560145230964e-05, 2.6359713956480846e-05, 3.678794382722117e-05, 5.134171078680083e-05, 7.1653128543403e-05, 9.999999747378752e-05]), requires_grad=True)
        self.Fir6 = torch.nn.Dropout(p=0.05)
        self.all_parameters["PFir15W"] = torch.nn.Parameter(torch.tensor([[4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06], [6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06], [7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06], [9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06], [1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05], [1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05], [1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05], [2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05], [2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05], [3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05], [4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05], [5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05], [6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05], [8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05], [9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir48W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir50W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir52W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir20b"] = torch.nn.Parameter(torch.tensor([4.978706783731468e-06, 6.948345344426343e-06, 9.697197128843982e-06, 1.3533528544940054e-05, 1.8887560145230964e-05, 2.6359713956480846e-05, 3.678794382722117e-05, 5.134171078680083e-05, 7.1653128543403e-05, 9.999999747378752e-05]), requires_grad=True)
        self.Fir9 = torch.nn.Dropout(p=0.05)
        self.all_parameters["PFir20W"] = torch.nn.Parameter(torch.tensor([[4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06], [6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06], [7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06], [9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06], [1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05], [1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05], [1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05], [2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05], [2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05], [3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05], [4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05], [5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05], [6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05], [8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05], [9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir54W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PLinear23W"] = torch.nn.Parameter(torch.tensor([[0.19923532009124756], [0.694827139377594], [0.5830032825469971], [0.6318286061286926], [0.5558860898017883], [0.12624889612197876], [0.9790288209915161], [0.8442656397819519], [0.1255868673324585], [0.4456220865249634]]), requires_grad=True)
        self.all_parameters["PLinear18W"] = torch.nn.Parameter(torch.tensor([[0.46568745374679565], [0.23276680707931519], [0.4527209997177124], [0.5871122479438782], [0.40864473581314087], [0.1271744966506958], [0.6372835040092468], [0.2420617938041687], [0.7311904430389404], [0.722437858581543]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart101"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart104"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart107"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], requires_grad=True)
        self.all_constants["SamplePart110"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart112"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart114"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], requires_grad=True)
        self.all_constants["SamplePart13"] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart16"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart19"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart4"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart490"] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart494"] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart496"] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart499"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart501"] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart95"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart98"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], requires_grad=True)
        self.all_constants["Select130"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select131"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select133"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select151"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select152"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select154"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select172"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select173"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select175"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select193"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select194"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select196"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select214"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select215"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select217"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select235"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select236"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select238"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select256"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select257"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select259"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select27"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select277"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select278"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select280"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select298"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select299"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select301"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select319"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select320"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select322"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select340"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select341"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select343"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select36"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select361"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select362"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select364"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select382"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select383"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select385"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select403"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select404"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select406"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select424"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select425"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select427"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select445"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select446"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select448"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select45"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select466"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select467"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select471"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select472"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select476"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select477"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["Select481"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select482"] = torch.tensor([0.0, 1.0], requires_grad=True)
        self.all_constants["Select54"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select63"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select72"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select79"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select82"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select85"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select88"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, delta, vel, accy, torque, acc, yaw_rate):
        getitem = delta
        relation_forward_sample_part496_w = self.all_constants.SamplePart496
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part496_w);  getitem = relation_forward_sample_part496_w = None
        getitem_1 = vel
        relation_forward_sample_part13_w = self.all_constants.SamplePart13
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part13_w);  getitem_1 = relation_forward_sample_part13_w = None
        zeros_like = torch.zeros_like(einsum_1)
        repeat = zeros_like.repeat(1, 1, 4);  zeros_like = None
        sub = einsum_1 - 6.065
        neg = -sub;  sub = None
        truediv = neg / 6.158;  neg = None
        add = truediv + 1;  truediv = None
        _tensor_constant66 = self._tensor_constant66
        maximum = torch.maximum(add, _tensor_constant66);  add = _tensor_constant66 = None
        _tensor_constant67 = self._tensor_constant67
        minimum = torch.minimum(maximum, _tensor_constant67);  maximum = _tensor_constant67 = None
        _tensor_constant68 = self._tensor_constant68
        slicing = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant68, minimum);  _tensor_constant68 = minimum = slicing = None
        sub_1 = einsum_1 - 6.065
        truediv_1 = sub_1 / 6.158;  sub_1 = None
        _tensor_constant69 = self._tensor_constant69
        maximum_1 = torch.maximum(truediv_1, _tensor_constant69);  truediv_1 = _tensor_constant69 = None
        sub_2 = einsum_1 - 12.223
        neg_1 = -sub_2;  sub_2 = None
        truediv_2 = neg_1 / 6.1579999999999995;  neg_1 = None
        add_1 = truediv_2 + 1;  truediv_2 = None
        _tensor_constant70 = self._tensor_constant70
        maximum_2 = torch.maximum(add_1, _tensor_constant70);  add_1 = _tensor_constant70 = None
        minimum_1 = torch.minimum(maximum_1, maximum_2);  maximum_1 = maximum_2 = None
        _tensor_constant71 = self._tensor_constant71
        slicing_1 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant71, minimum_1);  _tensor_constant71 = minimum_1 = slicing_1 = None
        sub_3 = einsum_1 - 12.223
        truediv_3 = sub_3 / 6.1579999999999995;  sub_3 = None
        _tensor_constant72 = self._tensor_constant72
        maximum_3 = torch.maximum(truediv_3, _tensor_constant72);  truediv_3 = _tensor_constant72 = None
        sub_4 = einsum_1 - 18.381
        neg_2 = -sub_4;  sub_4 = None
        truediv_4 = neg_2 / 6.158000000000001;  neg_2 = None
        add_2 = truediv_4 + 1;  truediv_4 = None
        _tensor_constant73 = self._tensor_constant73
        maximum_4 = torch.maximum(add_2, _tensor_constant73);  add_2 = _tensor_constant73 = None
        minimum_2 = torch.minimum(maximum_3, maximum_4);  maximum_3 = maximum_4 = None
        _tensor_constant74 = self._tensor_constant74
        slicing_2 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant74, minimum_2);  _tensor_constant74 = minimum_2 = slicing_2 = None
        sub_5 = einsum_1 - 18.381;  einsum_1 = None
        truediv_5 = sub_5 / 6.158000000000001;  sub_5 = None
        _tensor_constant75 = self._tensor_constant75
        maximum_5 = torch.maximum(truediv_5, _tensor_constant75);  truediv_5 = _tensor_constant75 = None
        _tensor_constant76 = self._tensor_constant76
        minimum_3 = torch.minimum(maximum_5, _tensor_constant76);  maximum_5 = _tensor_constant76 = None
        _tensor_constant77 = self._tensor_constant77
        slicing_3 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant77, minimum_3);  _tensor_constant77 = minimum_3 = slicing_3 = None
        relation_forward_select88_w = self.all_constants.Select88
        einsum_2 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select88_w);  relation_forward_select88_w = None
        unsqueeze = einsum_2.unsqueeze(2);  einsum_2 = None
        getitem_2 = accy
        relation_forward_sample_part16_w = self.all_constants.SamplePart16
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part16_w);  getitem_2 = relation_forward_sample_part16_w = None
        zeros_like_1 = torch.zeros_like(einsum_3)
        repeat_1 = zeros_like_1.repeat(1, 1, 6);  zeros_like_1 = None
        sub_6 = einsum_3 - -16.0538
        neg_3 = -sub_6;  sub_6 = None
        truediv_6 = neg_3 / 6.09188;  neg_3 = None
        add_3 = truediv_6 + 1;  truediv_6 = None
        _tensor_constant78 = self._tensor_constant78
        maximum_6 = torch.maximum(add_3, _tensor_constant78);  add_3 = _tensor_constant78 = None
        _tensor_constant79 = self._tensor_constant79
        minimum_4 = torch.minimum(maximum_6, _tensor_constant79);  maximum_6 = _tensor_constant79 = None
        _tensor_constant80 = self._tensor_constant80
        slicing_4 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant80, minimum_4);  _tensor_constant80 = minimum_4 = slicing_4 = None
        sub_7 = einsum_3 - -16.0538
        truediv_7 = sub_7 / 6.09188;  sub_7 = None
        _tensor_constant81 = self._tensor_constant81
        maximum_7 = torch.maximum(truediv_7, _tensor_constant81);  truediv_7 = _tensor_constant81 = None
        sub_8 = einsum_3 - -9.96192
        neg_4 = -sub_8;  sub_8 = None
        truediv_8 = neg_4 / 6.09188;  neg_4 = None
        add_4 = truediv_8 + 1;  truediv_8 = None
        _tensor_constant82 = self._tensor_constant82
        maximum_8 = torch.maximum(add_4, _tensor_constant82);  add_4 = _tensor_constant82 = None
        minimum_5 = torch.minimum(maximum_7, maximum_8);  maximum_7 = maximum_8 = None
        _tensor_constant83 = self._tensor_constant83
        slicing_5 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant83, minimum_5);  _tensor_constant83 = minimum_5 = slicing_5 = None
        sub_9 = einsum_3 - -9.96192
        truediv_9 = sub_9 / 6.09188;  sub_9 = None
        _tensor_constant84 = self._tensor_constant84
        maximum_9 = torch.maximum(truediv_9, _tensor_constant84);  truediv_9 = _tensor_constant84 = None
        sub_10 = einsum_3 - -3.8700399999999995
        neg_5 = -sub_10;  sub_10 = None
        truediv_10 = neg_5 / 6.09188;  neg_5 = None
        add_5 = truediv_10 + 1;  truediv_10 = None
        _tensor_constant85 = self._tensor_constant85
        maximum_10 = torch.maximum(add_5, _tensor_constant85);  add_5 = _tensor_constant85 = None
        minimum_6 = torch.minimum(maximum_9, maximum_10);  maximum_9 = maximum_10 = None
        _tensor_constant86 = self._tensor_constant86
        slicing_6 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant86, minimum_6);  _tensor_constant86 = minimum_6 = slicing_6 = None
        sub_11 = einsum_3 - -3.8700399999999995
        truediv_11 = sub_11 / 6.09188;  sub_11 = None
        _tensor_constant87 = self._tensor_constant87
        maximum_11 = torch.maximum(truediv_11, _tensor_constant87);  truediv_11 = _tensor_constant87 = None
        sub_12 = einsum_3 - 2.2218400000000003
        neg_6 = -sub_12;  sub_12 = None
        truediv_12 = neg_6 / 6.09188;  neg_6 = None
        add_6 = truediv_12 + 1;  truediv_12 = None
        _tensor_constant88 = self._tensor_constant88
        maximum_12 = torch.maximum(add_6, _tensor_constant88);  add_6 = _tensor_constant88 = None
        minimum_7 = torch.minimum(maximum_11, maximum_12);  maximum_11 = maximum_12 = None
        _tensor_constant89 = self._tensor_constant89
        slicing_7 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant89, minimum_7);  _tensor_constant89 = minimum_7 = slicing_7 = None
        sub_13 = einsum_3 - 2.2218400000000003
        truediv_13 = sub_13 / 6.09188;  sub_13 = None
        _tensor_constant90 = self._tensor_constant90
        maximum_13 = torch.maximum(truediv_13, _tensor_constant90);  truediv_13 = _tensor_constant90 = None
        sub_14 = einsum_3 - 8.31372
        neg_7 = -sub_14;  sub_14 = None
        truediv_14 = neg_7 / 6.09188;  neg_7 = None
        add_7 = truediv_14 + 1;  truediv_14 = None
        _tensor_constant91 = self._tensor_constant91
        maximum_14 = torch.maximum(add_7, _tensor_constant91);  add_7 = _tensor_constant91 = None
        minimum_8 = torch.minimum(maximum_13, maximum_14);  maximum_13 = maximum_14 = None
        _tensor_constant92 = self._tensor_constant92
        slicing_8 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant92, minimum_8);  _tensor_constant92 = minimum_8 = slicing_8 = None
        sub_15 = einsum_3 - 8.31372;  einsum_3 = None
        truediv_15 = sub_15 / 6.09188;  sub_15 = None
        _tensor_constant93 = self._tensor_constant93
        maximum_15 = torch.maximum(truediv_15, _tensor_constant93);  truediv_15 = _tensor_constant93 = None
        _tensor_constant94 = self._tensor_constant94
        minimum_9 = torch.minimum(maximum_15, _tensor_constant94);  maximum_15 = _tensor_constant94 = None
        _tensor_constant95 = self._tensor_constant95
        slicing_9 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant95, minimum_9);  _tensor_constant95 = minimum_9 = slicing_9 = None
        relation_forward_select72_w = self.all_constants.Select72
        einsum_4 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select72_w);  relation_forward_select72_w = None
        unsqueeze_1 = einsum_4.unsqueeze(2);  einsum_4 = None
        getitem_3 = accy
        relation_forward_sample_part19_w = self.all_constants.SamplePart19
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part19_w);  getitem_3 = relation_forward_sample_part19_w = None
        all_constants_ay_center_5 = self.all_constants.ay_center_5
        all_parameters_k1_5 = self.all_parameters.k1_5
        all_parameters_k2_5 = self.all_parameters.k2_5
        long_corr_lat_local = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_5, all_parameters_k1_5, all_parameters_k2_5);  all_constants_ay_center_5 = all_parameters_k1_5 = all_parameters_k2_5 = None
        mul = long_corr_lat_local * unsqueeze_1;  long_corr_lat_local = unsqueeze_1 = None
        relation_forward_select63_w = self.all_constants.Select63
        einsum_6 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select63_w);  relation_forward_select63_w = None
        unsqueeze_2 = einsum_6.unsqueeze(2);  einsum_6 = None
        all_constants_ay_center_4 = self.all_constants.ay_center_4
        all_parameters_k1_4 = self.all_parameters.k1_4
        all_parameters_k2_4 = self.all_parameters.k2_4
        long_corr_lat_local_1 = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_4, all_parameters_k1_4, all_parameters_k2_4);  all_constants_ay_center_4 = all_parameters_k1_4 = all_parameters_k2_4 = None
        mul_1 = long_corr_lat_local_1 * unsqueeze_2;  long_corr_lat_local_1 = unsqueeze_2 = None
        relation_forward_select54_w = self.all_constants.Select54
        einsum_7 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select54_w);  relation_forward_select54_w = None
        unsqueeze_3 = einsum_7.unsqueeze(2);  einsum_7 = None
        all_constants_ay_center_3 = self.all_constants.ay_center_3
        all_parameters_k1_3 = self.all_parameters.k1_3
        all_parameters_k2_3 = self.all_parameters.k2_3
        long_corr_lat_local_2 = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_3, all_parameters_k1_3, all_parameters_k2_3);  all_constants_ay_center_3 = all_parameters_k1_3 = all_parameters_k2_3 = None
        mul_2 = long_corr_lat_local_2 * unsqueeze_3;  long_corr_lat_local_2 = unsqueeze_3 = None
        relation_forward_select45_w = self.all_constants.Select45
        einsum_8 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select45_w);  relation_forward_select45_w = None
        unsqueeze_4 = einsum_8.unsqueeze(2);  einsum_8 = None
        all_constants_ay_center_2 = self.all_constants.ay_center_2
        all_parameters_k1_2 = self.all_parameters.k1_2
        all_parameters_k2_2 = self.all_parameters.k2_2
        long_corr_lat_local_3 = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_2, all_parameters_k1_2, all_parameters_k2_2);  all_constants_ay_center_2 = all_parameters_k1_2 = all_parameters_k2_2 = None
        mul_3 = long_corr_lat_local_3 * unsqueeze_4;  long_corr_lat_local_3 = unsqueeze_4 = None
        relation_forward_select36_w = self.all_constants.Select36
        einsum_9 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select36_w);  relation_forward_select36_w = None
        unsqueeze_5 = einsum_9.unsqueeze(2);  einsum_9 = None
        all_constants_ay_center_1 = self.all_constants.ay_center_1
        all_parameters_k1_1 = self.all_parameters.k1_1
        all_parameters_k2_1 = self.all_parameters.k2_1
        long_corr_lat_local_4 = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_1, all_parameters_k1_1, all_parameters_k2_1);  all_constants_ay_center_1 = all_parameters_k1_1 = all_parameters_k2_1 = None
        mul_4 = long_corr_lat_local_4 * unsqueeze_5;  long_corr_lat_local_4 = unsqueeze_5 = None
        relation_forward_select27_w = self.all_constants.Select27
        einsum_10 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select27_w);  repeat_1 = relation_forward_select27_w = None
        unsqueeze_6 = einsum_10.unsqueeze(2);  einsum_10 = None
        all_constants_ay_center_0 = self.all_constants.ay_center_0
        all_parameters_k1_0 = self.all_parameters.k1_0
        all_parameters_k2_0 = self.all_parameters.k2_0
        long_corr_lat_local_5 = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_0, all_parameters_k1_0, all_parameters_k2_0);  einsum_5 = all_constants_ay_center_0 = all_parameters_k1_0 = all_parameters_k2_0 = None
        mul_5 = long_corr_lat_local_5 * unsqueeze_6;  long_corr_lat_local_5 = unsqueeze_6 = None
        add_8 = mul_5 + mul_4;  mul_5 = mul_4 = None
        add_9 = add_8 + mul_3;  add_8 = mul_3 = None
        add_10 = add_9 + mul_2;  add_9 = mul_2 = None
        add_11 = add_10 + mul_1;  add_10 = mul_1 = None
        add_12 = add_11 + mul;  add_11 = mul = None
        mul_6 = add_12 * unsqueeze;  unsqueeze = None
        size = mul_6.size(0)
        relation_forward_fir90_weights = self.all_parameters.PFir54W
        size_1 = relation_forward_fir90_weights.size(1)
        squeeze = mul_6.squeeze(-1);  mul_6 = None
        matmul = torch.matmul(squeeze, relation_forward_fir90_weights);  squeeze = relation_forward_fir90_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        relation_forward_select85_w = self.all_constants.Select85
        einsum_11 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select85_w);  relation_forward_select85_w = None
        unsqueeze_7 = einsum_11.unsqueeze(2);  einsum_11 = None
        mul_7 = add_12 * unsqueeze_7;  unsqueeze_7 = None
        size_2 = mul_7.size(0)
        relation_forward_fir87_weights = self.all_parameters.PFir52W
        size_3 = relation_forward_fir87_weights.size(1)
        squeeze_1 = mul_7.squeeze(-1);  mul_7 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir87_weights);  squeeze_1 = relation_forward_fir87_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        relation_forward_select82_w = self.all_constants.Select82
        einsum_12 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select82_w);  relation_forward_select82_w = None
        unsqueeze_8 = einsum_12.unsqueeze(2);  einsum_12 = None
        mul_8 = add_12 * unsqueeze_8;  unsqueeze_8 = None
        size_4 = mul_8.size(0)
        relation_forward_fir84_weights = self.all_parameters.PFir50W
        size_5 = relation_forward_fir84_weights.size(1)
        squeeze_2 = mul_8.squeeze(-1);  mul_8 = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir84_weights);  squeeze_2 = relation_forward_fir84_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        relation_forward_select79_w = self.all_constants.Select79
        einsum_13 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select79_w);  repeat = relation_forward_select79_w = None
        unsqueeze_9 = einsum_13.unsqueeze(2);  einsum_13 = None
        mul_9 = add_12 * unsqueeze_9;  add_12 = unsqueeze_9 = None
        size_6 = mul_9.size(0)
        relation_forward_fir81_weights = self.all_parameters.PFir48W
        size_7 = relation_forward_fir81_weights.size(1)
        squeeze_3 = mul_9.squeeze(-1);  mul_9 = None
        matmul_3 = torch.matmul(squeeze_3, relation_forward_fir81_weights);  squeeze_3 = relation_forward_fir81_weights = None
        to_3 = matmul_3.to(dtype = torch.float32);  matmul_3 = None
        view_3 = to_3.view(size_6, 1, size_7);  to_3 = size_6 = size_7 = None
        add_13 = view_3 + view_2;  view_3 = view_2 = None
        add_14 = add_13 + view_1;  add_13 = view_1 = None
        add_15 = add_14 + view;  add_14 = view = None
        getitem_4 = vel
        relation_forward_sample_part494_w = self.all_constants.SamplePart494
        einsum_14 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part494_w);  getitem_4 = relation_forward_sample_part494_w = None
        getitem_5 = torque
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum_15 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part1_w);  getitem_5 = relation_forward_sample_part1_w = None
        acc_pos = nnodely_layers_parametricfunction_acc_pos(einsum_15);  einsum_15 = None
        size_8 = acc_pos.size(0)
        relation_forward_fir6_weights = self.all_parameters.PFir15W
        size_9 = relation_forward_fir6_weights.size(1)
        squeeze_4 = acc_pos.squeeze(-1);  acc_pos = None
        matmul_4 = torch.matmul(squeeze_4, relation_forward_fir6_weights);  squeeze_4 = relation_forward_fir6_weights = None
        to_4 = matmul_4.to(dtype = torch.float32);  matmul_4 = None
        view_4 = to_4.view(size_8, 1, size_9);  to_4 = size_8 = size_9 = None
        relation_forward_fir6_bias = self.all_parameters.PFir15b
        add_16 = view_4 + relation_forward_fir6_bias;  view_4 = relation_forward_fir6_bias = None
        relation_forward_fir6_dropout = self.Fir6(add_16);  add_16 = None
        tanh = torch.tanh(relation_forward_fir6_dropout);  relation_forward_fir6_dropout = None
        relation_forward_linear8_weights = self.all_parameters.PLinear18W
        einsum_16 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear8_weights);  tanh = relation_forward_linear8_weights = None
        getitem_6 = torque
        relation_forward_sample_part4_w = self.all_constants.SamplePart4
        einsum_17 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part4_w);  getitem_6 = relation_forward_sample_part4_w = None
        acc_neg = nnodely_layers_parametricfunction_acc_neg(einsum_17);  einsum_17 = None
        size_10 = acc_neg.size(0)
        relation_forward_fir9_weights = self.all_parameters.PFir20W
        size_11 = relation_forward_fir9_weights.size(1)
        squeeze_5 = acc_neg.squeeze(-1);  acc_neg = None
        matmul_5 = torch.matmul(squeeze_5, relation_forward_fir9_weights);  squeeze_5 = relation_forward_fir9_weights = None
        to_5 = matmul_5.to(dtype = torch.float32);  matmul_5 = None
        view_5 = to_5.view(size_10, 1, size_11);  to_5 = size_10 = size_11 = None
        relation_forward_fir9_bias = self.all_parameters.PFir20b
        add_17 = view_5 + relation_forward_fir9_bias;  view_5 = relation_forward_fir9_bias = None
        relation_forward_fir9_dropout = self.Fir9(add_17);  add_17 = None
        tanh_1 = torch.tanh(relation_forward_fir9_dropout);  relation_forward_fir9_dropout = None
        relation_forward_linear11_weights = self.all_parameters.PLinear23W
        einsum_18 = torch.functional.einsum('bwi,io->bwo', tanh_1, relation_forward_linear11_weights);  tanh_1 = relation_forward_linear11_weights = None
        add_18 = einsum_18 + einsum_16;  einsum_18 = einsum_16 = None
        all_parameters_r1 = self.all_parameters.r1
        all_parameters_mass = self.all_parameters.mass
        all_parameters_kd = self.all_parameters.Kd
        all_parameters_cv = self.all_parameters.Cv
        all_parameters_cr = self.all_parameters.Cr
        all_parameters_iw1 = self.all_parameters.Iw1
        acc_model_based = nnodely_layers_parametricfunction_acc_model_based(add_18, einsum_14, add_15, einsum, all_parameters_r1, all_parameters_mass, all_parameters_kd, all_parameters_cv, all_parameters_cr, all_parameters_iw1);  add_18 = einsum_14 = add_15 = einsum = all_parameters_r1 = all_parameters_mass = all_parameters_kd = all_parameters_cv = all_parameters_cr = all_parameters_iw1 = None
        getitem_7 = acc
        relation_forward_sample_part501_w = self.all_constants.SamplePart501
        einsum_19 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part501_w);  getitem_7 = relation_forward_sample_part501_w = None
        getitem_8 = acc
        relation_forward_sample_part98_w = self.all_constants.SamplePart98
        einsum_20 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part98_w);  getitem_8 = relation_forward_sample_part98_w = None
        zeros_like_2 = torch.zeros_like(einsum_20)
        repeat_2 = zeros_like_2.repeat(1, 1, 2);  zeros_like_2 = None
        sub_16 = einsum_20 - -12.6427
        neg_8 = -sub_16;  sub_16 = None
        truediv_16 = neg_8 / 21.4358;  neg_8 = None
        add_19 = truediv_16 + 1;  truediv_16 = None
        _tensor_constant96 = self._tensor_constant96
        maximum_16 = torch.maximum(add_19, _tensor_constant96);  add_19 = _tensor_constant96 = None
        _tensor_constant97 = self._tensor_constant97
        minimum_10 = torch.minimum(maximum_16, _tensor_constant97);  maximum_16 = _tensor_constant97 = None
        _tensor_constant98 = self._tensor_constant98
        slicing_10 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant98, minimum_10);  _tensor_constant98 = minimum_10 = slicing_10 = None
        sub_17 = einsum_20 - -12.6427;  einsum_20 = None
        truediv_17 = sub_17 / 21.4358;  sub_17 = None
        _tensor_constant99 = self._tensor_constant99
        maximum_17 = torch.maximum(truediv_17, _tensor_constant99);  truediv_17 = _tensor_constant99 = None
        _tensor_constant100 = self._tensor_constant100
        minimum_11 = torch.minimum(maximum_17, _tensor_constant100);  maximum_17 = _tensor_constant100 = None
        _tensor_constant101 = self._tensor_constant101
        slicing_11 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant101, minimum_11);  _tensor_constant101 = minimum_11 = slicing_11 = None
        relation_forward_select482_w = self.all_constants.Select482
        einsum_21 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select482_w);  relation_forward_select482_w = None
        unsqueeze_10 = einsum_21.unsqueeze(2);  einsum_21 = None
        getitem_9 = vel
        relation_forward_sample_part95_w = self.all_constants.SamplePart95
        einsum_22 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part95_w);  getitem_9 = relation_forward_sample_part95_w = None
        zeros_like_3 = torch.zeros_like(einsum_22)
        repeat_3 = zeros_like_3.repeat(1, 1, 2);  zeros_like_3 = None
        sub_18 = einsum_22 - 6.065
        neg_9 = -sub_18;  sub_18 = None
        truediv_18 = neg_9 / 18.474;  neg_9 = None
        add_20 = truediv_18 + 1;  truediv_18 = None
        _tensor_constant102 = self._tensor_constant102
        maximum_18 = torch.maximum(add_20, _tensor_constant102);  add_20 = _tensor_constant102 = None
        _tensor_constant103 = self._tensor_constant103
        minimum_12 = torch.minimum(maximum_18, _tensor_constant103);  maximum_18 = _tensor_constant103 = None
        _tensor_constant104 = self._tensor_constant104
        slicing_12 = nnodely_layers_fuzzify_slicing(repeat_3, _tensor_constant104, minimum_12);  _tensor_constant104 = minimum_12 = slicing_12 = None
        sub_19 = einsum_22 - 6.065;  einsum_22 = None
        truediv_19 = sub_19 / 18.474;  sub_19 = None
        _tensor_constant105 = self._tensor_constant105
        maximum_19 = torch.maximum(truediv_19, _tensor_constant105);  truediv_19 = _tensor_constant105 = None
        _tensor_constant106 = self._tensor_constant106
        minimum_13 = torch.minimum(maximum_19, _tensor_constant106);  maximum_19 = _tensor_constant106 = None
        _tensor_constant107 = self._tensor_constant107
        slicing_13 = nnodely_layers_fuzzify_slicing(repeat_3, _tensor_constant107, minimum_13);  _tensor_constant107 = minimum_13 = slicing_13 = None
        relation_forward_select481_w = self.all_constants.Select481
        einsum_23 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select481_w);  relation_forward_select481_w = None
        unsqueeze_11 = einsum_23.unsqueeze(2);  einsum_23 = None
        mul_10 = unsqueeze_11 * unsqueeze_10;  unsqueeze_11 = unsqueeze_10 = None
        getitem_10 = acc
        relation_forward_sample_part107_w = self.all_constants.SamplePart107
        einsum_24 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part107_w);  getitem_10 = relation_forward_sample_part107_w = None
        zeros_like_4 = torch.zeros_like(einsum_24)
        repeat_4 = zeros_like_4.repeat(1, 1, 2);  zeros_like_4 = None
        sub_20 = einsum_24 - -12.6427
        neg_10 = -sub_20;  sub_20 = None
        truediv_20 = neg_10 / 21.4358;  neg_10 = None
        add_21 = truediv_20 + 1;  truediv_20 = None
        _tensor_constant108 = self._tensor_constant108
        maximum_20 = torch.maximum(add_21, _tensor_constant108);  add_21 = _tensor_constant108 = None
        _tensor_constant109 = self._tensor_constant109
        minimum_14 = torch.minimum(maximum_20, _tensor_constant109);  maximum_20 = _tensor_constant109 = None
        _tensor_constant110 = self._tensor_constant110
        slicing_14 = nnodely_layers_fuzzify_slicing(repeat_4, _tensor_constant110, minimum_14);  _tensor_constant110 = minimum_14 = slicing_14 = None
        sub_21 = einsum_24 - -12.6427;  einsum_24 = None
        truediv_21 = sub_21 / 21.4358;  sub_21 = None
        _tensor_constant111 = self._tensor_constant111
        maximum_21 = torch.maximum(truediv_21, _tensor_constant111);  truediv_21 = _tensor_constant111 = None
        _tensor_constant112 = self._tensor_constant112
        minimum_15 = torch.minimum(maximum_21, _tensor_constant112);  maximum_21 = _tensor_constant112 = None
        _tensor_constant113 = self._tensor_constant113
        slicing_15 = nnodely_layers_fuzzify_slicing(repeat_4, _tensor_constant113, minimum_15);  _tensor_constant113 = minimum_15 = slicing_15 = None
        relation_forward_select448_w = self.all_constants.Select448
        einsum_25 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select448_w);  relation_forward_select448_w = None
        unsqueeze_12 = einsum_25.unsqueeze(2);  einsum_25 = None
        getitem_11 = delta
        relation_forward_sample_part101_w = self.all_constants.SamplePart101
        einsum_26 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part101_w);  getitem_11 = relation_forward_sample_part101_w = None
        zeros_like_5 = torch.zeros_like(einsum_26)
        repeat_5 = zeros_like_5.repeat(1, 1, 4);  zeros_like_5 = None
        sub_22 = einsum_26 - -1.669
        neg_11 = -sub_22;  sub_22 = None
        truediv_22 = neg_11 / 1.1927;  neg_11 = None
        add_22 = truediv_22 + 1;  truediv_22 = None
        _tensor_constant114 = self._tensor_constant114
        maximum_22 = torch.maximum(add_22, _tensor_constant114);  add_22 = _tensor_constant114 = None
        _tensor_constant115 = self._tensor_constant115
        minimum_16 = torch.minimum(maximum_22, _tensor_constant115);  maximum_22 = _tensor_constant115 = None
        _tensor_constant116 = self._tensor_constant116
        slicing_16 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant116, minimum_16);  _tensor_constant116 = minimum_16 = slicing_16 = None
        sub_23 = einsum_26 - -1.669
        truediv_23 = sub_23 / 1.1927;  sub_23 = None
        _tensor_constant117 = self._tensor_constant117
        maximum_23 = torch.maximum(truediv_23, _tensor_constant117);  truediv_23 = _tensor_constant117 = None
        sub_24 = einsum_26 - -0.47629999999999995
        neg_12 = -sub_24;  sub_24 = None
        truediv_24 = neg_12 / 1.1927;  neg_12 = None
        add_23 = truediv_24 + 1;  truediv_24 = None
        _tensor_constant118 = self._tensor_constant118
        maximum_24 = torch.maximum(add_23, _tensor_constant118);  add_23 = _tensor_constant118 = None
        minimum_17 = torch.minimum(maximum_23, maximum_24);  maximum_23 = maximum_24 = None
        _tensor_constant119 = self._tensor_constant119
        slicing_17 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant119, minimum_17);  _tensor_constant119 = minimum_17 = slicing_17 = None
        sub_25 = einsum_26 - -0.47629999999999995
        truediv_25 = sub_25 / 1.1927;  sub_25 = None
        _tensor_constant120 = self._tensor_constant120
        maximum_25 = torch.maximum(truediv_25, _tensor_constant120);  truediv_25 = _tensor_constant120 = None
        sub_26 = einsum_26 - 0.7164000000000001
        neg_13 = -sub_26;  sub_26 = None
        truediv_26 = neg_13 / 1.1926999999999999;  neg_13 = None
        add_24 = truediv_26 + 1;  truediv_26 = None
        _tensor_constant121 = self._tensor_constant121
        maximum_26 = torch.maximum(add_24, _tensor_constant121);  add_24 = _tensor_constant121 = None
        minimum_18 = torch.minimum(maximum_25, maximum_26);  maximum_25 = maximum_26 = None
        _tensor_constant122 = self._tensor_constant122
        slicing_18 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant122, minimum_18);  _tensor_constant122 = minimum_18 = slicing_18 = None
        sub_27 = einsum_26 - 0.7164000000000001;  einsum_26 = None
        truediv_27 = sub_27 / 1.1926999999999999;  sub_27 = None
        _tensor_constant123 = self._tensor_constant123
        maximum_27 = torch.maximum(truediv_27, _tensor_constant123);  truediv_27 = _tensor_constant123 = None
        _tensor_constant124 = self._tensor_constant124
        minimum_19 = torch.minimum(maximum_27, _tensor_constant124);  maximum_27 = _tensor_constant124 = None
        _tensor_constant125 = self._tensor_constant125
        slicing_19 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant125, minimum_19);  _tensor_constant125 = minimum_19 = slicing_19 = None
        relation_forward_select446_w = self.all_constants.Select446
        einsum_27 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select446_w);  relation_forward_select446_w = None
        unsqueeze_13 = einsum_27.unsqueeze(2);  einsum_27 = None
        getitem_12 = vel
        relation_forward_sample_part104_w = self.all_constants.SamplePart104
        einsum_28 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part104_w);  getitem_12 = relation_forward_sample_part104_w = None
        zeros_like_6 = torch.zeros_like(einsum_28)
        repeat_6 = zeros_like_6.repeat(1, 1, 2);  zeros_like_6 = None
        sub_28 = einsum_28 - 6.065
        neg_14 = -sub_28;  sub_28 = None
        truediv_28 = neg_14 / 18.474;  neg_14 = None
        add_25 = truediv_28 + 1;  truediv_28 = None
        _tensor_constant126 = self._tensor_constant126
        maximum_28 = torch.maximum(add_25, _tensor_constant126);  add_25 = _tensor_constant126 = None
        _tensor_constant127 = self._tensor_constant127
        minimum_20 = torch.minimum(maximum_28, _tensor_constant127);  maximum_28 = _tensor_constant127 = None
        _tensor_constant128 = self._tensor_constant128
        slicing_20 = nnodely_layers_fuzzify_slicing(repeat_6, _tensor_constant128, minimum_20);  _tensor_constant128 = minimum_20 = slicing_20 = None
        sub_29 = einsum_28 - 6.065;  einsum_28 = None
        truediv_29 = sub_29 / 18.474;  sub_29 = None
        _tensor_constant129 = self._tensor_constant129
        maximum_29 = torch.maximum(truediv_29, _tensor_constant129);  truediv_29 = _tensor_constant129 = None
        _tensor_constant130 = self._tensor_constant130
        minimum_21 = torch.minimum(maximum_29, _tensor_constant130);  maximum_29 = _tensor_constant130 = None
        _tensor_constant131 = self._tensor_constant131
        slicing_21 = nnodely_layers_fuzzify_slicing(repeat_6, _tensor_constant131, minimum_21);  _tensor_constant131 = minimum_21 = slicing_21 = None
        relation_forward_select445_w = self.all_constants.Select445
        einsum_29 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select445_w);  relation_forward_select445_w = None
        unsqueeze_14 = einsum_29.unsqueeze(2);  einsum_29 = None
        mul_11 = unsqueeze_14 * unsqueeze_13;  unsqueeze_14 = unsqueeze_13 = None
        mul_12 = mul_11 * unsqueeze_12;  mul_11 = unsqueeze_12 = None
        getitem_13 = acc
        relation_forward_sample_part114_w = self.all_constants.SamplePart114
        einsum_30 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part114_w);  getitem_13 = relation_forward_sample_part114_w = None
        getitem_14 = vel
        relation_forward_sample_part112_w = self.all_constants.SamplePart112
        einsum_31 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part112_w);  getitem_14 = relation_forward_sample_part112_w = None
        getitem_15 = delta
        relation_forward_sample_part110_w = self.all_constants.SamplePart110
        einsum_32 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part110_w);  getitem_15 = relation_forward_sample_part110_w = None
        all_constants_steer_center_3 = self.all_constants.steer_center_3
        all_constants_vx_center_1 = self.all_constants.vx_center_1
        all_constants_ax_center_1 = self.all_constants.ax_center_1
        all_parameters_lat_k1_1_3_1 = self.all_parameters.lat_k1_1_3_1
        all_parameters_lat_k2_1_3_1 = self.all_parameters.lat_k2_1_3_1
        all_parameters_lat_k3_1_3_1 = self.all_parameters.lat_k3_1_3_1
        all_parameters_lat_k4_1_3_1 = self.all_parameters.lat_k4_1_3_1
        understeer_corr_local = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_3, all_constants_vx_center_1, einsum_30, all_constants_ax_center_1, all_parameters_lat_k1_1_3_1, all_parameters_lat_k2_1_3_1, all_parameters_lat_k3_1_3_1, all_parameters_lat_k4_1_3_1);  all_parameters_lat_k1_1_3_1 = all_parameters_lat_k2_1_3_1 = all_parameters_lat_k3_1_3_1 = all_parameters_lat_k4_1_3_1 = None
        mul_13 = understeer_corr_local * mul_12;  understeer_corr_local = mul_12 = None
        relation_forward_select427_w = self.all_constants.Select427
        einsum_33 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select427_w);  relation_forward_select427_w = None
        unsqueeze_15 = einsum_33.unsqueeze(2);  einsum_33 = None
        relation_forward_select425_w = self.all_constants.Select425
        einsum_34 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select425_w);  relation_forward_select425_w = None
        unsqueeze_16 = einsum_34.unsqueeze(2);  einsum_34 = None
        relation_forward_select424_w = self.all_constants.Select424
        einsum_35 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select424_w);  relation_forward_select424_w = None
        unsqueeze_17 = einsum_35.unsqueeze(2);  einsum_35 = None
        mul_14 = unsqueeze_17 * unsqueeze_16;  unsqueeze_17 = unsqueeze_16 = None
        mul_15 = mul_14 * unsqueeze_15;  mul_14 = unsqueeze_15 = None
        all_constants_ax_center_0 = self.all_constants.ax_center_0
        all_parameters_lat_k1_1_3_0 = self.all_parameters.lat_k1_1_3_0
        all_parameters_lat_k2_1_3_0 = self.all_parameters.lat_k2_1_3_0
        all_parameters_lat_k3_1_3_0 = self.all_parameters.lat_k3_1_3_0
        all_parameters_lat_k4_1_3_0 = self.all_parameters.lat_k4_1_3_0
        understeer_corr_local_1 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_3, all_constants_vx_center_1, einsum_30, all_constants_ax_center_0, all_parameters_lat_k1_1_3_0, all_parameters_lat_k2_1_3_0, all_parameters_lat_k3_1_3_0, all_parameters_lat_k4_1_3_0);  all_parameters_lat_k1_1_3_0 = all_parameters_lat_k2_1_3_0 = all_parameters_lat_k3_1_3_0 = all_parameters_lat_k4_1_3_0 = None
        mul_16 = understeer_corr_local_1 * mul_15;  understeer_corr_local_1 = mul_15 = None
        relation_forward_select406_w = self.all_constants.Select406
        einsum_36 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select406_w);  relation_forward_select406_w = None
        unsqueeze_18 = einsum_36.unsqueeze(2);  einsum_36 = None
        relation_forward_select404_w = self.all_constants.Select404
        einsum_37 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select404_w);  relation_forward_select404_w = None
        unsqueeze_19 = einsum_37.unsqueeze(2);  einsum_37 = None
        relation_forward_select403_w = self.all_constants.Select403
        einsum_38 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select403_w);  relation_forward_select403_w = None
        unsqueeze_20 = einsum_38.unsqueeze(2);  einsum_38 = None
        mul_17 = unsqueeze_20 * unsqueeze_19;  unsqueeze_20 = unsqueeze_19 = None
        mul_18 = mul_17 * unsqueeze_18;  mul_17 = unsqueeze_18 = None
        all_constants_steer_center_2 = self.all_constants.steer_center_2
        all_parameters_lat_k1_1_2_1 = self.all_parameters.lat_k1_1_2_1
        all_parameters_lat_k2_1_2_1 = self.all_parameters.lat_k2_1_2_1
        all_parameters_lat_k3_1_2_1 = self.all_parameters.lat_k3_1_2_1
        all_parameters_lat_k4_1_2_1 = self.all_parameters.lat_k4_1_2_1
        understeer_corr_local_2 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_2, all_constants_vx_center_1, einsum_30, all_constants_ax_center_1, all_parameters_lat_k1_1_2_1, all_parameters_lat_k2_1_2_1, all_parameters_lat_k3_1_2_1, all_parameters_lat_k4_1_2_1);  all_parameters_lat_k1_1_2_1 = all_parameters_lat_k2_1_2_1 = all_parameters_lat_k3_1_2_1 = all_parameters_lat_k4_1_2_1 = None
        mul_19 = understeer_corr_local_2 * mul_18;  understeer_corr_local_2 = mul_18 = None
        relation_forward_select385_w = self.all_constants.Select385
        einsum_39 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select385_w);  relation_forward_select385_w = None
        unsqueeze_21 = einsum_39.unsqueeze(2);  einsum_39 = None
        relation_forward_select383_w = self.all_constants.Select383
        einsum_40 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select383_w);  relation_forward_select383_w = None
        unsqueeze_22 = einsum_40.unsqueeze(2);  einsum_40 = None
        relation_forward_select382_w = self.all_constants.Select382
        einsum_41 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select382_w);  relation_forward_select382_w = None
        unsqueeze_23 = einsum_41.unsqueeze(2);  einsum_41 = None
        mul_20 = unsqueeze_23 * unsqueeze_22;  unsqueeze_23 = unsqueeze_22 = None
        mul_21 = mul_20 * unsqueeze_21;  mul_20 = unsqueeze_21 = None
        all_parameters_lat_k1_1_2_0 = self.all_parameters.lat_k1_1_2_0
        all_parameters_lat_k2_1_2_0 = self.all_parameters.lat_k2_1_2_0
        all_parameters_lat_k3_1_2_0 = self.all_parameters.lat_k3_1_2_0
        all_parameters_lat_k4_1_2_0 = self.all_parameters.lat_k4_1_2_0
        understeer_corr_local_3 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_2, all_constants_vx_center_1, einsum_30, all_constants_ax_center_0, all_parameters_lat_k1_1_2_0, all_parameters_lat_k2_1_2_0, all_parameters_lat_k3_1_2_0, all_parameters_lat_k4_1_2_0);  all_parameters_lat_k1_1_2_0 = all_parameters_lat_k2_1_2_0 = all_parameters_lat_k3_1_2_0 = all_parameters_lat_k4_1_2_0 = None
        mul_22 = understeer_corr_local_3 * mul_21;  understeer_corr_local_3 = mul_21 = None
        relation_forward_select364_w = self.all_constants.Select364
        einsum_42 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select364_w);  relation_forward_select364_w = None
        unsqueeze_24 = einsum_42.unsqueeze(2);  einsum_42 = None
        relation_forward_select362_w = self.all_constants.Select362
        einsum_43 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select362_w);  relation_forward_select362_w = None
        unsqueeze_25 = einsum_43.unsqueeze(2);  einsum_43 = None
        relation_forward_select361_w = self.all_constants.Select361
        einsum_44 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select361_w);  relation_forward_select361_w = None
        unsqueeze_26 = einsum_44.unsqueeze(2);  einsum_44 = None
        mul_23 = unsqueeze_26 * unsqueeze_25;  unsqueeze_26 = unsqueeze_25 = None
        mul_24 = mul_23 * unsqueeze_24;  mul_23 = unsqueeze_24 = None
        all_constants_steer_center_1 = self.all_constants.steer_center_1
        all_parameters_lat_k1_1_1_1 = self.all_parameters.lat_k1_1_1_1
        all_parameters_lat_k2_1_1_1 = self.all_parameters.lat_k2_1_1_1
        all_parameters_lat_k3_1_1_1 = self.all_parameters.lat_k3_1_1_1
        all_parameters_lat_k4_1_1_1 = self.all_parameters.lat_k4_1_1_1
        understeer_corr_local_4 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_1, all_constants_vx_center_1, einsum_30, all_constants_ax_center_1, all_parameters_lat_k1_1_1_1, all_parameters_lat_k2_1_1_1, all_parameters_lat_k3_1_1_1, all_parameters_lat_k4_1_1_1);  all_parameters_lat_k1_1_1_1 = all_parameters_lat_k2_1_1_1 = all_parameters_lat_k3_1_1_1 = all_parameters_lat_k4_1_1_1 = None
        mul_25 = understeer_corr_local_4 * mul_24;  understeer_corr_local_4 = mul_24 = None
        relation_forward_select343_w = self.all_constants.Select343
        einsum_45 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select343_w);  relation_forward_select343_w = None
        unsqueeze_27 = einsum_45.unsqueeze(2);  einsum_45 = None
        relation_forward_select341_w = self.all_constants.Select341
        einsum_46 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select341_w);  relation_forward_select341_w = None
        unsqueeze_28 = einsum_46.unsqueeze(2);  einsum_46 = None
        relation_forward_select340_w = self.all_constants.Select340
        einsum_47 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select340_w);  relation_forward_select340_w = None
        unsqueeze_29 = einsum_47.unsqueeze(2);  einsum_47 = None
        mul_26 = unsqueeze_29 * unsqueeze_28;  unsqueeze_29 = unsqueeze_28 = None
        mul_27 = mul_26 * unsqueeze_27;  mul_26 = unsqueeze_27 = None
        all_parameters_lat_k1_1_1_0 = self.all_parameters.lat_k1_1_1_0
        all_parameters_lat_k2_1_1_0 = self.all_parameters.lat_k2_1_1_0
        all_parameters_lat_k3_1_1_0 = self.all_parameters.lat_k3_1_1_0
        all_parameters_lat_k4_1_1_0 = self.all_parameters.lat_k4_1_1_0
        understeer_corr_local_5 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_1, all_constants_vx_center_1, einsum_30, all_constants_ax_center_0, all_parameters_lat_k1_1_1_0, all_parameters_lat_k2_1_1_0, all_parameters_lat_k3_1_1_0, all_parameters_lat_k4_1_1_0);  all_parameters_lat_k1_1_1_0 = all_parameters_lat_k2_1_1_0 = all_parameters_lat_k3_1_1_0 = all_parameters_lat_k4_1_1_0 = None
        mul_28 = understeer_corr_local_5 * mul_27;  understeer_corr_local_5 = mul_27 = None
        relation_forward_select322_w = self.all_constants.Select322
        einsum_48 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select322_w);  relation_forward_select322_w = None
        unsqueeze_30 = einsum_48.unsqueeze(2);  einsum_48 = None
        relation_forward_select320_w = self.all_constants.Select320
        einsum_49 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select320_w);  relation_forward_select320_w = None
        unsqueeze_31 = einsum_49.unsqueeze(2);  einsum_49 = None
        relation_forward_select319_w = self.all_constants.Select319
        einsum_50 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select319_w);  relation_forward_select319_w = None
        unsqueeze_32 = einsum_50.unsqueeze(2);  einsum_50 = None
        mul_29 = unsqueeze_32 * unsqueeze_31;  unsqueeze_32 = unsqueeze_31 = None
        mul_30 = mul_29 * unsqueeze_30;  mul_29 = unsqueeze_30 = None
        all_constants_steer_center_0 = self.all_constants.steer_center_0
        all_parameters_lat_k1_1_0_1 = self.all_parameters.lat_k1_1_0_1
        all_parameters_lat_k2_1_0_1 = self.all_parameters.lat_k2_1_0_1
        all_parameters_lat_k3_1_0_1 = self.all_parameters.lat_k3_1_0_1
        all_parameters_lat_k4_1_0_1 = self.all_parameters.lat_k4_1_0_1
        understeer_corr_local_6 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_0, all_constants_vx_center_1, einsum_30, all_constants_ax_center_1, all_parameters_lat_k1_1_0_1, all_parameters_lat_k2_1_0_1, all_parameters_lat_k3_1_0_1, all_parameters_lat_k4_1_0_1);  all_parameters_lat_k1_1_0_1 = all_parameters_lat_k2_1_0_1 = all_parameters_lat_k3_1_0_1 = all_parameters_lat_k4_1_0_1 = None
        mul_31 = understeer_corr_local_6 * mul_30;  understeer_corr_local_6 = mul_30 = None
        relation_forward_select301_w = self.all_constants.Select301
        einsum_51 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select301_w);  relation_forward_select301_w = None
        unsqueeze_33 = einsum_51.unsqueeze(2);  einsum_51 = None
        relation_forward_select299_w = self.all_constants.Select299
        einsum_52 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select299_w);  relation_forward_select299_w = None
        unsqueeze_34 = einsum_52.unsqueeze(2);  einsum_52 = None
        relation_forward_select298_w = self.all_constants.Select298
        einsum_53 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select298_w);  relation_forward_select298_w = None
        unsqueeze_35 = einsum_53.unsqueeze(2);  einsum_53 = None
        mul_32 = unsqueeze_35 * unsqueeze_34;  unsqueeze_35 = unsqueeze_34 = None
        mul_33 = mul_32 * unsqueeze_33;  mul_32 = unsqueeze_33 = None
        all_parameters_lat_k1_1_0_0 = self.all_parameters.lat_k1_1_0_0
        all_parameters_lat_k2_1_0_0 = self.all_parameters.lat_k2_1_0_0
        all_parameters_lat_k3_1_0_0 = self.all_parameters.lat_k3_1_0_0
        all_parameters_lat_k4_1_0_0 = self.all_parameters.lat_k4_1_0_0
        understeer_corr_local_7 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_0, all_constants_vx_center_1, einsum_30, all_constants_ax_center_0, all_parameters_lat_k1_1_0_0, all_parameters_lat_k2_1_0_0, all_parameters_lat_k3_1_0_0, all_parameters_lat_k4_1_0_0);  all_constants_vx_center_1 = all_parameters_lat_k1_1_0_0 = all_parameters_lat_k2_1_0_0 = all_parameters_lat_k3_1_0_0 = all_parameters_lat_k4_1_0_0 = None
        mul_34 = understeer_corr_local_7 * mul_33;  understeer_corr_local_7 = mul_33 = None
        relation_forward_select280_w = self.all_constants.Select280
        einsum_54 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select280_w);  relation_forward_select280_w = None
        unsqueeze_36 = einsum_54.unsqueeze(2);  einsum_54 = None
        relation_forward_select278_w = self.all_constants.Select278
        einsum_55 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select278_w);  relation_forward_select278_w = None
        unsqueeze_37 = einsum_55.unsqueeze(2);  einsum_55 = None
        relation_forward_select277_w = self.all_constants.Select277
        einsum_56 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select277_w);  relation_forward_select277_w = None
        unsqueeze_38 = einsum_56.unsqueeze(2);  einsum_56 = None
        mul_35 = unsqueeze_38 * unsqueeze_37;  unsqueeze_38 = unsqueeze_37 = None
        mul_36 = mul_35 * unsqueeze_36;  mul_35 = unsqueeze_36 = None
        all_constants_vx_center_0 = self.all_constants.vx_center_0
        all_parameters_lat_k1_0_3_1 = self.all_parameters.lat_k1_0_3_1
        all_parameters_lat_k2_0_3_1 = self.all_parameters.lat_k2_0_3_1
        all_parameters_lat_k3_0_3_1 = self.all_parameters.lat_k3_0_3_1
        all_parameters_lat_k4_0_3_1 = self.all_parameters.lat_k4_0_3_1
        understeer_corr_local_8 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_3, all_constants_vx_center_0, einsum_30, all_constants_ax_center_1, all_parameters_lat_k1_0_3_1, all_parameters_lat_k2_0_3_1, all_parameters_lat_k3_0_3_1, all_parameters_lat_k4_0_3_1);  all_parameters_lat_k1_0_3_1 = all_parameters_lat_k2_0_3_1 = all_parameters_lat_k3_0_3_1 = all_parameters_lat_k4_0_3_1 = None
        mul_37 = understeer_corr_local_8 * mul_36;  understeer_corr_local_8 = mul_36 = None
        relation_forward_select259_w = self.all_constants.Select259
        einsum_57 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select259_w);  relation_forward_select259_w = None
        unsqueeze_39 = einsum_57.unsqueeze(2);  einsum_57 = None
        relation_forward_select257_w = self.all_constants.Select257
        einsum_58 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select257_w);  relation_forward_select257_w = None
        unsqueeze_40 = einsum_58.unsqueeze(2);  einsum_58 = None
        relation_forward_select256_w = self.all_constants.Select256
        einsum_59 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select256_w);  relation_forward_select256_w = None
        unsqueeze_41 = einsum_59.unsqueeze(2);  einsum_59 = None
        mul_38 = unsqueeze_41 * unsqueeze_40;  unsqueeze_41 = unsqueeze_40 = None
        mul_39 = mul_38 * unsqueeze_39;  mul_38 = unsqueeze_39 = None
        all_parameters_lat_k1_0_3_0 = self.all_parameters.lat_k1_0_3_0
        all_parameters_lat_k2_0_3_0 = self.all_parameters.lat_k2_0_3_0
        all_parameters_lat_k3_0_3_0 = self.all_parameters.lat_k3_0_3_0
        all_parameters_lat_k4_0_3_0 = self.all_parameters.lat_k4_0_3_0
        understeer_corr_local_9 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_3, all_constants_vx_center_0, einsum_30, all_constants_ax_center_0, all_parameters_lat_k1_0_3_0, all_parameters_lat_k2_0_3_0, all_parameters_lat_k3_0_3_0, all_parameters_lat_k4_0_3_0);  all_constants_steer_center_3 = all_parameters_lat_k1_0_3_0 = all_parameters_lat_k2_0_3_0 = all_parameters_lat_k3_0_3_0 = all_parameters_lat_k4_0_3_0 = None
        mul_40 = understeer_corr_local_9 * mul_39;  understeer_corr_local_9 = mul_39 = None
        relation_forward_select238_w = self.all_constants.Select238
        einsum_60 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select238_w);  relation_forward_select238_w = None
        unsqueeze_42 = einsum_60.unsqueeze(2);  einsum_60 = None
        relation_forward_select236_w = self.all_constants.Select236
        einsum_61 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select236_w);  relation_forward_select236_w = None
        unsqueeze_43 = einsum_61.unsqueeze(2);  einsum_61 = None
        relation_forward_select235_w = self.all_constants.Select235
        einsum_62 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select235_w);  relation_forward_select235_w = None
        unsqueeze_44 = einsum_62.unsqueeze(2);  einsum_62 = None
        mul_41 = unsqueeze_44 * unsqueeze_43;  unsqueeze_44 = unsqueeze_43 = None
        mul_42 = mul_41 * unsqueeze_42;  mul_41 = unsqueeze_42 = None
        all_parameters_lat_k1_0_2_1 = self.all_parameters.lat_k1_0_2_1
        all_parameters_lat_k2_0_2_1 = self.all_parameters.lat_k2_0_2_1
        all_parameters_lat_k3_0_2_1 = self.all_parameters.lat_k3_0_2_1
        all_parameters_lat_k4_0_2_1 = self.all_parameters.lat_k4_0_2_1
        understeer_corr_local_10 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_2, all_constants_vx_center_0, einsum_30, all_constants_ax_center_1, all_parameters_lat_k1_0_2_1, all_parameters_lat_k2_0_2_1, all_parameters_lat_k3_0_2_1, all_parameters_lat_k4_0_2_1);  all_parameters_lat_k1_0_2_1 = all_parameters_lat_k2_0_2_1 = all_parameters_lat_k3_0_2_1 = all_parameters_lat_k4_0_2_1 = None
        mul_43 = understeer_corr_local_10 * mul_42;  understeer_corr_local_10 = mul_42 = None
        relation_forward_select217_w = self.all_constants.Select217
        einsum_63 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select217_w);  relation_forward_select217_w = None
        unsqueeze_45 = einsum_63.unsqueeze(2);  einsum_63 = None
        relation_forward_select215_w = self.all_constants.Select215
        einsum_64 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select215_w);  relation_forward_select215_w = None
        unsqueeze_46 = einsum_64.unsqueeze(2);  einsum_64 = None
        relation_forward_select214_w = self.all_constants.Select214
        einsum_65 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select214_w);  relation_forward_select214_w = None
        unsqueeze_47 = einsum_65.unsqueeze(2);  einsum_65 = None
        mul_44 = unsqueeze_47 * unsqueeze_46;  unsqueeze_47 = unsqueeze_46 = None
        mul_45 = mul_44 * unsqueeze_45;  mul_44 = unsqueeze_45 = None
        all_parameters_lat_k1_0_2_0 = self.all_parameters.lat_k1_0_2_0
        all_parameters_lat_k2_0_2_0 = self.all_parameters.lat_k2_0_2_0
        all_parameters_lat_k3_0_2_0 = self.all_parameters.lat_k3_0_2_0
        all_parameters_lat_k4_0_2_0 = self.all_parameters.lat_k4_0_2_0
        understeer_corr_local_11 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_2, all_constants_vx_center_0, einsum_30, all_constants_ax_center_0, all_parameters_lat_k1_0_2_0, all_parameters_lat_k2_0_2_0, all_parameters_lat_k3_0_2_0, all_parameters_lat_k4_0_2_0);  all_constants_steer_center_2 = all_parameters_lat_k1_0_2_0 = all_parameters_lat_k2_0_2_0 = all_parameters_lat_k3_0_2_0 = all_parameters_lat_k4_0_2_0 = None
        mul_46 = understeer_corr_local_11 * mul_45;  understeer_corr_local_11 = mul_45 = None
        relation_forward_select196_w = self.all_constants.Select196
        einsum_66 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select196_w);  relation_forward_select196_w = None
        unsqueeze_48 = einsum_66.unsqueeze(2);  einsum_66 = None
        relation_forward_select194_w = self.all_constants.Select194
        einsum_67 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select194_w);  relation_forward_select194_w = None
        unsqueeze_49 = einsum_67.unsqueeze(2);  einsum_67 = None
        relation_forward_select193_w = self.all_constants.Select193
        einsum_68 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select193_w);  relation_forward_select193_w = None
        unsqueeze_50 = einsum_68.unsqueeze(2);  einsum_68 = None
        mul_47 = unsqueeze_50 * unsqueeze_49;  unsqueeze_50 = unsqueeze_49 = None
        mul_48 = mul_47 * unsqueeze_48;  mul_47 = unsqueeze_48 = None
        all_parameters_lat_k1_0_1_1 = self.all_parameters.lat_k1_0_1_1
        all_parameters_lat_k2_0_1_1 = self.all_parameters.lat_k2_0_1_1
        all_parameters_lat_k3_0_1_1 = self.all_parameters.lat_k3_0_1_1
        all_parameters_lat_k4_0_1_1 = self.all_parameters.lat_k4_0_1_1
        understeer_corr_local_12 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_1, all_constants_vx_center_0, einsum_30, all_constants_ax_center_1, all_parameters_lat_k1_0_1_1, all_parameters_lat_k2_0_1_1, all_parameters_lat_k3_0_1_1, all_parameters_lat_k4_0_1_1);  all_parameters_lat_k1_0_1_1 = all_parameters_lat_k2_0_1_1 = all_parameters_lat_k3_0_1_1 = all_parameters_lat_k4_0_1_1 = None
        mul_49 = understeer_corr_local_12 * mul_48;  understeer_corr_local_12 = mul_48 = None
        relation_forward_select175_w = self.all_constants.Select175
        einsum_69 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select175_w);  relation_forward_select175_w = None
        unsqueeze_51 = einsum_69.unsqueeze(2);  einsum_69 = None
        relation_forward_select173_w = self.all_constants.Select173
        einsum_70 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select173_w);  relation_forward_select173_w = None
        unsqueeze_52 = einsum_70.unsqueeze(2);  einsum_70 = None
        relation_forward_select172_w = self.all_constants.Select172
        einsum_71 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select172_w);  relation_forward_select172_w = None
        unsqueeze_53 = einsum_71.unsqueeze(2);  einsum_71 = None
        mul_50 = unsqueeze_53 * unsqueeze_52;  unsqueeze_53 = unsqueeze_52 = None
        mul_51 = mul_50 * unsqueeze_51;  mul_50 = unsqueeze_51 = None
        all_parameters_lat_k1_0_1_0 = self.all_parameters.lat_k1_0_1_0
        all_parameters_lat_k2_0_1_0 = self.all_parameters.lat_k2_0_1_0
        all_parameters_lat_k3_0_1_0 = self.all_parameters.lat_k3_0_1_0
        all_parameters_lat_k4_0_1_0 = self.all_parameters.lat_k4_0_1_0
        understeer_corr_local_13 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_1, all_constants_vx_center_0, einsum_30, all_constants_ax_center_0, all_parameters_lat_k1_0_1_0, all_parameters_lat_k2_0_1_0, all_parameters_lat_k3_0_1_0, all_parameters_lat_k4_0_1_0);  all_constants_steer_center_1 = all_parameters_lat_k1_0_1_0 = all_parameters_lat_k2_0_1_0 = all_parameters_lat_k3_0_1_0 = all_parameters_lat_k4_0_1_0 = None
        mul_52 = understeer_corr_local_13 * mul_51;  understeer_corr_local_13 = mul_51 = None
        relation_forward_select154_w = self.all_constants.Select154
        einsum_72 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select154_w);  relation_forward_select154_w = None
        unsqueeze_54 = einsum_72.unsqueeze(2);  einsum_72 = None
        relation_forward_select152_w = self.all_constants.Select152
        einsum_73 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select152_w);  relation_forward_select152_w = None
        unsqueeze_55 = einsum_73.unsqueeze(2);  einsum_73 = None
        relation_forward_select151_w = self.all_constants.Select151
        einsum_74 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select151_w);  relation_forward_select151_w = None
        unsqueeze_56 = einsum_74.unsqueeze(2);  einsum_74 = None
        mul_53 = unsqueeze_56 * unsqueeze_55;  unsqueeze_56 = unsqueeze_55 = None
        mul_54 = mul_53 * unsqueeze_54;  mul_53 = unsqueeze_54 = None
        all_parameters_lat_k1_0_0_1 = self.all_parameters.lat_k1_0_0_1
        all_parameters_lat_k2_0_0_1 = self.all_parameters.lat_k2_0_0_1
        all_parameters_lat_k3_0_0_1 = self.all_parameters.lat_k3_0_0_1
        all_parameters_lat_k4_0_0_1 = self.all_parameters.lat_k4_0_0_1
        understeer_corr_local_14 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_0, all_constants_vx_center_0, einsum_30, all_constants_ax_center_1, all_parameters_lat_k1_0_0_1, all_parameters_lat_k2_0_0_1, all_parameters_lat_k3_0_0_1, all_parameters_lat_k4_0_0_1);  all_constants_ax_center_1 = all_parameters_lat_k1_0_0_1 = all_parameters_lat_k2_0_0_1 = all_parameters_lat_k3_0_0_1 = all_parameters_lat_k4_0_0_1 = None
        mul_55 = understeer_corr_local_14 * mul_54;  understeer_corr_local_14 = mul_54 = None
        relation_forward_select133_w = self.all_constants.Select133
        einsum_75 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select133_w);  repeat_4 = relation_forward_select133_w = None
        unsqueeze_57 = einsum_75.unsqueeze(2);  einsum_75 = None
        relation_forward_select131_w = self.all_constants.Select131
        einsum_76 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select131_w);  repeat_5 = relation_forward_select131_w = None
        unsqueeze_58 = einsum_76.unsqueeze(2);  einsum_76 = None
        relation_forward_select130_w = self.all_constants.Select130
        einsum_77 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select130_w);  repeat_6 = relation_forward_select130_w = None
        unsqueeze_59 = einsum_77.unsqueeze(2);  einsum_77 = None
        mul_56 = unsqueeze_59 * unsqueeze_58;  unsqueeze_59 = unsqueeze_58 = None
        mul_57 = mul_56 * unsqueeze_57;  mul_56 = unsqueeze_57 = None
        all_parameters_lat_k1_0_0_0 = self.all_parameters.lat_k1_0_0_0
        all_parameters_lat_k2_0_0_0 = self.all_parameters.lat_k2_0_0_0
        all_parameters_lat_k3_0_0_0 = self.all_parameters.lat_k3_0_0_0
        all_parameters_lat_k4_0_0_0 = self.all_parameters.lat_k4_0_0_0
        understeer_corr_local_15 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_32, einsum_31, all_constants_steer_center_0, all_constants_vx_center_0, einsum_30, all_constants_ax_center_0, all_parameters_lat_k1_0_0_0, all_parameters_lat_k2_0_0_0, all_parameters_lat_k3_0_0_0, all_parameters_lat_k4_0_0_0);  einsum_32 = einsum_31 = all_constants_steer_center_0 = all_constants_vx_center_0 = einsum_30 = all_constants_ax_center_0 = all_parameters_lat_k1_0_0_0 = all_parameters_lat_k2_0_0_0 = all_parameters_lat_k3_0_0_0 = all_parameters_lat_k4_0_0_0 = None
        mul_58 = understeer_corr_local_15 * mul_57;  understeer_corr_local_15 = mul_57 = None
        add_26 = mul_58 + mul_55;  mul_58 = mul_55 = None
        add_27 = add_26 + mul_52;  add_26 = mul_52 = None
        add_28 = add_27 + mul_49;  add_27 = mul_49 = None
        add_29 = add_28 + mul_46;  add_28 = mul_46 = None
        add_30 = add_29 + mul_43;  add_29 = mul_43 = None
        add_31 = add_30 + mul_40;  add_30 = mul_40 = None
        add_32 = add_31 + mul_37;  add_31 = mul_37 = None
        add_33 = add_32 + mul_34;  add_32 = mul_34 = None
        add_34 = add_33 + mul_31;  add_33 = mul_31 = None
        add_35 = add_34 + mul_28;  add_34 = mul_28 = None
        add_36 = add_35 + mul_25;  add_35 = mul_25 = None
        add_37 = add_36 + mul_22;  add_36 = mul_22 = None
        add_38 = add_37 + mul_19;  add_37 = mul_19 = None
        add_39 = add_38 + mul_16;  add_38 = mul_16 = None
        add_40 = add_39 + mul_13;  add_39 = mul_13 = None
        mul_59 = add_40 * mul_10;  mul_10 = None
        size_12 = mul_59.size(0)
        relation_forward_fir485_weights = self.all_parameters.PFir142W
        size_13 = relation_forward_fir485_weights.size(1)
        squeeze_6 = mul_59.squeeze(-1);  mul_59 = None
        matmul_6 = torch.matmul(squeeze_6, relation_forward_fir485_weights);  squeeze_6 = relation_forward_fir485_weights = None
        to_6 = matmul_6.to(dtype = torch.float32);  matmul_6 = None
        view_6 = to_6.view(size_12, 1, size_13);  to_6 = size_12 = size_13 = None
        relation_forward_select477_w = self.all_constants.Select477
        einsum_78 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select477_w);  relation_forward_select477_w = None
        unsqueeze_60 = einsum_78.unsqueeze(2);  einsum_78 = None
        relation_forward_select476_w = self.all_constants.Select476
        einsum_79 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select476_w);  relation_forward_select476_w = None
        unsqueeze_61 = einsum_79.unsqueeze(2);  einsum_79 = None
        mul_60 = unsqueeze_61 * unsqueeze_60;  unsqueeze_61 = unsqueeze_60 = None
        mul_61 = add_40 * mul_60;  mul_60 = None
        size_14 = mul_61.size(0)
        relation_forward_fir480_weights = self.all_parameters.PFir140W
        size_15 = relation_forward_fir480_weights.size(1)
        squeeze_7 = mul_61.squeeze(-1);  mul_61 = None
        matmul_7 = torch.matmul(squeeze_7, relation_forward_fir480_weights);  squeeze_7 = relation_forward_fir480_weights = None
        to_7 = matmul_7.to(dtype = torch.float32);  matmul_7 = None
        view_7 = to_7.view(size_14, 1, size_15);  to_7 = size_14 = size_15 = None
        relation_forward_select472_w = self.all_constants.Select472
        einsum_80 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select472_w);  relation_forward_select472_w = None
        unsqueeze_62 = einsum_80.unsqueeze(2);  einsum_80 = None
        relation_forward_select471_w = self.all_constants.Select471
        einsum_81 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select471_w);  relation_forward_select471_w = None
        unsqueeze_63 = einsum_81.unsqueeze(2);  einsum_81 = None
        mul_62 = unsqueeze_63 * unsqueeze_62;  unsqueeze_63 = unsqueeze_62 = None
        mul_63 = add_40 * mul_62;  mul_62 = None
        size_16 = mul_63.size(0)
        relation_forward_fir475_weights = self.all_parameters.PFir138W
        size_17 = relation_forward_fir475_weights.size(1)
        squeeze_8 = mul_63.squeeze(-1);  mul_63 = None
        matmul_8 = torch.matmul(squeeze_8, relation_forward_fir475_weights);  squeeze_8 = relation_forward_fir475_weights = None
        to_8 = matmul_8.to(dtype = torch.float32);  matmul_8 = None
        view_8 = to_8.view(size_16, 1, size_17);  to_8 = size_16 = size_17 = None
        relation_forward_select467_w = self.all_constants.Select467
        einsum_82 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select467_w);  repeat_2 = relation_forward_select467_w = None
        unsqueeze_64 = einsum_82.unsqueeze(2);  einsum_82 = None
        relation_forward_select466_w = self.all_constants.Select466
        einsum_83 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select466_w);  repeat_3 = relation_forward_select466_w = None
        unsqueeze_65 = einsum_83.unsqueeze(2);  einsum_83 = None
        mul_64 = unsqueeze_65 * unsqueeze_64;  unsqueeze_65 = unsqueeze_64 = None
        mul_65 = add_40 * mul_64;  add_40 = mul_64 = None
        size_18 = mul_65.size(0)
        relation_forward_fir470_weights = self.all_parameters.PFir136W
        size_19 = relation_forward_fir470_weights.size(1)
        squeeze_9 = mul_65.squeeze(-1);  mul_65 = None
        matmul_9 = torch.matmul(squeeze_9, relation_forward_fir470_weights);  squeeze_9 = relation_forward_fir470_weights = None
        to_9 = matmul_9.to(dtype = torch.float32);  matmul_9 = None
        view_9 = to_9.view(size_18, 1, size_19);  to_9 = size_18 = size_19 = None
        add_41 = view_9 + view_8;  view_9 = view_8 = None
        add_42 = add_41 + view_7;  add_41 = view_7 = None
        add_43 = add_42 + view_6;  add_42 = view_6 = None
        getitem_16 = yaw_rate
        relation_forward_sample_part499_w = self.all_constants.SamplePart499
        einsum_84 = torch.functional.einsum('bij,ki->bkj', getitem_16, relation_forward_sample_part499_w);  getitem_16 = relation_forward_sample_part499_w = None
        getitem_17 = vel;  kwargs = None
        relation_forward_sample_part490_w = self.all_constants.SamplePart490
        einsum_85 = torch.functional.einsum('bij,ki->bkj', getitem_17, relation_forward_sample_part490_w);  getitem_17 = relation_forward_sample_part490_w = None
        mul_66 = add_43 * einsum_85;  einsum_85 = None
        outputs = ({'acceleration': acc_model_based, 'yaw_rate_': add_43, 'accy_computed': mul_66}, {'SamplePart499': einsum_84, 'SamplePart501': einsum_19, 'Add488': add_43, 'ParamFun497': acc_model_based}, {}, {})
        return (outputs[0]['acceleration'],outputs[0]['yaw_rate_'],outputs[0]['accy_computed'],), (outputs[1]['SamplePart499'], outputs[1]['SamplePart501'], outputs[1]['Add488'], outputs[1]['ParamFun497'], ), (), ()
