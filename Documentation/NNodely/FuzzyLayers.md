# Fuzzy Layers

A Fuzzy layer in nnodely (the Fuzzify block) is a membership-function encoder that converts a crisp numeric variable (e.g. gear number, speed range, temperature, etc.) into a set of fuzzy activations, each representing how much the input belongs to a predefined region.

In other words:

A Fuzzy layer transforms a scalar input into multiple “membership signals” that softly activate different sub-models depending on the operating region.

This is the core idea behind local models in nnodely: instead of using a single global neural model, you define several simpler models — each valid in a specific operating region — and the fuzzy layer blends them.

Dreams4Cars --> diviso in due parti: sleep mode and wake mode. In wake mode prende i dati per il training del predictive model, mentre poi nella sleep mode sintetizza il controllo e il comportamento. 

Che è come dovrebbe lavorare nnodely: prima in un  modello costruisce la dinamica e poi usando il modello costruito ci costruisce sopra il controllo.

Reinforcement learning è un approccio che combina le soluzioni di controllo ottimo provando delle azioni e osservando i risultati.

- **Forward model:**
  - **Modeling**: Selezionare modelli neurali fisici
  - **Training**: Allenare la rete con i dati
  - **Evaluation**: Valutare tramite le varie metriche i residui della rete neurale
- **Inverse model:**
  - **Modelling**: Selezionare il modello di rete neurale
  - **Training**: Allenare la rete con simulazioni/dati
  - **Evaluation**: Valutare la rete e la predizione della traiettoria, usando dati reali/sul veicolo

Forward model --> dinamica diretta, dato acceleratore/freno/sterzo costruire la traiettoria e il profilo di velocità

Inverse model --> dinamica inversa, opposta a quella descritta sopra

Learning ***Forward Model*** può essere diviso in tre parti:
 - Modeling
 - Training
 - Evaluate

Per le recurrent networks esistono gli stati interni, quindi si può descrivere una dinamica per questi.

**Unica forza:**
- 
Si utilizza un'unica equazione che descrive tutte le forze che modificano il modello dinamico del sistema.

**Superposition of the effects:**
- 
Ricordarsi che le forze si possono sommare.
Per esempio possiamo descrivere l'accelerazione di un veicolo in questo modo:

$$
  m a = T_e - F_D - F_v - F_r - F_y sin(\delta)
$$

dove:
- $T_e$ è la torque esterna,
- $F_D$ è la forza di Drag,
- $F_v$ è la forza dovuta agli attriti viscosi,
- $F_r$ è la forza dovuta agli attriti di rotolamento,
- $F_ysin(\delta)$ è la forza dovuta al combinato.

Queste varie forze possono essere strutturate con reti divise fra di loro e poi sommate.

**Superposition + weights:**
- 
$$
  m a = \phi_0T_t - \phi_1T_b - \phi_2F_D - \phi_3F_v - \phi_4F_r - \phi_5F_y sin(\delta)
$$

$\phi_i$ è la funzione di attivazione, molto utile nel caso in cui ci siano componenti che dovrebbero essere a zero in certi casi, come mostrato nell'equazione d'esempio. 

Dove $\phi_0$ serve per attivare la torque dei motori, mentre $\phi_1$ serve per attivare la torque frenante. 

*Rircorda:* BSPD, il pilota tende a frenare e accelerare contemporaneamente:
Le funzioni di attivazione possono essere soggette a learning o possono essere predefinite dall'utente.