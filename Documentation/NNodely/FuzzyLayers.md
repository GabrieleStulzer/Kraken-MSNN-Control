# Fuzzy Layers

A Fuzzy layer in nnodely (the Fuzzify block) is a membership-function encoder that converts a crisp numeric variable (e.g. gear number, speed range, temperature, etc.) into a set of fuzzy activations, each representing how much the input belongs to a predefined region.

In other words:

A Fuzzy layer transforms a scalar input into multiple “membership signals” that softly activate different sub-models depending on the operating region.

This is the core idea behind local models in nnodely: instead of using a single global neural model, you define several simpler models — each valid in a specific operating region — and the fuzzy layer blends them.