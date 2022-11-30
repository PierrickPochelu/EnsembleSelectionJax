# Ensemble Selection with Jax
High-Performance Ensemble Selection Gradient-Informed with Jax.

Ensemble Selection is described by Caruana et al. 2004. It is a combinatorial optimization problem consisting in extracting the best subset (Ensemble) of models in a bigger library of models. We know that the best ensemble is not made of the best individual model but complementary models canceling out their error. Finding complementary models is a non-trivial problem due to the huge number of potential ensembles (combinations).


For comparison, I compare different designs of Ensemble Selection. L is the number of models in the library and n is the desired size of the subset (Ensemble) of models:

Ensemble selection | Complexity | When L=1000 and n=10
| :--- | ---: | :---:
Brut force  | O(Combi(L,n)) forwards* | ~2.6e23
Greedy  | O(n*L) forwards | 10K
Greedy with Jax  | O(n) backwards** | 10

*1 forward means 'ensemble of models evaluated in forward phase'

**cost of 1 forward is approximatively equal to 1 backward

Greedy with Jax is a gradient-informed method much less computing intensive than the standard Greedy approach. For all methods, n controls the trade-off between prediction quality (e.g., accuracy, MSE,...) and speed (ensemble selection time, pred-per-second, prediction latency).
