# Ensemble Selection with Jax
High Performance Ensemble Selection Gradient-Informed with Jax

Complexity of Ensemble Selection with L models in the library of models and n the size of the subset of models:

Ensemble selection | Complexity | When L=1000 and n=10
| :--- | ---: | :---:
Brut force  | O(Combi(L,n)) forwards* | ~2.6e23
Greedy  | O(n*L) forwards | 10K
Greedy with Jax  | O(n) backwards** | 10

* 1 forward means 'ensemble of models in prediction phase'
**cost of 1 forward approximatively equal 1 backward

Greedy with Jax is a gradient-informed method much less computing intensive than standard Greedy approach. It is possible to vary n to control the trade-of between prediction quality (accuracy, MSE,...) and prediction speed (e.g. pred-per-second, latency,...) by varying n.
