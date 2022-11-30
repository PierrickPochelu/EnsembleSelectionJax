# EnsembleSelectionJax
High Performance Ensemble Selection Gradient-Informed with Jax

Complexity of Ensemble Selection with L models in the library of models and n the size of the subset of models:

Brut force: O(Combi(L,n)) forwards. Computing intractable in practice.
Greedy implementation: O(L*n) forwards . Still computing intensive in practice with large library, larg
Proposed implementation: O(n) backwards operation (cost of 1 forward approximatively equal 1 backward)


Ensemble selection | Complexity | When L=1000 and n=10
| :--- | ---: | :---:
Brut force  | O(Combi(L,n)) forwards | ~2.6e23
Greedy  | O(n*L) forwards | 10K
Greedy with Jax  | O(n) backwards | 10

*cost of 1 forward approximatively equal 1 backward
