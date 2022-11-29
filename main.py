from typing import *
import jax
from jax import numpy as jnp
import numpy as np

TYPE = float


def pred(model_selection: jnp.ndarray, models: jnp.ndarray) -> jnp.ndarray:
    assert (len(models.shape) == 3)
    assert (len(model_selection.shape) == 2)
    nb_selected = jnp.sum(model_selection)
    model_selection = model_selection.reshape((models.shape[0], models.shape[1], 1))
    preds = model_selection * models
    final_pred = jnp.sum(jnp.array(preds), axis=0) / nb_selected
    return final_pred


def loss(y_pred, y):
    elements_loss = (y_pred - y) ** 2
    global_loss = jnp.mean(elements_loss)
    return global_loss


def f(bag_of_models, model_preds, y):
    y_pred = pred(bag_of_models, model_preds)
    l = loss(y_pred, y)
    return l


def from_set_to_array(s: Set, array_size: int) -> jnp.array:
    array = np.zeros((array_size, 1), dtype=float)
    for si in s:
        array[si][0] = 1.
    return array


def linear_scan_init(f, model_preds, y):
    nb_models = len(model_preds)
    scores = [jnp.inf] * nb_models
    for i in range(nb_models):
        start = from_set_to_array({i}, nb_models)
        scores[i] = f(start, model_preds, y)
    best_start_id = np.argmin(scores)
    cur_loss = np.min(scores)
    cur_ensemble = from_set_to_array({best_start_id}, len(scores))
    return cur_ensemble, cur_loss


def gradient_informed_ensemble_selection(f, model_preds, y, min_models=0, max_models=4, init_strat="linear"):
    nb_models = len(model_preds)

    # INITIALISATION
    if init_strat == "linear":
        cur_ensemble, best_loss = linear_scan_init(f, model_preds, y)
    elif init_strat == "epsilon":
        cur_ensemble = np.array([[1]] * nb_models, dtype=TYPE)
        best_loss = f(cur_ensemble, model_preds, y)
    else:
        raise ValueError("Init. not understood")

    # GRADIENT INFORMED ENSEMBLE CONSTRUCTION
    df = jax.grad(f, 0)
    for i in range(max_models):

        delta_ensemble = df(cur_ensemble, model_preds, y)
        if i == 0 and init_strat == "epsilon":  # rollback to 0
            cur_ensemble = np.array([[0]] * nb_models, dtype=TYPE)

        best_start_id = np.argmin(delta_ensemble)
        cur_ensemble[best_start_id][0] += 1  # selection

        cur_loss = f(cur_ensemble, model_preds, y)

        print("Score: ", cur_loss)

        if cur_loss > best_loss and i >= min_models:  # if worsen and min number of model reached
            cur_ensemble[best_start_id][0] -= 1  # rollback because it worsen
            return cur_ensemble
        best_loss = cur_loss
    return cur_ensemble


if __name__ == "__main__":
    a = np.array([[0.6, 0.2, 0.2]])
    b = np.array([[0.2, 0.6, 0.2]])
    c = np.array([[0.2, 0.2, 0.6]])
    d = np.array([[0.3, 0.3, 0.4]])

    model_preds = jnp.array([a, b, c, d])

    y = np.array([[0, 0.5, 0.5]])

    bag_of_models = gradient_informed_ensemble_selection(f, model_preds, y,
                                                         10, 20, init_strat="epsilon")
    print(bag_of_models)
