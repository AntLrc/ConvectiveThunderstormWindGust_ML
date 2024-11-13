import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from nn_losses import gev_crps_loss
from flax.training import train_state

from functools import partial
from jax import jit


def l1_loss(x, alpha):
    return alpha * jnp.abs(x).mean()


def l2_loss(x, alpha):
    return alpha * (x**2).mean()


@partial(jit, static_argnums=(3, 4, 5, 6, 7))
def loss_and_crps(
    state,
    params,
    batch,
    batch_size,
    total_len,
    n_clusters,
    regularisation=None,
    alpha=0.01,
):
    x_s, x_t, y_true = batch
    y_pred = state.apply_fn(params, x_s, x_t)
    crps = gev_crps_loss(y_pred, y_true, total_len, batch_size, n_clusters)
    return crps, crps + (regularisation == "l2") * sum(
        l2_loss(w, alpha=alpha) for w in jax.tree.leaves(params)
    ) + (regularisation == "l1") * sum(
        l1_loss(w, alpha=alpha) for w in jax.tree.leaves(params)
    )


def mae(
    state,
    params,
    batch,
    batch_size,
    total_len,
    n_clusters,
    regularisation=None,
    alpha=0.01,
):
    x_s, x_t, y_true = batch
    y_pred = state.apply_fn(params, x_s, x_t)
    y_true = jnp.concatenate(y_true, axis=1)
    mae_loss = jnp.abs(y_pred - y_true).mean()
    return mae_loss, mae_loss + (regularisation == "l2") * sum(
        l2_loss(w, alpha=alpha) for w in jax.tree.leaves(params)
    ) + (regularisation == "l1") * sum(
        l1_loss(w, alpha=alpha) for w in jax.tree.leaves(params)
    )


class TrainState(train_state.TrainState):
    key: jnp.ndarray


def create_train_state(
    model, rng, learning_rate, batch_size, features, stationwise=False, n_stations=0
):
    if stationwise:
        params = model.init(
            rng,
            jnp.ones((batch_size, n_stations, features)),
            jnp.ones((batch_size, 4)),
            jnp.ones((batch_size, 4)),
        )
    else:
        params = model.init(
            rng, jnp.ones((batch_size, 20, 34, features)), jnp.ones((batch_size, 4))
        )
    tx = optax.adam(learning_rate=learning_rate)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, key=rng)


def create_batches(features_s, features_t, labels, batch_size, rng):
    n_batches = len(features_s) // batch_size
    # Create shuffled indices
    indices = jax.random.permutation(rng, len(features_s))
    for i in range(n_batches):
        yield features_s[indices[i * batch_size : (i + 1) * batch_size]], features_t[
            indices[i * batch_size : (i + 1) * batch_size]
        ], tuple(
            map(lambda x: x[indices[i * batch_size : (i + 1) * batch_size]], labels)
        )


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9))
def train_step(
    state,
    x_s,
    x_t,
    y_true,
    batch_size,
    total_len,
    n_clusters,
    regularisation=None,
    alpha=0.01,
    target=0,
):
    if target == 0:

        def loss_fn(
            state,
            params,
            batch,
            batch_size,
            total_len,
            n_clusters,
            regularisation=None,
            alpha=0.01,
            dropout_train_key=None,
        ):
            x_s, x_t, y_true = batch
            y_pred = state.apply_fn(
                params, x_s, x_t, training=True, rngs={"dropout": dropout_train_key}
            )
            loss = gev_crps_loss(y_pred, y_true, total_len, batch_size, n_clusters)
            return (
                loss
                + (regularisation == "l2")
                * sum(l2_loss(w, alpha=alpha) for w in jax.tree.leaves(params))
                + (regularisation == "l1")
                * sum(l1_loss(w, alpha=alpha) for w in jax.tree.leaves(params))
            )

    else:

        def loss_fn(
            state,
            params,
            batch,
            batch_size,
            total_len,
            n_clusters,
            regularisation=None,
            alpha=0.01,
            dropout_train_key=None,
        ):
            x_s, x_t, y_true = batch
            y_pred = state.apply_fn(
                params, x_s, x_t, training=True, rngs={"dropout": dropout_train_key}
            )
            y_true = jnp.concatenate(y_true, axis=1)
            loss = jnp.abs(y_pred - y_true).mean()
            return (
                loss
                + (regularisation == "l2")
                * sum(l2_loss(w, alpha=alpha) for w in jax.tree.leaves(params))
                + (regularisation == "l1")
                * sum(l1_loss(w, alpha=alpha) for w in jax.tree.leaves(params))
            )

    dropout_train_key = jax.random.fold_in(state.key, state.step)
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(
        state,
        state.params,
        (x_s, x_t, y_true),
        batch_size,
        total_len,
        n_clusters,
        regularisation,
        alpha,
        dropout_train_key,
    )
    state = state.apply_gradients(grads=grads)
    return state


def train_loop(
    state,
    train_features_s,
    train_features_t,
    train_labels,
    val_features_s,
    val_features_t,
    val_labels,
    batch_size,
    epochs,
    total_len,
    n_clusters,
    rng_shuffle,
    regularisation=None,
    alpha=0.01,
    n_best_states=3,
    target=0,
    early_stopping=6,
):
    train_loss_arr = []
    train_crps_arr = []
    val_loss_arr = []
    val_crps_arr = []
    best_params_with_scores = []
    metrics_fn = (
        loss_and_crps if target == 0 else mae
    )  # Convention: 0 for GEV, 1 for values
    n_steps = train_features_t.shape[0]
    for epoch in range(epochs):
        print("[                    ]", end="\r[")
        percent = 0
        # Training on the epoch
        for x_s, x_t, y_true in create_batches(
            train_features_s, train_features_t, train_labels, batch_size, rng_shuffle
        ):
            percent += (100 * batch_size) / n_steps
            if percent >= 5:
                print("=", end="", flush=True)
                percent -= 5
            state = train_step(
                state,
                x_s,
                x_t,
                y_true,
                batch_size,
                total_len,
                n_clusters,
                regularisation=regularisation,
                alpha=alpha,
                target=target,
            )
        print("")
        # Calculating loss and CRPS associated to the epoch
        train_loss = 0
        train_crps = 0
        count = 0
        for x_s, x_t, y_true in create_batches(
            train_features_s, train_features_t, train_labels, batch_size, rng_shuffle
        ):
            tmp_crps, tmp_loss = metrics_fn(
                state,
                state.params,
                (x_s, x_t, y_true),
                batch_size,
                total_len,
                n_clusters,
                regularisation,
                alpha,
            )
            train_crps += tmp_crps
            train_loss += tmp_loss
            count += 1
        print(f"Epoch {epoch} - Training CRPS: {train_crps/count}", flush=True)
        print(f"Epoch {epoch} - Training loss: {train_loss/count}", flush=True)
        train_loss_arr.append(train_loss / count)
        train_crps_arr.append(train_crps / count)

        # Calculating loss and CRPS associated to the validation set
        val_loss = 0
        val_crps = 0
        count = 0
        for x_s, x_t, y_true in create_batches(
            val_features_s, val_features_t, val_labels, batch_size, rng_shuffle
        ):
            tmp_crps, tmp_loss = metrics_fn(
                state,
                state.params,
                (x_s, x_t, y_true),
                batch_size,
                total_len,
                n_clusters,
                regularisation,
                alpha,
            )
            val_crps += tmp_crps
            val_loss += tmp_loss
            count += 1
        print(f"Epoch {epoch} - Validation CRPS: {val_crps/count}", flush=True)
        print(f"Epoch {epoch} - Validation loss: {val_loss/count}", flush=True)
        val_loss_arr.append(val_loss / count)
        val_crps_arr.append(val_crps / count)

        best_params_with_scores.append((state.params, val_crps / count))

        # Reorder best_states_with_scores
        best_params_with_scores = sorted(best_params_with_scores, key=lambda x: x[1])

        # Keep only the best states
        best_params_with_scores = best_params_with_scores[:n_best_states]

        # If CRPS have been increasing steadily on five last state : break from the for loop (early stopping)
        if early_stopping > 0 and epoch > early_stopping:
            if all(
                [
                    val_crps_arr[-i] > val_crps_arr[-i - 1]
                    for i in range(1, early_stopping + 1)
                ]
            ):
                break

    return (
        best_params_with_scores,
        train_loss_arr,
        val_loss_arr,
        train_crps_arr,
        val_crps_arr,
    )
