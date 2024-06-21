import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from cnn_losses import gevCRPSLoss
from flax.training import train_state

from functools import partial
from jax import jit

def l2_loss(x, alpha):
    return alpha*(x**2).mean()

def l1_loss(x, alpha):
    return alpha*jnp.abs(x).mean()


def loss_and_CRPS(state, params, batch, batch_size, total_len, regularisation = None, alpha = 0.01):
    x_s, x_t, y_true = batch
    y_pred = state.apply_fn(params, x_s, x_t)
    crps = gevCRPSLoss(y_pred, y_true, total_len, batch_size)
    return crps, crps +\
        (regularisation == "l2")*sum(l2_loss(w, alpha=alpha) for w in jax.tree.leaves(params)) +\
            (regularisation == "l1")*sum(l1_loss(w, alpha=alpha) for w in jax.tree.leaves(params))

def calculate_loss(state, params, batch, batch_size, total_len, regularisation = None, alpha = 0.01):
    x_s, x_t, y_true = batch
    y_pred = state.apply_fn(params, x_s, x_t)
    loss = gevCRPSLoss(y_pred, y_true, total_len, batch_size)
    return loss +\
        (regularisation == "l2")*sum(l2_loss(w, alpha=alpha) for w in jax.tree.leaves(params)) +\
            (regularisation == "l1")*sum(l1_loss(w, alpha=alpha) for w in jax.tree.leaves(params))

def createTrainState(model, rng, learning_rate, batch_size, features):
    params = model.init(rng,jnp.ones((batch_size, 20, 34, features)), jnp.ones((batch_size, 3)))
    tx = optax.adam(learning_rate = learning_rate)
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def createBatches(features_s, features_t, labels, batch_size, rng):
    nbatches = len(features_s)//batch_size
    # Create shuffled indices
    indices = jax.random.permutation(rng, len(features_s))
    for i in range(nbatches):
        yield features_s[indices[i*batch_size:(i+1)*batch_size]], features_t[indices[i*batch_size:(i+1)*batch_size]], tuple(map(lambda x: x[indices[i*batch_size:(i+1)*batch_size]], labels))
        

@partial(jit, static_argnums = (4,5,6,7))
def train_step(state, x_s, x_t, y_true, batch_size, total_len, regularisation = None, alpha = 0.01):
    loss, grads = jax.value_and_grad(calculate_loss, argnums = 1)(state, state.params, (x_s, x_t, y_true), batch_size, total_len, regularisation, alpha)
    state = state.apply_gradients(grads = grads)
    return state, loss, grads


def train_loop(state,
               train_features_s,
               train_features_t,
               train_labels,
               val_features_s,
               val_features_t,
               val_labels,
               batch_size,
               epochs,
               total_len,
               rngshuffle,
               regularisation = None,
               alpha = 0.01,
               n_best_states = 3):
    train_loss_arr = []
    train_CRPS_arr = []
    val_loss_arr = []
    val_CRPS_arr = []
    best_states_with_scores = []
    for epoch in range(epochs):
        grads = 0.
        
        # Training on the epoch
        for x_s, x_t, y_true in createBatches(train_features_s, train_features_t, train_labels, batch_size, rngshuffle):
            state, loss, grads = train_step(state, x_s, x_t, y_true,
                                            batch_size, total_len,
                                            regularisation, alpha)
            
            print(f"\rEpoch {epoch} - Training Loss: {loss}", end="")
            
        # Calculating loss and CRPS associated to the epoch
        train_loss = 0
        train_CRPS = 0
        count = 0
        for x_s, x_t, y_true in createBatches(train_features_s, train_features_t, train_labels, batch_size, rngshuffle):
            tmp_crps, tmp_loss = loss_and_CRPS(state, state.params, (x_s, x_t, y_true), batch_size, total_len, regularisation, alpha)
            train_CRPS += tmp_crps
            train_loss += tmp_loss
            count += 1
        print(f"\rEpoch {epoch} - Training Loss: {train_loss}")
        train_loss_arr.append(train_loss/count)
        train_CRPS_arr.append(train_CRPS/count)
        
        
        # Calculating loss and CRPS associated to the validation set
        val_loss = 0
        val_CRPS = 0
        count = 0
        for x_s, x_t, y_true in createBatches(val_features_s, val_features_t, val_labels, batch_size, rngshuffle):
            tmp_crps, tmp_loss = loss_and_CRPS(state, state.params, (x_s, x_t, y_true), batch_size, total_len, regularisation, alpha)
            val_CRPS += tmp_crps
            val_loss += tmp_loss
            count += 1
        print(f"Epoch {epoch} - Validation Loss: {val_loss/count}")
        val_loss_arr.append(val_loss/count)
        val_CRPS_arr.append(val_CRPS/count)
        
        best_states_with_scores.append((state, val_CRPS/count))
        
        # Reorder best_states_with_scores
        best_states_with_scores = sorted(best_states_with_scores, key = lambda x: x[1])
        
        # Keep only the best states
        best_states_with_scores = best_states_with_scores[:n_best_states]
        
    return best_states_with_scores, train_loss_arr, val_loss_arr, train_CRPS_arr, val_CRPS_arr