import jax
import jax.numpy as jnp

from jax.scipy.special import gammainc, expi
from jax.lax import lgamma
from jax.numpy import exp, log

from functools import partial
from jax import jit, custom_jvp


def return_level(mu, sigma, xi, p):
    """
    Compute the return levels for the Generalized Extreme Value distribution.
    
    Parameters
    ----------
    mu : float or array-like
        Location parameter.
    sigma : float or array-like
        Scale parameter.
    xi : float or array-like
        Shape parameter.
    p : float or array-like
        Return period.
        
    Returns
    -------
    float or array-like
        Value of the return levels.
    """
    yp = -jnp.log(1 - p)
    xi_val = jnp.where(xi == 0, 0.5, xi)
    return jnp.where(
        xi == 0, mu - sigma * jnp.log(yp), mu - sigma / xi_val * (1 - yp ** (-xi_val))
    )


def return_level_loss(param_pred, y_true, total_len, batch_size, n_clusters, p):
    """
    Compute the return level loss for the Generalized Extreme Value distribution.
    
    Parameters
    ----------
    param_pred : array-like
        Predicted parameters of the GEV distribution.
    y_true : array-like
        True values.
    total_len : int
        Total length of the concatenated true values.
    batch_size : int
        Batch size.
    n_clusters : int
        Number of clusters.
    p : float
        Return period.
    """
    mu, sigma, xi = jnp.split(param_pred, 3, axis=1)
    r_levs = return_level(mu, sigma, xi, jnp.repeat(p, n_clusters))
    emp_r_levs = jnp.asarray(
        tuple(
            map(
                lambda x: jnp.percentile(x, 100 * (1 - p), method="linear", axis=-1),
                y_true,
            )
        )
    ).T
    return jnp.mean((r_levs - emp_r_levs) ** 2)
