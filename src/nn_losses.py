import jax
import jax.numpy as jnp

from jax.scipy.special import gammainc, expi
from jax.lax import lgamma
from jax.numpy import exp, log

from functools import partial
from jax import jit, custom_jvp


# Custom definition of exp(-exp) to avoid overflow. ONLY VALID FOR SIGMA AND MU "SLOWLY" VARYING
@custom_jvp
def double_exp(mu, sigma, y):
    return exp(-exp(-(y - mu) / sigma))


@double_exp.defjvp
def double_exp_jvp(primals, tangents):
    mu, sigma, y = primals
    mu_dot, sigma_dot, y_dot = tangents
    primals_out = double_exp(mu, sigma, y)
    mu_val = jnp.where(jnp.logical_or(primals_out == 0.0, primals_out == 1.0), 0.0, mu)
    sigma_val = jnp.where(
        jnp.logical_or(primals_out == 0.0, primals_out == 1.0), 1.0, sigma
    )
    y_val = jnp.where(jnp.logical_or(primals_out == 0.0, primals_out == 1.0), 0.0, y)

    res_var = jnp.where(
        jnp.logical_or(primals_out == 0.0, primals_out == 1.0),
        0.0 * (mu_dot + sigma_dot + y_dot),
        -primals_out
        * exp(-(y_val - mu_val) / sigma_val)
        * (
            mu_dot / sigma_val - y_dot / sigma_val + (y_val - mu_val) * sigma_dot / sigma_val**2
        ),
    )
    return primals_out, res_var


def gev(mu, sigma, xi, y):
    """
    Computes the Generalized Extreme Value CDF.
    """

    y_red = (y - mu) / sigma
    xi_null_mask = xi == 0
    xi_val = jnp.where(xi_null_mask, 0.5, xi)

    y0 = jnp.logical_and(xi > 0, y_red <= -1 / xi_val)
    y1 = jnp.logical_and(xi < 0, y_red >= -1 / xi_val)

    y_in_boundary = jnp.logical_or(
        jnp.logical_and(xi < 0, y_red < -1 / xi_val),
        jnp.logical_and(xi > 0, y_red > -1 / xi_val),
    )

    y_red_val = jnp.where(
        jnp.logical_or(y0, y1), (jnp.log(2) ** (-xi_val) - 1) / xi_val, y_red
    )

    return jnp.where(
        y_in_boundary,
        exp(-((1 + xi_val * y_red_val) ** (-1 / xi_val))),
        jnp.where(xi_null_mask, double_exp(mu, sigma, y), jnp.where(y1, 1.0, 0.0)),
    )


def gev_pdf(mu, sigma, xi, y):
    y_red = (y - mu) / sigma
    y0 = jnp.logical_and(xi > 0, y_red <= -1 / xi)
    y1 = jnp.logical_and(xi < 0, y_red >= -1 / xi)

    xi_val = jnp.where(xi == 0, 0.5, xi)
    y_red_val = jnp.where(
        jnp.logical_or(y0, y1), (jnp.log(2) ** (-xi_val) - 1) / xi_val, y_red
    )

    y_in_boundary = jnp.logical_or(
        jnp.logical_and(xi < 0, y_red < -1 / xi_val),
        jnp.logical_and(xi > 0, y_red > -1 / xi_val),
    )

    y_in_boundary = jnp.logical_or(y_in_boundary, xi == 0)

    ty = jnp.where(xi == 0, exp(-y_red_val), (1 + xi_val * y_red_val) ** (-1 / xi))

    return jnp.where(y_in_boundary, (1 / sigma) * ty ** (xi + 1) * exp(-ty), 0.0)


def gev_crps(mu, sigma, xi, y):
    """
    Compute the closed form of the Continuous Ranked Probability Score (CRPS) for the Generalized Extreme Value distribution.
    Based on Friedrichs and Thorarinsdottir (2012).
    """

    y_red = (y - mu) / sigma
    xi_null_mask = xi == 0
    xi_val = jnp.where(xi_null_mask, 0.5, xi)

    gev_val = gev(mu, sigma, xi, y)

    y0 = jnp.logical_and(xi > 0, y_red <= -1 / xi_val)
    y1 = jnp.logical_and(xi < 0, y_red >= -1 / xi_val)

    y_in_boundary = jnp.logical_and(
        jnp.logical_not(xi_null_mask), jnp.logical_not(jnp.logical_or(y1, y0))
    )

    y_red_val = jnp.where(
        jnp.logical_or(y0, y1), (jnp.log(2) ** (-xi_val) - 1) / xi_val, y_red
    )

    exp_y_red_null = -exp(jnp.where(xi_null_mask, -y_red, 0.0))

    return jnp.where(
        y_in_boundary,
        sigma * (-y_red_val - 1 / xi_val) * (1 - 2 * gev_val)
        - sigma
        / xi_val
        * exp(lgamma(1 - xi_val))
        * (2**xi_val - 2 * gammainc(1 - xi_val, (1 + xi_val * y_red_val) ** (-1 / xi_val))),
        jnp.where(
            xi_null_mask,
            mu
            - y
            + sigma * (jnp.euler_gamma - jnp.log(2))
            - 2 * sigma * expi(exp_y_red_null),
            jnp.where(
                y1,
                sigma * (-y_red - 1 / xi_val) * (1 - 2 * gev_val)
                - sigma / xi_val * exp(lgamma(1 - xi_val)) * 2**xi_val,
                sigma * (-y_red - 1 / xi_val) * (1 - 2 * gev_val)
                - sigma / xi_val * exp(lgamma(1 - xi_val)) * (2**xi_val - 2),
            ),
        ),
    )


@partial(jit, static_argnums=(2, 3, 4))
def gev_crps_loss(param_pred, y_true, total_len, batch_size, n_clusters):
    """
    Compute the CRPS loss for the Generalized Extreme Value distribution.
    Based on Friedrichs and Thorarinsdottir (2012), adapted from https://github.com/louisPoulain/TCBench_0.1
    """

    mu, sigma, xi = jnp.split(param_pred, 3, axis=1)

    clusters_len = jnp.asarray(jax.tree_map(lambda x: x.shape[1], y_true))

    mu = jnp.repeat(mu, clusters_len, axis=1, total_repeat_length=total_len)
    sigma = jnp.repeat(sigma, clusters_len, axis=1, total_repeat_length=total_len)
    xi = jnp.repeat(xi, clusters_len, axis=1, total_repeat_length=total_len)

    coeffs = (
        jnp.repeat(
            1 / clusters_len, clusters_len, axis=0, total_repeat_length=total_len
        )
        / n_clusters
    )

    y_concat = jnp.concatenate(y_true, axis=1)

    crps = gev_crps(mu, sigma, xi, y_concat)

    crps = crps @ coeffs

    return crps.sum() / batch_size


def return_level(mu, sigma, xi, p):
    yp = -jnp.log(1 - p)
    xi_val = jnp.where(xi == 0, 0.5, xi)
    return jnp.where(
        xi == 0, mu - sigma * jnp.log(yp), mu - sigma / xi_val * (1 - yp ** (-xi_val))
    )


def return_level_loss(param_pred, y_true, total_len, batch_size, n_clusters, p):
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
