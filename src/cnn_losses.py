import jax
import jax.numpy as jnp

from jax.scipy.special import gammainc, expi
from jax.lax import lgamma
from jax.numpy import exp, log

from functools import partial
from jax import jit, custom_jvp

#Custom definition of exp(-exp) to avoid overflow. ONLY VALID FOR SIGMA AND MU "SLOWLY" VARYING
@custom_jvp
def doubleexp(mu,sigma,y):
    return exp(-exp(-(y-mu)/sigma))

@doubleexp.defjvp
def doubleexp_jvp(primals, tangents):
    mu,sigma,y = primals
    mudot,sigmadot,ydot = tangents
    primals_out = doubleexp(mu,sigma,y)
    muval = jnp.where(jnp.logical_or(primals_out == 0., primals_out == 1.), 0., mu)
    sigmaval = jnp.where(jnp.logical_or(primals_out == 0., primals_out == 1.), 1., sigma)
    yval = jnp.where(jnp.logical_or(primals_out == 0., primals_out == 1.), 0., y)
    
    resvar = jnp.where(jnp.logical_or(primals_out == 0., primals_out == 1.), 0.*(mudot+sigmadot+ydot), -primals_out*exp(-(yval-muval)/sigmaval)*(mudot/sigmaval - ydot/sigmaval + (yval-muval)*sigmadot/sigmaval**2))
    return primals_out, resvar


def GEV(mu, sigma, xi, y):
    """
    Computes the Generalized Extreme Value CDF. Raises error if xi equal to zero.
    """
    
    yred = (y - mu)/sigma
    xiNullMask = (xi == 0)
    xival = jnp.where(xiNullMask, 0.5, xi)
    
    y0 = jnp.logical_and(xi > 0, yred <= -1/xival)
    y1 = jnp.logical_and(xi < 0, yred >= -1/xival)
    
    yInBoundary = jnp.logical_or(
        jnp.logical_and(xi < 0, yred < -1/xival),
        jnp.logical_and(xi > 0, yred > -1/xival)
    )
    
    yredval = jnp.where(jnp.logical_or(y0,y1), (jnp.log(2)**(-xival) - 1)/xival, yred)
    
    return jnp.where(yInBoundary,
                    exp(-(1+xival*yredval)**(-1/xival)),
                    jnp.where(xiNullMask,
                              doubleexp(mu,sigma,y),
                              jnp.where(y1,
                                        1.,
                                        0.)))
    
    
def gevCRPS(mu, sigma, xi, y):
    """
    Compute the closed form of the Continuous Ranked Probability Score (CRPS) for the Generalized Extreme Value distribution.
    Based on Friedrichs and Thorarinsdottir (2012).
    """
    
    yred = (y - mu)/sigma
    xiNullMask = (xi == 0)
    xival = jnp.where(xiNullMask, 0.5, xi)
    
    gevval = GEV(mu, sigma, xi, y)
    
    y0 = jnp.logical_and(xi > 0, yred <= -1/xival)
    y1 = jnp.logical_and(xi < 0, yred >= -1/xival)

    yInBoundary = jnp.logical_and(
        jnp.logical_not(xiNullMask),
        jnp.logical_not(jnp.logical_or(y1, y0))
    )
    
    yredval = jnp.where(jnp.logical_or(y0,y1), (jnp.log(2)**(-xival) - 1)/xival, yred)
    
    expyrednull = -exp(jnp.where(xiNullMask, -yred, 0.))
    
    return jnp.where(yInBoundary,
                     sigma*(-yredval - 1/xival)*(1- 2*gevval) - sigma/xival*exp(lgamma(1-xival)) * (2**xival - 2*gammainc(1-xival,(1+xival*yredval)**(-1/xival))),
                    jnp.where(xiNullMask,
                              mu - y + sigma*(jnp.euler_gamma - jnp.log(2)) - 2 * sigma * expi(expyrednull),
                              jnp.where(y1,
                                        sigma*(-yred - 1/xival)*(1- 2*gevval) - sigma/xival*exp(lgamma(1-xival)) * 2**xival,
                                        sigma*(-yred - 1/xival)*(1- 2*gevval) - sigma/xival*exp(lgamma(1-xival)) * (2**xival - 2))))

@partial(jit, static_argnums = (2,3))
def gevCRPSLoss(param_pred, y_true, total_len, batch_size):
    """
    Compute the CRPS loss for the Generalized Extreme Value distribution.
    Based on Friedrichs and Thorarinsdottir (2012), adapted from https://github.com/louisPoulain/TCBench_0.1
    """
    
    mu, sigma, xi = jnp.split(param_pred, 3, axis = 1)
    
    clusters_len = jnp.asarray(jax.tree_map(lambda x: x.shape[1], y_true))
        
    mu = jnp.repeat(mu, clusters_len, axis = 1, total_repeat_length = total_len)
    sigma = jnp.repeat(sigma, clusters_len, axis = 1, total_repeat_length = total_len)
    xi = jnp.repeat(xi, clusters_len, axis = 1, total_repeat_length = total_len)
    
    coeffs = jnp.repeat(1/clusters_len, clusters_len, axis = 0, total_repeat_length = total_len)
    coeffs = jnp.repeat(jnp.expand_dims(coeffs, 0), batch_size, axis = 0)
    
    crps = gevCRPS(mu, sigma, xi, jnp.concat(y_true, axis = 1))
    
    crps = crps * coeffs
    
    return crps.sum()/batch_size

# def expi_tree(x):
#      sh = x.shape
#      xl = jnp.split(x, sh[1], axis = 1)
#      xl = jax.tree.map(lambda x:jnp.concatenate(jax.tree.map(expi, jnp.split(x, sh[0], axis = 0))), xl)
#      return jnp.concatenate(xl, axis = 1)
 
 




 