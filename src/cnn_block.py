import jax
import jax.numpy as jnp

from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn



class AddTrainableXi(nn.Module):
    constant_shape: tuple

    def setup(self):
        # Initialize the trainable constant with 0.1
        self.xi = self.param('xi', lambda rng, shape: jnp.full(shape, 0.1), self.constant_shape)

    def __call__(self, x):
        # Apply sigmoid activation function to the trainable constant
        xi_activated = nn.sigmoid(self.xi)*1.5 -0.5
        # Repeat the activated xi for the batch size
        xi_repeated = jnp.tile(xi_activated, (x.shape[0], 1))
        # Concatenate the original output x with the trainable constant xi
        return jnp.concatenate([x, xi_repeated], axis=-1)

class CNN_Alpth(nn.Module):
    kernel_size: Tuple[int] = (2,2)
    features: int = 16
    strides: int = (1,1)
    kernel_dilation: int = 1
    use_bias: bool = True
    n_clusters: int = 5
        
    
    @nn.compact
    def __call__(self, x_s, x_t):
        x_s = nn.Conv(features = self.features,
                      kernel_size = self.kernel_size,
                      strides = self.strides,
                      kernel_dilation = self.kernel_dilation,
                      use_bias = self.use_bias)(x_s)
        x_s = nn.leaky_relu(x_s, 0.2)
        x_s = nn.Conv(features = self.features*2,
                      kernel_size = self.kernel_size,
                      strides = self.strides,
                      kernel_dilation = self.kernel_dilation,
                      use_bias = self.use_bias)(x_s)
        x_s = nn.leaky_relu(x_s, 0.2)
        
        x_s = jnp.reshape(x_s, (x_s.shape[0], -1))

        x_t = nn.Dense(features = 64)(x_t)
        
        x = jnp.concatenate([x_s, x_t], axis = 1)
        
        x = nn.Dense(features = 10)(x)
        
        mu, sigma = jnp.split(x, 2, axis = 1)
        
        sigma = nn.softplus(sigma) + 1e-6
        
        x = jnp.concatenate([mu, sigma], axis = 1)
        
        
        x = AddTrainableXi(constant_shape = (self.n_clusters,))(x)
        
        return x