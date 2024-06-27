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

class AddTrainableSigma(nn.Module):
    constant_shape: tuple

    def setup(self):
        # Initialize the trainable constant with 0.1
        self.sigma = self.param('sigma', lambda rng, shape: jnp.full(shape, 1.), self.constant_shape)

    def __call__(self, x):
        # Apply softplus activation function to the trainable constant
        sigma_activated = nn.softplus(self.sigma) + 1e-6
        # Repeat the activated sigma for the batch size
        sigma_repeated = jnp.tile(sigma_activated, (x.shape[0], 1))
        # Concatenate the original output x with the trainable constant mu
        return jnp.concatenate([x, sigma_repeated], axis=-1)

class AddTrainableMu(nn.Module):
    constant_shape: tuple

    def setup(self):
        # Initialize the trainable constant with 0.1
        self.mu = self.param('mu', lambda rng, shape: jnp.full(shape, 5.), self.constant_shape)

    def __call__(self, x):
        mu_activated = self.mu
        # Repeat the activated mu for the batch size
        mu_repeated = jnp.tile(mu_activated, (x.shape[0], 1))
        # Return the original mu (and forgets xi)
        return mu_repeated



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
    
class CNN_Alpth_Dropout(nn.Module):
    kernel_size: Tuple[int] = (2,2)
    features: int = 16
    strides: int = (1,1)
    kernel_dilation: int = 1
    use_bias: bool = True
    n_clusters: int = 5
        
    
    @nn.compact
    def __call__(self, x_s, x_t, training: bool = False):
        x_s = nn.Conv(features = self.features,
                      kernel_size = self.kernel_size,
                      strides = self.strides,
                      kernel_dilation = self.kernel_dilation,
                      use_bias = self.use_bias)(x_s)
        x_s = nn.leaky_relu(x_s, 0.2)
        x_s = nn.Dropout(rate = 0.2)(x_s, deterministic = not training)
        x_s = nn.Conv(features = self.features*2,
                      kernel_size = self.kernel_size,
                      strides = self.strides,
                      kernel_dilation = self.kernel_dilation,
                      use_bias = self.use_bias)(x_s)
        x_s = nn.leaky_relu(x_s, 0.2)
        x_s = nn.Dropout(rate = 0.2)(x_s, deterministic = not training)
        
        x_s = jnp.reshape(x_s, (x_s.shape[0], -1))

        x_t = nn.Dense(features = 64)(x_t)
        x_t = nn.Dropout(rate = 0.2)(x_t, deterministic = not training)
        
        x = jnp.concatenate([x_s, x_t], axis = 1)
        
        x = nn.Dense(features = 10)(x)
        
        mu, sigma = jnp.split(x, 2, axis = 1)
        
        sigma = nn.softplus(sigma) + 1e-6
        
        x = jnp.concatenate([mu, sigma], axis = 1)
        
        
        x = AddTrainableXi(constant_shape = (self.n_clusters,))(x)
        
        return x
    
class SimpleBaseline(nn.Module):
    n_clusters: int = 5
    
    @nn.compact
    def __call__(self, x_s, x_t, training: bool = False):
        x  = nn.Dense(features = 1)(x_t)
        x = AddTrainableMu(constant_shape = (self.n_clusters,))(x)
        x = AddTrainableSigma(constant_shape = (self.n_clusters,))(x)
        x = AddTrainableXi(constant_shape = (self.n_clusters,))(x)
        return x
    
    
class ConvNeXt_Block(nn.Module):
    """
    Implementation from paper 'A ConvNet for the 2020s'
    """
    features: int
    layer_scale_init_value: float = 1e-6
    
    def setup(self):
        # Initialize the trainable constant with 0.1
        if self.layer_scale_init_value > 0:
            self.gamma = self.param('gamma', lambda rng, shape: jnp.full(shape, self.layer_scale_init_value), (self.features,))
        else:
            self.gamma = None

    
    @nn.compact
    def __call__(self, x, training: bool = False):
        xcop = x
        
        x = nn.Conv(features = self.features,
                    kernel_size = (7,7),
                    padding = 'SAME',
                    feature_group_count=self.features
                    )(x)
        
        x = nn.LayerNorm()(x)
        
        x = nn.Conv(features = self.features*4,
                    kernel_size=(1,1)
                    )(x)
        x = nn.gelu(x)
        x = nn.Conv(features = self.features,
                    kernel_size=(1,1)
                    )(x)
        
        if self.gamma is not None:
            x = x*self.gamma
        x = x + xcop
        
        return x
        
class TestNN(nn.Module):
    n_clusters: int = 5
    n_ConvNeXt_Blocks: int = 1
        
    
    @nn.compact
    def __call__(self, x_s, x_t, training: bool = False):
        x_s = nn.Conv(features = 64,
                      kernel_size = (1,1))(x_s)
        
        for _ in range(self.n_ConvNeXt_Blocks):
            x_s = ConvNeXt_Block(features = 64)(x_s)
        
        x_s = jnp.reshape(x_s, (x_s.shape[0], -1))
        
        x_t = nn.Dense(features = 64)(x_t)
        
        x = jnp.concatenate([x_s, x_t], axis = 1)
        
        x = nn.Dense(features = 10)(x)
        
        mu, sigma = jnp.split(x, 2, axis = 1)
        
        sigma = nn.softplus(sigma) + 1e-6
        
        x = jnp.concatenate([mu, sigma], axis = 1)
        
        x = AddTrainableXi(constant_shape = (self.n_clusters,))(x)
        
        return x

# Super simple Dense layer with training variable
class Dense(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        return nn.Dense(features = self.features)(x)

class Identity(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool = False):
        return x

class Killed(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool = False):
        return jnp.zeros((x.shape))

class Conv_NN(nn.Module):
    features: int
    kernel_size: Tuple[int]
    strides: Tuple[int]
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Conv(features = self.features,
                       kernel_size = self.kernel_size,
                       strides = self.strides)(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Conv(features = self.features*2,
                    kernel_size = self.kernel_size,
                    strides = self.strides)(x)
        x = nn.leaky_relu(x, 0.2)
        return x

class Conv(nn.Module):
    features: int
    kernel_size: Tuple[int]
    strides: Tuple[int]
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Conv(features = self.features,
                       kernel_size = self.kernel_size,
                       strides = self.strides)(x)
        return x

class ConvDropout(nn.Module):
    features: int
    kernel_size: Tuple[int]
    strides: Tuple[int]
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Conv(features = self.features,
                    kernel_size = self.kernel_size,
                    strides = self.strides)(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Dropout(rate = 0.2)(x, deterministic = not training)
        x = nn.Conv(features = self.features*2,
                    kernel_size = self.kernel_size,
                    strides = self.strides)(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Dropout(rate = 0.2)(x, deterministic = not training)
        return x


# ConvNeXt inspired architecture
class ConvNeXt_NN(nn.Module):
    """
    Inspired by ConvNeXt, adapted to specific architecture.
    """
    width: int = 20
    height: int = 34
    features: int = 96
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Training variable for consistency, although not needed as no dropout layers
        x = jnp.pad(x,
                    ((0,0),
                     ((56-self.width)//2, (56-self.width)//2),
                     ((56-self.height)//2, (56-self.height)//2),
                     (0,0))
                    )
        # 0-padding following
        # https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0263-7
        
        x = nn.Conv(features = 96,
                    kernel_size = (1,1))(x)
        # 1x1 convolution to augment the number of features
        
        for i in range(3):
            x = ConvNeXt_Block(features = 96)(x)
        
        # Downsampling
        x = nn.LayerNorm()(x)
        x = nn.Conv(features = 192,
                    kernel_size = (2,2),
                    strides = (2,2))(x)
        
        
        for i in range(3):
            x = ConvNeXt_Block(features = 192)(x)
            
        # Downsampling
        x = nn.LayerNorm()(x)
        x = nn.Conv(features = 384,
                    kernel_size = (2,2),
                    strides = (2,2))(x)
        
        for i in range(9):
            x = ConvNeXt_Block(features = 384)(x)
        
        # Downsampling
        x = nn.LayerNorm()(x)
        x = nn.Conv(features = 768,
                    kernel_size = (2,2),
                    strides = (2,2))(x)
        
        for i in range(3):
            x = ConvNeXt_Block(features = 768)(x)
            
        x = nn.LayerNorm()(x)
        
        return x


class DDNN_GEV(nn.Module):
    n_clusters: int = 5
    
    @nn.compact
    def __call__(self, x_s, x_t, training: bool = False):
        x_s = jnp.reshape(x_s, (x_s.shape[0], -1))
        
        x = jnp.concatenate([x_s, x_t], axis = 1)
        
        x = nn.Dense(features = self.n_clusters*2)(x)
        
        mu, sigma = jnp.split(x, 2, axis = 1)
        
        sigma = nn.softplus(sigma) + 1e-6 # epsilon to avoid instability
        
        x = jnp.concatenate([mu, sigma], axis = 1)
        
        x = AddTrainableXi(constant_shape = (self.n_clusters,))(x)
        
        return x
        

class AlpTh_NN(nn.Module):
    n_clusters: int = 5
    Spatial_NN: Callable = ConvNeXt_NN(width = 20, height = 34)
    Temporal_NN: Callable = nn.Dense(features = 64)
    DDNN: Callable = DDNN_GEV(n_clusters = 10)
    
    @nn.compact
    def __call__(self, x_s, x_t, training: bool = False):
        x_s = self.Spatial_NN(x_s, training = training)
        x_t = self.Temporal_NN(x_t, training = training)
        x = self.DDNN(x_s, x_t, training = training)
        return x

