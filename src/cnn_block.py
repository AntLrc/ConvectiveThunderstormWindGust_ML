import jax
import jax.numpy as jnp

from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn

# Temporal NN
class Dense_NN(nn.Module):
    features: int
    hidden_layers: int = 0
    hidden_features: int = 64
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i in range(self.hidden_layers):
            x = nn.Dense(features = self.hidden_features * (self.hidden_layers - i))(x)
            x = nn.leaky_relu(x, 0.2)
        return nn.Dense(features = self.features)(x)


# Spatial NN
class DeepFCN(nn.Module):
    features: int
    hidden_layers: int = 0
    hidden_features: int = 64
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i in range(self.hidden_layers):
            x = nn.Dense(features = self.hidden_features)(x)
            x = nn.leaky_relu(x, 0.2)
        return nn.Dense(features = self.features)(x)

class Identity_NN(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool = False):
        return x

class Killed_NN(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool = False):
        return jnp.zeros((x.shape))
    
class Conv_NN(nn.Module):
    features: int
    kernel_size: Tuple[int] = (2,2)
    strides: int = 1
    dropout_rate: float = 0.
    hidden_layers: int = 0
    hidden_features: int = 64
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i in range(self.hidden_layers):
            x = nn.Conv(features = self.hidden_features * (self.hidden_layers - i),
                        kernel_size = self.kernel_size,
                        strides = self.strides)(x)
            x = nn.leaky_relu(x, 0.2)
            x = nn.Dropout(rate = self.dropout_rate)(x, deterministic = not training)
        x = nn.Conv(features = self.features,
                    kernel_size = self.kernel_size,
                    strides = self.strides)(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Dropout(rate = self.dropout_rate)(x, deterministic = not training)
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

class ConvNeXt_Blocks_NN(nn.Module):
    features: int
    num_blocks: int = 3
    layer_scale_init_value: float = 1e-6
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Conv(features = self.features,
                    kernel_size = (1,1))(x)
        for i in range(self.num_blocks):
            x = ConvNeXt_Block(features = self.features,
                               layer_scale_init_value = self.layer_scale_init_value)(x)
        return x

def img_to_patch(x, patch_size):
    # Padding first to create an image the height and width of which are multiples of the patch size
    x = jnp.pad(x,
                ((0,0),
                    ((x.shape[1] % patch_size)//2, (x.shape[1] % patch_size) - (x.shape[1] % patch_size)//2),
                    ((x.shape[2] % patch_size)//2, (x.shape[2] % patch_size) - (x.shape[2] % patch_size)//2),
                    (0,0))
                )
    B, H, W, C = x.shape
    x = x.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C)
    x = x.transpose((0, 1, 3, 2, 4, 5))
    x = x.reshape(B, -1, patch_size*patch_size*C)
    return x

class AttentionBlock(nn.Module):
    # Taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial15/Vision_Transformer.html
    embed_dim : int   # Dimensionality of input and attention feature vectors
    hidden_dim : int  # Dimensionality of hidden layer in feed-forward network
    num_heads : int   # Number of heads to use in the Multi-Head Attention block
    dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim)
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, training=True):
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + self.dropout(attn_out, deterministic=not training)

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not training)
        x = x + self.dropout(linear_out, deterministic=not training)
        return x

class VisionTransformer(nn.Module):
    # Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial15/Vision_Transformer.html
    embed_dim : int     # Dimensionality of input and attention feature vectors
    hidden_dim : int    # Dimensionality of hidden layer in feed-forward network
    num_heads : int     # Number of heads to use in the Multi-Head Attention block
    num_channels : int  # Number of channels of the input (3 for RGB)
    num_layers : int    # Number of layers to use in the Transformer
    patch_size : int    # Number of pixels that the patches have per dimension
    num_patches : int   # Maximum number of patches an image can have
    dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        # Layers/Networks
        self.input_layer = nn.Dense(self.embed_dim)
        self.transformer = [AttentionBlock(self.embed_dim,
                                           self.hidden_dim,
                                           self.num_heads,
                                           self.dropout_prob) for _ in range(self.num_layers)]
        self.dropout = nn.Dropout(self.dropout_prob)

        # Parameters/Embeddings
        self.pos_embedding = self.param('pos_embedding',
                                        nn.initializers.normal(stddev=1.0),
                                        (1, self.num_patches, self.embed_dim))


    def __call__(self, x, training=True):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add positional encoding
        x = x + self.pos_embedding[:,:T]

        # Apply Transformer
        x = self.dropout(x, deterministic=not training)
        for attn_block in self.transformer:
            x = attn_block(x, training=training)

        return x


# Distributional NNs
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

class SimpleBaseline(nn.Module):
    n_clusters: int = 5
    
    def setup(self):
        # Initialize the trainable constant with 0.1
        self.mu = self.param('mu', lambda rng, shape: jnp.full(shape, 5.), (self.n_clusters,))
        self.sigma = self.param('sigma', lambda rng, shape: jnp.full(shape, 1.), (self.n_clusters,))
        self.xi = self.param('xi', lambda rng, shape: jnp.full(shape, 0.1), (self.n_clusters,))
    
    @nn.compact
    def __call__(self, x_s, x_t, training: bool = False):
        mu_activated = self.mu
        sigma_activated = nn.softplus(self.sigma) + 1e-6
        xi_activated = nn.sigmoid(self.xi)*1.5 -0.5
        mu_repeated = jnp.tile(mu_activated, (x_t.shape[0], 1))
        sigma_repeated = jnp.tile(sigma_activated, (x_t.shape[0], 1))
        xi_repeated = jnp.tile(xi_activated, (x_t.shape[0], 1))
        return jnp.concatenate([mu_repeated, sigma_repeated, xi_repeated], axis = 1)
    
class DDNN_GEV(nn.Module):
    n_clusters: int = 5
    hidden_layers: int = 0
    hidden_features: int = 64
    
    @nn.compact
    def __call__(self, x_s, x_t, training: bool = False):
        x_s = jnp.reshape(x_s, (x_s.shape[0], -1))
        
        x = jnp.concatenate([x_s, x_t], axis = 1)
        for i in range(self.hidden_layers):
            x = nn.Dense(features = self.hidden_features * (self.hidden_layers - i))(x)
            x = nn.leaky_relu(x, 0.2)
        x = nn.Dense(features = self.n_clusters*2)(x)
        
        mu, sigma = jnp.split(x, 2, axis = 1)
        
        sigma = nn.softplus(sigma) + 1e-6 # epsilon to avoid instability
        
        x = jnp.concatenate([mu, sigma], axis = 1)
        
        x = AddTrainableXi(constant_shape = (self.n_clusters,))(x)
        
        return x

class Discrete_NN(nn.Module):
    n_stations: int
    hidden_layers: int = 0
    hidden_features: int = 64
    
    @nn.compact
    def __call__(self, x_s, x_t, training: bool = False):
        x_s = jnp.reshape(x_s, (x_s.shape[0], -1))
        
        x = jnp.concatenate([x_s, x_t], axis = 1)
        for i in range(self.hidden_layers):
            x = nn.Dense(features = self.hidden_features * (self.hidden_layers - i))(x)
            x = nn.leaky_relu(x, 0.2)
        return nn.Dense(features = self.n_stations)(x)

    
# Main model
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

