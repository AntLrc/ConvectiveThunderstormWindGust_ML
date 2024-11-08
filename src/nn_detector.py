import jax
import jax.numpy as jnp

import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn

from flax.training import train_state
from collections import defaultdict
import optax

from jax import grad, jit, vmap

# Inspired by https://github.com/8bitmp3/JAX-Flax-Tutorial-Image-Classification-with-Linen

# Define the NN: a CNN and a FCN followed by a FCN with as many outputs as the number of clusters, +1 output.
# Output will be a classification problem: 1 for the cluster if a storm is detected in the cluster, 0 otherwise.
# If no storm is detected, the last output will be 1.


# Create the labels
def create_labels(StationWithStorms, clusters, train_dates, val_dates, test_dates):
  """
  StationWithStorms: defaultdict(set) with timesteps as keys and sets of stations with storms as values.
  clusters: list of list of stations corresponding to the clusters.
  """
  res = []
  for dates in [train_dates, val_dates, test_dates]:
    possibleDates = pd.DatetimeIndex(StationWithStorms.keys()).intersection(dates)
    possibleStations = {k:StationWithStorms[k] for k in possibleDates}
    labels = np.zeros((len(possibleStations), len(clusters) + 1))
    for i, (key, value) in enumerate(possibleStations.items()):
      for j, cluster in enumerate(clusters):
        if len(value.intersection(cluster)) > 0:
          labels[i, j] = 1
    res.append(jnp.array(labels))


  return res



# Define the NNs block
class CNN(nn.Module):
    features: int = 32
    kernel_size: int = 3
    strides: int = 1
    padding: int = 0
    use_bias: bool = True
    layers: int = 2
    conv: nn.Module = nn.Conv

    @nn.compact
    def __call__(self, x, train = False):
        for i in range(self.layers):
            x = self.conv(self.features*(i+1),
                          self.kernel_size,
                          self.strides,
                          self.padding,
                          self.use_bias)(x)
            x = nn.leaky_relu(x)
        return x

class FCN(nn.Module):
    features: int = 64
    use_bias: bool = True
    layers: int = 1
    dense: nn.Module = nn.Dense

    @nn.compact
    def __call__(self, x, train = False):
        for i in range(self.layers):
            x = self.dense(self.features*(i+1), self.use_bias)(x)
            x = nn.leaky_relu(x)
        return x
    
class Detector(nn.Module):
    cnn: nn.Module = CNN()
    fcn: nn.Module = FCN()
    out: nn.Module = nn.Dense(features = 5)
    

    @nn.compact
    def __call__(self, x_s, x_t, train = False):
        x_s = self.cnn(x_s, train)
        x_t = self.fcn(x_t, train)
        
        x_s = jnp.reshape(x_s, (x_s.shape[0], -1))
        
        x = jnp.concatenate([x_s, x_t], axis = 1)
        
        x = self.out(x)
        
        return x

# Addtional spatial NN to test
class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal'),
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal'),
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(2, 2), kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal'))(x)

        x_out = self.act_fn(z + x)
        return x_out

class PreActResNetBlock(ResNetBlock):

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal'),
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal'),
                    use_bias=False)(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal'),
                        use_bias=False)(x)

        x_out = z + x
        return x_out



class ResNet(nn.Module):
    act_fn : callable = nn.relu
    block_class : nn.Module = ResNetBlock
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal'), use_bias=False)(x)
        if self.block_class == ResNetBlock:  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))
        return x

class DenseLayer(nn.Module):
    bn_size : int  # Bottleneck size (factor of growth rate) for the output of the 1x1 convolution
    growth_rate : int  # Number of output channels of the 3x3 convolution
    act_fn : callable  # Activation function

    @nn.compact
    def __call__(self, x, train=True):
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.bn_size * self.growth_rate,
                    kernel_size=(1, 1),
                    kernel_init=nn.initializers.kaiming_normal(),
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.growth_rate,
                    kernel_size=(3, 3),
                    kernel_init=nn.initializers.kaiming_normal(),
                    use_bias=False)(z)
        x_out = jnp.concatenate([x, z], axis=-1)
        return x_out

class DenseBlock(nn.Module):
    num_layers : int  # Number of dense layers to apply in the block
    bn_size : int  # Bottleneck size to use in the dense layers
    growth_rate : int  # Growth rate to use in the dense layers
    act_fn : callable  # Activation function to use in the dense layers

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.num_layers):
            x = DenseLayer(bn_size=self.bn_size,
                           growth_rate=self.growth_rate,
                           act_fn=self.act_fn)(x, train=train)
        return x

class TransitionLayer(nn.Module):
    c_out : int  # Output feature size
    act_fn : callable  # Activation function

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = nn.Conv(self.c_out,
                    kernel_size=(1, 1),
                    kernel_init=nn.initializers.kaiming_normal(),
                    use_bias=False)(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x

class DenseNet(nn.Module):
    act_fn : callable = nn.relu
    num_layers : tuple = (6, 6, 6, 6)
    bn_size : int = 2
    growth_rate : int = 16

    @nn.compact
    def __call__(self, x, train=True):
        c_hidden = self.growth_rate * self.bn_size  # The start number of hidden channels

        x = nn.Conv(c_hidden,
                    kernel_size=(3, 3),
                    kernel_init=nn.initializers.kaiming_normal())(x)

        for block_idx, num_layers in enumerate(self.num_layers):
            x = DenseBlock(num_layers=num_layers,
                           bn_size=self.bn_size,
                           growth_rate=self.growth_rate,
                           act_fn=self.act_fn)(x, train=train)
            c_hidden += num_layers * self.growth_rate
            if block_idx < len(self.num_layers)-1:  # Don't apply transition layer on last block
                x = TransitionLayer(c_out=c_hidden//2,
                                    act_fn=self.act_fn)(x, train=train)
                c_hidden //= 2

        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = x.mean(axis=(1, 2))
        return x

# Define the loss function
def compute_metrics(logits, labels,
                    threshold = 0.5):
  loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
  accuracy = jnp.mean(jnp.all(jnp.where(nn.sigmoid(logits) >= threshold, 1., 0.) == labels, axis = 1))
  metrics = {
      'loss': loss,
      'accuracy': accuracy
  }
  return metrics

# Extension of trainstate
class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any

#Define the training and evaluation steps
@jax.jit
def train_step(state, x_s, x_t, y,
               alpha = 0., threshold = 0.5):
  
  def loss_fn(params):
    logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, x_s, x_t, train = True, mutable=['batch_stats'])
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
        logits=logits, 
        labels=y))
    l2_penalty = 0.5 * sum((w**2).mean() for w in jax.tree.leaves(params))
    return loss - alpha*l2_penalty, logits
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, y, threshold = threshold)
  return state, metrics

@jax.jit
def eval_step(state, params, x_s, x_t, y,
              threshold = 0.5):
  logits = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, x_s, x_t)
  return compute_metrics(logits, y, threshold = threshold)

def train_epoch(state, train_ds, batch_size, epoch, rng,
                alpha = 0., threshold = 0.5):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  batch_metrics = []

  for perm in perms:
    x_s = train_ds['image'][perm, ...]
    x_t = train_ds['time'][perm, ...]
    y = train_ds['label'][perm, ...]
    state, metrics = train_step(state, x_s, x_t, y, alpha, threshold)
    batch_metrics.append(metrics)

  training_batch_metrics = jax.device_get(batch_metrics)
  training_epoch_metrics = {
      k: np.mean([metrics[k] for metrics in training_batch_metrics])
      for k in training_batch_metrics[0]}

  print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

  return state, training_epoch_metrics

def eval_model(state, params, test_ds,
               threshold = 0.5):
  x_s = test_ds['image']
  x_t = test_ds['time']
  y = test_ds['label']
  metrics = eval_step(state, params, x_s, x_t, y, threshold)
  metrics = jax.device_get(metrics)
  eval_summary = jax.tree_map(lambda x: x.item(), metrics)
  return eval_summary['loss'], eval_summary['accuracy']

def __main__():
  import argparse
  
  parser = argparse.ArgumentParser(description='Train a CNN for storm detection')
  
  parser.add_argument('--folder', type=str, help='Path to the dir containing the datasets and to which will be plotted the results')
  parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
  parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
  parser.add_argument('--penalty', type=float, default=0., help='L2 penalty')
  parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for the classification')
  parser.add_argument('--model', type=str, default='cnn', help='Model to use')
  
  args = parser.parse_args()
  
  folder = args.folder
  
  # Save the command line as a text file in folder
  
  with open(os.path.join(folder, "command.txt"), "w") as f:
    f.write(f"run /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/src/nn_detector.py --folder {folder} --epochs {args.epochs} --learning-rate {args.learning_rate} --penalty {args.penalty} --threshold {args.threshold} --model {args.model}")
  
  set_path = os.path.join(folder, "sets.pkl")
  
  with open(set_path, "rb") as f:
    train_ds, val_ds, test_ds = pickle.load(f)
  
  # Plots
  sns.set_theme()
  xticks = np.arange(1, train_ds['label'].shape[1] + 1, dtype = int)
  xlabs = [str(x) for x in xticks]
  fig, axs = plt.subplots(1, 3, figsize = (15, 5), sharey=True)
  for i, ds in enumerate([train_ds, val_ds, test_ds]):
    weights = ds['label'].sum(axis = 0) / ds['label'].shape[0]
    sns.histplot(x = list(map(str, xticks)), weights = weights, ax=axs[i])
    axs[i].set_title(f"Number of storms in the dataset {['train', 'val', 'test'][i]}")
    axs[i].set_xlabel("Cluster")
  plt.savefig(os.path.join(folder, "labels.png"))
  plt.close()
  
  rng = jax.random.PRNGKey(10)
  rng, init_rng = jax.random.split(rng)
  
  learning_rate = args.learning_rate
  
  if args.model.lower() == 'cnn':
    tx = optax.adam(learning_rate=learning_rate)
    model = Detector(
      cnn = CNN(features = 32,
                kernel_size = 3,
                strides = 1,
                padding = 0,
                use_bias = True,
                layers = 2),
      fcn = FCN(features = 64,
                use_bias = True,
                layers = 1),
      out = nn.Dense(features = len(xticks))
    )
  elif args.model.lower() == 'resnet':
    tx = optax.sgd(learning_rate=learning_rate, momentum = 0.9)
    model = Detector(
      cnn = ResNet(),
      fcn = FCN(features = 64,
                use_bias = True,
                layers = 1),
      out = nn.Dense(features = len(xticks))
    )
  elif args.model.lower() == 'densenet':
    tx = optax.adam(learning_rate=learning_rate)
    model = Detector(
      cnn = DenseNet(),
      fcn = FCN(features = 64,
                use_bias = True,
                layers = 1),
      out = nn.Dense(features = len(xticks))
    )
  elif args.model.lower() == 'preactresnet':
    tx = optax.sgd(learning_rate=learning_rate, momentum = 0.9)
    model = Detector(
      cnn = ResNet(block_class = PreActResNetBlock),
      fcn = FCN(features = 64,
                use_bias = True,
                layers = 1),
      out = nn.Dense(features = len(xticks))
    )
  spshape = train_ds['image'].shape
  spshape = (1, *spshape[1:])
  variables = model.init(init_rng, jnp.ones(spshape), jnp.ones([1,4]))
  params = variables['params']
  batch_stats = variables.get('batch_stats', {})
  
  alpha = args.penalty
  
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
  
  num_epochs = args.epochs
  batch_size = 32
  
  train_metrics_array = defaultdict(list)
  test_metrics_array = defaultdict(list)
  
  
  for epoch in range(1, num_epochs + 1):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state, train_metrics = train_epoch(state, train_ds, batch_size, epoch, input_rng, alpha = alpha, threshold=args.threshold)
    for k, v in train_metrics.items():
      train_metrics_array[k].append(v)
    # Evaluate on the test set after each training epoch
    test_loss, test_accuracy = eval_model(state, state.params, test_ds, threshold = args.threshold)
    test_metrics_array['loss'].append(test_loss)
    test_metrics_array['accuracy'].append(test_accuracy)
    print('\tTesting - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))
  
  # Plot the evolution of the metrics
  fig, axs = plt.subplots(2, 1, figsize = (10, 10), sharex=True)
  axs[0].plot(train_metrics_array['loss'], label = 'Training loss')
  axs[0].plot(test_metrics_array['loss'], label = 'Testing loss')
  axs[0].set_title('Loss')
  axs[0].legend()
  axs[0].set_xlabel('Epoch')
  
  axs[1].plot(train_metrics_array['accuracy'], label = 'Training accuracy')
  axs[1].plot(test_metrics_array['accuracy'], label = 'Testing accuracy')
  axs[1].set_title('Accuracy')
  axs[1].legend()
  axs[1].set_xlabel('Epoch')
  
  plt.savefig(os.path.join(folder, "metrics.png"))
  plt.close()
  
  # Plot the outputs of the model
  x_s = test_ds['image']
  x_t = test_ds['time']
  y = test_ds['label']
  
  logits = model.apply({'params': state.params, 'batch_stats':state.batch_stats}, x_s, x_t)
  logits = jax.device_get(logits)
  res = nn.sigmoid(logits)
  res = jnp.where(res >= args.threshold, 1., 0.)
  
  # Replace all values in logits by 0 except the maximum
  
  fig, axs = plt.subplots(1, 2, figsize = (15, 5), sharey = True)
  
  weights = res.sum(axis = 0) / res.shape[0]
  sns.histplot(x = list(map(str, xticks)), weights=weights, ax=axs[0])
  axs[0].set_xlabel("Cluster")
  axs[0].set_title("Predicted labels")
  
  weights = y.sum(axis = 0) / y.shape[0]
  sns.histplot(x = list(map(str, xticks)), weights=weights, ax=axs[1])
  axs[1].set_xlabel("Cluster")
  axs[1].set_title("True labels")
  
  plt.savefig(os.path.join(folder, "outputs.png"))
  plt.close()
  

if __name__ == "__main__":
  __main__()