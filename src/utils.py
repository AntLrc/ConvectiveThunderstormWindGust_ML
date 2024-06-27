import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import jax.numpy as jnp
import jax
from cnn_losses import GEV, gevCRPSLoss, GEVpdf




def visualise_labels(train_labels, val_labels, test_labels, save_path, label_name):
    """
    Visualise labels distribution.
    """
    sns.set_theme(style="darkgrid")
    train_labels = pd.DataFrame(jnp.concat(train_labels, axis = None), columns = [label_name])
    train_labels['dataset'] = 'train'
    val_labels = pd.DataFrame(jnp.concat(val_labels, axis = None), columns = [label_name])
    val_labels['dataset'] = 'validation'
    test_labels = pd.DataFrame(jnp.concat(test_labels, axis = None), columns = [label_name])
    test_labels['dataset'] = 'test'
    dataset = pd.concat([train_labels, val_labels, test_labels])
    sns.histplot(data = dataset, x = label_name, hue = 'dataset', stat = 'proportion', binwidth = 0.5, common_norm = False)
    plt.savefig(save_path)
    plt.close()
    
def visualise_features(train_features, val_features, test_features, save_path, features_names):
    assert len(features_names) == train_features.shape[-1] == val_features.shape[-1] == test_features.shape[-1],\
        "Number of features names must be equal to the number of features in the dataset."
    
    sns.set_theme(style="darkgrid")
    # Create subplot with four rows (one for each set, plus one with all sets) and the number of features columns
    fig, axs = plt.subplots(4, len(features_names), figsize = (20,20))
    train_features = train_features.reshape(-1, train_features.shape[-1])
    val_features = val_features.reshape(-1, val_features.shape[-1])
    test_features = test_features.reshape(-1, test_features.shape[-1])
    
    train_features = pd.DataFrame(train_features, columns = features_names)
    val_features = pd.DataFrame(val_features, columns = features_names)
    test_features = pd.DataFrame(test_features, columns = features_names)
    
    train_features['dataset'] = 'train'
    val_features['dataset'] = 'validation'
    test_features['dataset'] = 'test'
    
    dataset = pd.concat([train_features, val_features, test_features])
    
    for i in range(len(features_names)):
        sns.histplot(data = dataset, x = features_names[i], hue = 'dataset', stat = 'proportion', common_norm = False, ax = axs[0,i])
        sns.histplot(data = train_features, x = features_names[i], stat = 'proportion', common_norm = False, ax = axs[1,i])
        sns.histplot(data = val_features, x = features_names[i], stat = 'proportion', common_norm = False, ax = axs[2,i])
        sns.histplot(data = test_features, x = features_names[i], stat = 'proportion', common_norm = False, ax = axs[3,i])
        
    plt.savefig(save_path)
    plt.close()
    
    
def visualise_loss(train_loss, val_loss,
                   output_loss, n_best_states,
                   save_path):
    """
    Visualise the training and validation loss.
    """
    sns.set_theme(style="darkgrid")
    epochs = np.arange(1, len(train_loss)+1)
    plt.plot(epochs, train_loss, label = 'train')
    plt.plot(epochs, val_loss, label = 'validation')
    plt.plot(epochs, [output_loss]*len(train_loss), label = "Loss of regularised output state", linestyle = '--')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)

def visualise_CRPS(train_CRPS, val_CRPS, output_CRPS, n_best_states, save_path):
    """
    Visualise the training and validation CRPS.
    """
    sns.set_theme(style="darkgrid")
    epochs = np.arange(1, len(train_CRPS)+1)
    plt.plot(epochs, train_CRPS, label = 'train')
    plt.plot(epochs, val_CRPS, label = 'validation')
    plt.plot(epochs, [output_CRPS]*len(train_CRPS), label = "CRPS of regularised output state", linestyle = '--')
    
    # Add n_best_states points corresponding to the n_best_states minima of val_CRPS
    best_args = np.argsort(val_CRPS)[:n_best_states]
    val_CRPS = np.array(val_CRPS)
    plt.scatter(best_args+1, val_CRPS[best_args], label = f"Best {n_best_states} validation CRPS", marker = 'x')
    
    plt.xlabel('Epochs')
    plt.ylabel('CRPS')
    plt.legend()
    plt.savefig(save_path)

def visualise_loss_and_CRPS(train_loss, val_loss, output_loss,
                            train_CRPS, val_CRPS, output_CRPS,
                            n_best_states, save_path):
    sns.set_theme(style="darkgrid")
    epochs = np.arange(1, len(train_loss)+1)
    fig, axs = plt.subplots(2, 1, figsize = (10,10))
    axs[0].plot(epochs, train_loss, label = 'train loss')
    axs[0].plot(epochs, val_loss, label = 'validation loss')
    axs[0].plot(epochs, [output_loss]*len(train_loss), label = "Loss of regularised output state", linestyle = '--')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    axs[1].plot(epochs, train_CRPS, label = 'train CRPS')
    axs[1].plot(epochs, val_CRPS, label = 'validation CRPS')
    axs[1].plot(epochs, [output_CRPS]*len(train_CRPS), label = "CRPS of regularised output state", linestyle = '--')
    
    # Add n_best_states points corresponding to the n_best_states minima of val_CRPS
    best_args = np.argsort(val_CRPS)[:n_best_states]
    val_CRPS = np.array(val_CRPS)
    axs[1].scatter(best_args+1, val_CRPS[best_args], label = f"Best {n_best_states} validation CRPS", marker = 'x')
    
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('CRPS')
    axs[1].legend()
    plt.savefig(save_path)
    plt.close()

def PIT_histogram(y_true, param_pred, save_path, title = None, leadtime = None):
    """
    Visualise the PIT histogram. For y_true and params_pred, follows the implementation
    of the neural networks:
    y_true:(jnp.array of shape (n_samples, n_observations) for cluster 1,
                jnp.array of shape (n_samples, n_observations) for cluster 2,
                ...)
    params_pred: jnp.array of shape (n_samples, n_params*n_clusters)
                    formatted as mu mu mu... sigma sigma sigma... xi xi xi...
    """
    # Beware: this function correctly selects the lead time ONLY IF the set on which it is used correspond to only one year
    mu, sigma, xi = jnp.split(param_pred, 3, axis = 1)
    
    clusters_len = jnp.asarray(jax.tree_map(lambda x: x.shape[1], y_true))
    
    n_clusters = len(clusters_len)
        
    mu = jnp.repeat(mu, clusters_len, axis = 1)
    sigma = jnp.repeat(sigma, clusters_len, axis = 1)
    xi = jnp.repeat(xi, clusters_len, axis = 1)
    
    y_true_concat = jnp.concatenate(y_true, axis = 1)
    total_len = y_true_concat.shape[1]
    
    sns.set_theme()
    PIT = GEV(mu, sigma, xi, y_true_concat)
    
    if leadtime is None:
        data = PIT.flatten()
        sns.histplot(data, stat = 'density', bins = 50)
        plt.plot([0,1], [1,1], color = "black")
        plt.vlines(x=[0.,1.], ymin=0, ymax=1, color = 'black')
        if title:
            plt.title(title)
    else:
        correspondance_lt = {
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            6:6,
            7:7,
            8:8,
            9:9,
            10:10,
            11:11,
            12:12,
            13:13,
            14:14,
            15:15,
            16:16,
            17:17,
            18:18,
            19:19,
            20:20,
            21:21,
            22:22,
            23:23,
            24:24,
            27:25,
            30:26,
            33:27,
            36:28,
            42:29,
            48:30,
            60:31,
            72:32
        }
        if not isinstance(leadtime, list):
            leadtime = [leadtime]
        ncols = (len(leadtime) + 1)//2
        fig, axs = plt.subplots(2, ncols, figsize = (5*ncols, 10), sharex=True, sharey=True)
        plt.tight_layout()
        ntimes = len(PIT)//33
        i = 0
        for lt in leadtime:
            data = PIT[correspondance_lt[lt]*ntimes:(correspondance_lt[lt]+1)*ntimes].flatten()
            dataCRPS = gevCRPSLoss(param_pred[correspondance_lt[lt]*ntimes:(correspondance_lt[lt]+1)*ntimes],
                                   tuple(map(lambda x: x[correspondance_lt[lt]*ntimes:(correspondance_lt[lt]+1)*ntimes], y_true)),
                                   total_len=total_len, batch_size = ntimes, n_clusters = n_clusters)
            sns.histplot(data, stat = 'density', bins = 50, ax = axs[i//ncols, i%ncols], label = f'CRPS: {dataCRPS:.3f}')
            axs[i//ncols, i%ncols].plot([0,1], [1,1], color = "black")
            axs[i//ncols, i%ncols].vlines(x=[0.,1.], ymin=0, ymax=1, color = 'black')
            axs[i//ncols, i%ncols].set_title(f'Rank histogram for lead time {lt} hours')
            axs[i//ncols, i%ncols].legend()
            i += 1
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.close()

def visualise_GEV(mu, sigma, xi, ys, save_path):
    """
    Visualise the Generalized Extreme Value distribution.
    """
    fig, axs = plt.subplots(2,1, figsize = (6.4, 9.6))
    sns.set_theme()
    x = np.linspace(-10, 50, 300)
    ypdf = GEVpdf(jnp.repeat(mu, 300), jnp.repeat(sigma,300), jnp.repeat(xi,300), x)
    ycdf = GEV(jnp.repeat(mu, 300), jnp.repeat(sigma,300), jnp.repeat(xi,300), x)
    
    ymax = ypdf.max()
    
    
    axs[0].plot(x, ypdf)
    # Add ticks corresponding to the true values
    for i in range(len(ys)):
        # Find i such that x[i] is the closest to ys[i]
        iref = np.argmin(np.abs(x - ys[i]))
        axs[0].vlines(x = x[iref], ymin = -ymax/50, ymax = ypdf[iref], color = 'black', linewidths = .5)
    
    # Plot empirical CDF
    axs[1].plot(x, ycdf)
    sns.ecdfplot(ys, ax = axs[1], stat = 'proportion', color = 'black')
    
    plt.savefig(save_path)
    plt.close()
    