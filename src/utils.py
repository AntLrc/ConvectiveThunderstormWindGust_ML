import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import jax.numpy as jnp

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
    