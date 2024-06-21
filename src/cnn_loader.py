import xarray as xr
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax

import pickle

import os

from cnn_block import CNN_Alpth
from cnn_train import train_loop, createTrainState, createBatches, train_step, calculate_loss, loss_and_CRPS
from utils import visualise_labels, visualise_features, visualise_loss_and_CRPS

class Experiment:
    """
    Class to load the data and preprocess it for the CNN model.
    """
    
    def __init__(self, experiment_file):
        # Largely inspired by Mauricio Experiment class https://github.com/maumlima/s2s-downscaling/blob/main/src/training/experiment.py
        with open(experiment_file, 'rb') as f:
            experiment = pickle.load(f)
        
        self.experiment_file = experiment_file
        
        self.train_files = experiment['train_files']
        self.val_files = experiment['val_files']
        self.test_files = experiment['test_files']
        self.train_label_files = experiment['train_label_files']
        self.val_label_files = experiment['val_label_files']
        self.test_label_files = experiment['test_label_files']
        self.scratch_dir = experiment['scratch_dir']
        self.plot_dir = experiment['plot_dir']
        self.features = experiment['features']
        self.label = experiment['label']
        self.batch_size = experiment['batch_size']
        self.cluster_file = experiment['cluster_file']
        self.storm_file = experiment['storm_file']
        self.storm_only = experiment['storm_only']
        self.rnginit = jax.random.key(0)
        self.rngshuffle = jax.random.key(1)
        self.epochs = experiment['epochs']
        self.learning_rate = experiment['learning_rate']
        self.regularisation = experiment['regularisation']
        self.alpha = experiment['alpha']
        self.n_best_states = experiment['n_best_states'] if 'n_best_states' in experiment.keys() else 3
        self.expnumber = experiment_file.split('_')[-1].split('.')[0]
        
        self.stations = xr.open_dataset(self.train_label_files[0]).station.values
        
        self.clusters = pd.read_csv(self.cluster_file, header = None)
        self.n_clusters = len(self.clusters)
        self.clusters = [list(np.intersect1d(self.clusters.iloc[i,:].dropna(), self.stations)) for i in range(self.n_clusters)]
        self.n_stations = np.array(list(map(len, self.clusters))).sum()
        
        self.scratch_folder = os.path.join(self.scratch_dir, "Experiment_" + self.expnumber)
        self.plot_folder = os.path.join(self.plot_dir, "Experiment_" + self.expnumber)
        
        os.makedirs(self.scratch_folder, exist_ok = True)
        os.makedirs(self.plot_folder, exist_ok = True)
        
        if self.storm_only:
            with open(self.storm_file, 'rb') as f:
                storms = pickle.load(f)
            self.dates = storms.index.unique()
            
    
    def create_inputs(self):
        """
        Create the input arrays for the CNN model.
        """
        print("Creating inputs...", flush = True)
        for files_set, labels_set, type_set in zip([self.train_files, self.val_files, self.test_files], [self.train_label_files, self.val_label_files, self.test_label_files], ['train', 'val', 'test']):
            npinputs_s = None
            npinputs_t = None
            nplabels = [None]*self.n_clusters
            for file, labels in zip(files_set, labels_set):
                inputs = xr.open_dataset(file, engine = "netcdf4")
                
                if self.storm_only:
                    loc_dates = pd.DatetimeIndex(np.intersect1d(inputs.time, self.dates))
                else:
                    loc_dates = inputs.time
                
                npinputs_t = np.array([[[time.day_of_year, time.hour, lt] for time in loc_dates] for lt in inputs.lead_time], dtype = np.float32).reshape(-1,3) if npinputs_t is None else np.concatenate([npinputs_t, np.array([[[time.day_of_year, time.hour, lt] for time in loc_dates] for lt in inputs.lead_time], dtype = np.float32).reshape(-1,3)], axis = 0)
                # Inputs
                tmp_var = None
                for ivar in range(len(self.features)):
                    # First obtaining corresponding data array
                    var = self.features[ivar]
                    if len(var.split('_')) == 2:
                        var, var_level = var.split('_')
                        var_level = int(var_level[:-3])
                        if var == "wind":
                            tmp = np.sqrt(inputs.sel(isobaricInhPa = var_level)['u10']**2 + inputs.sel(isobaricInhPa = var_level)['v10']**2)
                        else:
                            tmp = inputs.sel(isobaricInhPa = var_level)[var]
                    else:
                        if var == 'wind':
                            tmp = np.sqrt(inputs['u10']**2 + inputs['v10']**2)
                        else:
                            tmp = inputs[var]
                    tmp = tmp.sel(time = loc_dates)
                    
                    # Converting to numpy array
                    tmp = tmp.values
                    sh = tmp.shape
                    tmp = tmp.reshape((sh[0]*sh[1], sh[2], sh[3]))
                    # Now tmp should have a shape of (time*lead_time, lat, lon)
                    
                    # Normalizing
                    tmp = (tmp - self.mean[ivar])/self.std[ivar]
                    
                    tmp = np.expand_dims(tmp, axis = -1)
                    
                    tmp_var = tmp if tmp_var is None else np.concatenate([tmp_var, tmp], axis = -1)
                    # Now npinputs_s should have a shape of (time*lead_time, lat, lon, n_features)
                npinputs_s = tmp_var if npinputs_s is None else np.concatenate([npinputs_s, tmp_var], axis = 0)
                
                # Labels
                labels = xr.open_dataset(labels, engine = "netcdf4")
                labels = labels.sel(time = loc_dates)[self.label]
                for i in range(self.n_clusters):
                    labtmp = labels.sel(station = self.clusters[i]).values
                    labtmp = np.repeat(np.expand_dims(labtmp, axis = 0), len(inputs.lead_time), axis = 0)
                    labtmp = labtmp.reshape(labtmp.shape[0]*labtmp.shape[1], labtmp.shape[2])
                    nplabels[i] = labtmp if nplabels[i] is None else np.concatenate([nplabels[i], labtmp], axis = 0)
                
            jnpinputs_s = jnp.array(npinputs_s, dtype = jnp.float32)
            jnpinputs_t = jnp.array(npinputs_t, dtype = jnp.float32)
            jnplabels = tuple([jnp.array(nplabels[i], dtype = jnp.float32) for i in range(self.n_clusters)])
            
            os.makedirs(os.path.join(self.scratch_dir, "Experiment_" + self.expnumber), exist_ok = True)
            
            with open(os.path.join(self.scratch_dir, "Experiment_" + self.expnumber, f'{type_set}_set.pkl'), 'wb') as f:
                pickle.dump((jnpinputs_s, jnpinputs_t), f)
            with open(os.path.join(self.scratch_dir, "Experiment_" + self.expnumber, f'{type_set}_labels.pkl'), 'wb') as f:
                pickle.dump(jnplabels, f)
    
            
    def load_mean_std(self):
        """
        Get mean and standard values of features.
        """
        print("Computing mean and std of features...", end = " ", flush = True)
        self.mean = np.zeros((len(self.features)))
        mean_sq = np.zeros((len(self.features)))
        count = np.zeros((len(self.features)))
        
        for file in self.train_files:
            ds = xr.open_dataset(file, engine = "netcdf4")
            for ivar in range(len(self.features)):
                var = self.features[ivar]
                if len(var.split('_')) == 2:
                    var, var_level = var.split('_')
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(ds.sel(isobaricInhPa = var_level)['u10']**2 + ds.sel(isobaricInhPa = var_level)['v10']**2)
                    else:
                        tmp = ds.sel(isobaricInhPa = var_level)[var]
                else:
                    if var == 'wind':
                        tmp = np.sqrt(ds['u10']**2 + ds['v10']**2)
                    else:
                        tmp = ds[var]
                if self.storm_only:
                    count[ivar] += len(tmp.where(ds.time.isin(self.dates)).values)
                    self.mean[ivar] += tmp.where(ds.time.isin(self.dates)).mean().values*len(tmp.where(ds.time.isin(self.dates)).values)
                    mean_sq[ivar] += (tmp.where(ds.time.isin(self.dates))**2).mean().values*len(tmp.where(ds.time.isin(self.dates)).values)
                else:
                    count[ivar] += len(tmp.values)
                    self.mean[ivar] += (tmp**2).mean().values*len(tmp.values)
                    mean_sq[ivar] += (tmp**2).mean().values*len(tmp.values)   
            ds.close()
            
        self.mean = self.mean/count
        self.std = np.sqrt(mean_sq/count - self.mean**2)
        print("Done.", flush = True)
             
    
    def load_data(self):
        with open(os.path.join(self.scratch_folder, 'train_set.pkl'), 'rb') as f:
            self.train_s, self.train_t = pickle.load(f)
        with open(os.path.join(self.scratch_folder, 'val_set.pkl'), 'rb') as f:
            self.val_s, self.val_t = pickle.load(f)
        with open(os.path.join(self.scratch_folder, 'test_set.pkl'), 'rb') as f:
            self.test_s, self.test_t = pickle.load(f)
        with open(os.path.join(self.scratch_folder, 'train_labels.pkl'), 'rb') as f:
            self.train_l = pickle.load(f)
        with open(os.path.join(self.scratch_folder, 'val_labels.pkl'), 'rb') as f:
            self.val_l = pickle.load(f)
        with open(os.path.join(self.scratch_folder, 'test_labels.pkl'), 'rb') as f:
            self.test_l = pickle.load(f)
         
    
    def saveInformation(self):
        """
        Save the information of the experiment by printing it to a text file in self.plot_folder.
        """
        with open(os.path.join(self.plot_folder, 'Information.txt'), 'w') as f:
            f.write(str(self))
    
    
    def run(self):
        """
        Run the experiment.
        """
        data_exists = os.path.exists(os.path.join(self.scratch_folder, 'train_set.pkl')) and\
                      os.path.exists(os.path.join(self.scratch_folder, 'val_set.pkl')) and\
                      os.path.exists(os.path.join(self.scratch_folder, 'test_set.pkl')) and\
                      os.path.exists(os.path.join(self.scratch_folder, 'train_labels.pkl')) and\
                      os.path.exists(os.path.join(self.scratch_folder, 'val_labels.pkl')) and\
                      os.path.exists(os.path.join(self.scratch_folder, 'test_labels.pkl'))
        if not data_exists:
            self.load_mean_std()
            self.create_inputs()
        self.load_data()
        
        plot_exists = os.path.exists(os.path.join(self.plot_folder, 'LabelsDistribution.png')) and\
                      os.path.exists(os.path.join(self.plot_folder, 'FeaturesDistribution.png'))
        if not plot_exists:
            visualise_labels(self.train_l, self.val_l, self.test_l, os.path.join(self.plot_folder, 'LabelsDistribution.png'), self.label)
            visualise_features(self.train_s, self.val_s, self.test_s, os.path.join(self.plot_folder, 'FeaturesDistribution.png'), self.features)
        

        model_state = createTrainState(CNN_Alpth(n_clusters = self.n_clusters),
                                       self.rnginit,
                                       self.learning_rate,
                                        self.batch_size,
                                        len(self.features))
        
        best_states_with_scores, train_loss, val_loss, train_CRPS, val_CRPS = train_loop(model_state,
                                             self.train_s, self.train_t, self.train_l,
                                             self.val_s, self.val_t, self.val_l,
                                             self.batch_size, self.epochs, self.n_stations,
                                             self.rngshuffle, self.regularisation, self.alpha,
                                             self.n_best_states
                                             )
        
        # Create a custom state by taking the means of the params from best_states
    
        avg_params = jax.tree.map(lambda *x: jnp.stack(x).mean(axis = 0), *map(lambda x: x[0].params, best_states_with_scores))
        
        output_state = model_state.replace(params = avg_params)
        
        final_loss = 0
        final_CRPS = 0
        count = 0
        for x_s, x_t, y_true in createBatches(self.val_s, self.val_t, self.val_l,
                                              self.batch_size, self.rngshuffle):
            tmp_crps, tmp_loss = loss_and_CRPS(output_state, output_state.params, (x_s, x_t, y_true),
                                               self.batch_size, self.n_stations, self.regularisation, self.alpha)
            final_CRPS += tmp_crps
            final_loss += tmp_loss
            count += 1
        output_loss = final_loss/count
        output_CRPS = final_CRPS/count
        
        visualise_loss_and_CRPS(train_loss, val_loss, output_loss,
                                train_CRPS, val_CRPS, output_CRPS,
                                self.n_best_states, os.path.join(self.plot_folder, 'Loss.png'))
        
        with open(os.path.join(self.plot_folder, 'best_states.pkl'), 'wb') as f:
            pickle.dump((output_state.params, list(map(lambda x:x[0].params, best_states_with_scores))), f)
        
        self.output_CRPS = output_CRPS
        
        self.saveInformation()

    
    
    def saveExperimentFile(self):
        experiment_dict = {
            'train_files': self.train_files,
            'val_files': self.val_files,
            'test_files': self.test_files,
            'train_label_files': self.train_label_files,
            'val_label_files': self.val_label_files,
            'test_label_files': self.test_label_files,
            'scratch_dir': self.scratch_dir,
            'plot_dir': self.plot_dir,
            'features': self.features,
            'label': self.label,
            'batch_size': self.batch_size,
            'cluster_file': self.cluster_file,
            'storm_file': self.storm_file,
            'storm_only': self.storm_only,
            'rnginit': self.rnginit,
            'rngshuffle': self.rngshuffle,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'regularisation': self.regularisation,
            'alpha': self.alpha,
            'n_best_states': self.n_best_states,
        }
        
        with open(self.experiment_file, 'wb') as f:
            pickle.dump(experiment_dict, f)
    
    
    def __str__(self):
        result = f"Experiment n°{self.expnumber}\n\n"
        result += "Training files:\n"
        for file in self.train_files:
            result += f"{file}\n"
            
        result += "\nValidation files:\n"
        for file in self.val_files:
            result += f"{file}\n"
        
        result += "\nTest files:\n"
        for file in self.test_files:
            result += f"{file}\n"
            
        result += "\nTraining labels:\n"
        for file in self.train_label_files:
            result += f"{file}\n"
            
        result += "\nValidation labels:\n"
        for file in self.val_label_files:
            result += f"{file}\n"
        
        result += "\nTest labels:\n"
        for file in self.test_label_files:
            result += f"{file}\n"
        
        result += f"\nScratch directory: {self.scratch_folder}\n"
        
        result += f"\nPlotting directory: {self.plot_folder}\n"
        
        result += "\nFeatures: "
        for feature in self.features:
            result += f"{feature} "
        result += "\n"
        
        result += f"\nLabel: {self.label}\n"
        
        result += f"\nFiltered with storms only: {self.storm_only}\n"
        
        result += f"\nNumber of epochs: {self.epochs}"
        result += f"\nBatch size: {self.batch_size}"
        result += f"\nLearning rate: {self.learning_rate}"
        result += f"\nRegularization: {self.regularisation}"
        if self.regularisation is not None:
            result += f", alpha = {self.alpha}"
        
        result += f"\nNumber of states kept to create output state: {self.n_best_states}"
        
        if hasattr(self, 'output_CRPS'):
            result += f"\n\n --- CRPS of saved model on validation set: {self.output_CRPS} ---"
        
        return result
            
        
    
    @staticmethod
    def createExperimentFile(train_files, val_files, test_files,
                             train_label_files, val_label_files, test_label_files,
                             scratch_dir, plot_dir,
                             features, label,
                             batch_size,
                             cluster_file, storm_file, storm_only,
                             rnginit, rngshuffle, epochs, learning_rate,
                             regularisation, alpha,
                             n_best_states,
                             **kwargs):
        
        save_path = kwargs.pop('save_path', None)
        if save_path:
            if os.path.isdir(save_path):
                save_path = os.path.join(save_path, f'experiment_{len(os.listdir(save_path))}.pkl')
            else:
                if os.path.exists(save_path):
                    if not kwargs.pop('overwrite', False):
                        raise ValueError(f'{save_path} already exists. Set overwrite to True to overwrite it.')
        else:
            save_path = os.getcwd() 
            experiments = os.listdir(save_path)
            save_path = os.path.join(save_path, f'experiment_{len(experiments)}.pkl')
        
        experiment = {
            'train_files': train_files,
            'val_files': val_files,
            'test_files': test_files,
            'train_label_files': train_label_files,
            'val_label_files': val_label_files,
            'test_label_files': test_label_files,
            'scratch_dir': scratch_dir,
            'plot_dir': plot_dir,
            'features': features,
            'label': label,
            'batch_size': batch_size,
            'cluster_file': cluster_file,
            'storm_file': storm_file,
            'storm_only': storm_only,
            'rnginit': rnginit,
            'rngshuffle': rngshuffle,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'regularisation': regularisation,
            'alpha': alpha,
            'n_best_states': n_best_states,
        }
        
        # Add remaining arguments from kwargs
        for key, value in kwargs.items():
            experiment[key] = value
        
        with open(save_path, 'wb') as f:
            pickle.dump(experiment, f)
        
        print(f'Experiment file saved at {save_path}')
        
        return
    
