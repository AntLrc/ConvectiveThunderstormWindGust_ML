import xarray as xr
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
from flax import linen as nn


import pickle

import os

from cnn_block import *
from cnn_train import train_loop, createTrainState, createBatches, train_step, calculate_loss, loss_and_CRPS
from cnn_losses import gevCRPSLoss
from utils import visualise_labels, visualise_features, visualise_loss_and_CRPS, PIT_histogram, visualise_GEV


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
        self.model_spatial = experiment['model_spatial'] if 'model_spatial' in experiment.keys() else "Conv"
        self.model_temporal = experiment['model_temporal'] if 'model_temporal' in experiment.keys() else "Dense64"
        self.model_distributional = experiment['model_distributional'] if 'model_distributional' in experiment.keys() else "DDNN"
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
        
        self.clusters_len = np.array(list(map(len, self.clusters)))
        coeffs_corr = np.concatenate(tuple(jnp.arange(1,self.clusters_len[i]+1) for i in range(len(self.clusters_len))))
        self.clusters_len = np.repeat(self.clusters_len, self.clusters_len)
        self.coeffs_corr = (2*coeffs_corr - self.clusters_len - 1)/(self.clusters_len**2)
        
        self.scratch_folder = os.path.join(self.scratch_dir, "Experiment_" + self.expnumber)
        self.plot_folder = os.path.join(self.plot_dir, "Experiment_" + self.expnumber)
        
        os.makedirs(self.scratch_folder, exist_ok = True)
        os.makedirs(self.plot_folder, exist_ok = True)
        
        if self.storm_only:
            with open(self.storm_file, 'rb') as f:
                storms = pickle.load(f)
            self.dates = storms.index.unique()
            
        if "output_CRPS" in experiment.keys():
            self.output_CRPS = experiment['output_CRPS']
        if "test_CRPS" in experiment.keys():
            self.test_CRPS = experiment['test_CRPS']
            
    
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
                
                # Sinusoidal encoding of time
                tmp_t = np.array([[[(time.day_of_year - 242)/107, # Encoding of day of year between -1 and +1
                                    np.cos(time.hour*np.pi/12),
                                    np.sin(time.hour*np.pi/12),
                                    lt/72] for time in loc_dates] for lt in inputs.lead_time],
                                 dtype = np.float32).reshape(-1,4)
                
                npinputs_t = tmp_t if npinputs_t is None else np.concatenate([npinputs_t, tmp_t], axis = 0)
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
        print("Done.", flush = True)
    
            
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
    
    
    def run(self, preComputed = False):
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
        print("Loading data...", flush = True)
        self.load_data()
        
        self.train_corr = jnp.sort(jnp.concatenate(self.train_l, axis = 1))@self.coeffs_corr / self.n_clusters
        self.val_corr = jnp.sort(jnp.concatenate(self.val_l, axis = 1))@self.coeffs_corr / self.n_clusters
        self.test_corr = jnp.sort(jnp.concatenate(self.test_l, axis = 1))@self.coeffs_corr / self.n_clusters
        
        plot_exists = os.path.exists(os.path.join(self.plot_folder, 'LabelsDistribution.png')) and\
                      os.path.exists(os.path.join(self.plot_folder, 'FeaturesDistribution.png'))
        if not plot_exists:
            print("Visualising data...", flush = True)
            visualise_labels(self.train_l, self.val_l, self.test_l, os.path.join(self.plot_folder, 'LabelsDistribution.png'), self.label)
            visualise_features(self.train_s, self.val_s, self.test_s, os.path.join(self.plot_folder, 'FeaturesDistribution.png'), self.features)
        
        Spatial_NN = Conv_NN(features = 16, kernel_size = (2,2), strides = (1,1)) if self.model_spatial == "Conv" \
            else ConvDropout(features = 16, kernel_size = (2,2), strides = (1,1)) if self.model_spatial == "ConvDrop" \
            else ConvNeXt_NN(width = 20, height = 34) if self.model_spatial == "ConvNeXt" \
            else nn.Sequential((Conv(features = 96, kernel_size = (1,1), strides = (1,1)),
                                ConvNeXt_Block(features = 96))) if self.model_spatial == "ConvNeXt_Block" \
            else nn.Sequential((Conv(features = 96, kernel_size = (1,1), strides = (1,1)),
                                ConvNeXt_Block(features = 96),
                                ConvNeXt_Block(features = 96),
                                ConvNeXt_Block(features = 96))) if self.model_spatial == "3_ConvNeXt_Block" \
            else Identity()
        
        Temporal_NN = Dense(features = 32) if self.model_temporal == "Dense32" \
            else Dense(features = 64) if self.model_temporal == "Dense64" \
            else Killed() if self.model_temporal == "Killed" \
            else Identity()
        
        DDNN = DDNN_GEV(n_clusters = self.n_clusters) if self.model_distributional == "DDNN" \
            else SimpleBaseline(n_clusters = self.n_clusters)
        
        model = AlpTh_NN(
            n_clusters = self.n_clusters,
            Spatial_NN = Spatial_NN,
            Temporal_NN = Temporal_NN,
            DDNN = DDNN
        )
                    
        model_state = createTrainState(model,
                                    self.rnginit,
                                    self.learning_rate,
                                        self.batch_size,
                                        len(self.features))
            
        
        if not preComputed:
            print("Beginning training...", flush = True)
            
            best_states_with_scores, train_loss, val_loss, train_CRPS, val_CRPS = train_loop(model_state,
                                                self.train_s, self.train_t, self.train_l, self.train_corr,
                                                self.val_s, self.val_t, self.val_l, self.val_corr,
                                                self.batch_size, self.epochs, self.n_stations, self.n_clusters,
                                                self.rngshuffle, self.regularisation, self.alpha,
                                                self.n_best_states
                                                )
            
            # Create a custom state by taking the means of the params from best_states
        
            avg_params = jax.tree.map(lambda *x: jnp.stack(x).mean(axis = 0), *map(lambda x: x[0].params, best_states_with_scores))
            
            output_state = model_state.replace(params = avg_params)
        else:
            with open(os.path.join(self.plot_folder, 'best_states.pkl'), 'rb') as f:
                output_params, best_states_with_scores = pickle.load(f)
            output_state = model_state.replace(params = output_params)
        
        final_loss = 0
        final_CRPS = 0
        count = 0
        for x_s, x_t, y_true, corr in createBatches(self.val_s, self.val_t, self.val_l, self.val_corr,
                                              self.batch_size, self.rngshuffle):
            tmp_crps, tmp_loss = loss_and_CRPS(output_state, output_state.params, (x_s, x_t, y_true),
                                               self.batch_size, self.n_stations, self.n_clusters,
                                               self.regularisation, self.alpha)
            tmp_crps -= corr.mean()
            final_CRPS += tmp_crps
            final_loss += tmp_loss
            count += 1
        output_loss = final_loss/count
        output_CRPS = final_CRPS/count
        
        if not preComputed:
            visualise_loss_and_CRPS(train_loss, val_loss, output_loss,
                                    train_CRPS, val_CRPS, output_CRPS,
                                    self.n_best_states, os.path.join(self.plot_folder, 'Loss.png'))
        
        if not preComputed:
            with open(os.path.join(self.plot_folder, 'best_states.pkl'), 'wb') as f:
                pickle.dump((output_state.params, list(map(lambda x:x[0].params, best_states_with_scores))), f)
        
        self.output_CRPS = output_CRPS
        
        self.test_CRPS = self.PIT(leadtime=[0,1,2,3,4,5,6,12,24,36,48,72])
        
        self.saveInformation()
        self.saveExperimentFile()
        self.saveSummary()
        self.plotGEV()

    
    def PIT(self, model_state = None, leadtime = None, ensemble = 'test'):
        assert os.path.exists(os.path.join(self.plot_folder, 'best_states.pkl')), "Run the experiment first."
        if not hasattr(self, 'test_s'):
            self.load_data()
        
        with open(os.path.join(self.plot_folder, 'best_states.pkl'), 'rb') as f:
            output_params, best_states = pickle.load(f)
        
        set_s = self.test_s if ensemble == 'test' else self.val_s
        set_t = self.test_t if ensemble == 'test' else self.val_t
        set_l = self.test_l if ensemble == 'test' else self.val_l
        set_corr = self.test_corr if ensemble == 'test' else self.val_corr
        
        ntimes = len(set_s)//33 #33 different lead_times
        
        Spatial_NN = Conv_NN(features = 16, kernel_size = (2,2), strides = (1,1)) if self.model_spatial == "Conv" \
            else ConvDropout(features = 16, kernel_size = (2,2), strides = (1,1)) if self.model_spatial == "ConvDrop" \
            else ConvNeXt_NN(width = 20, height = 34) if self.model_spatial == "ConvNeXt" \
            else nn.Sequential((Conv(features = 96, kernel_size = (1,1), strides = (1,1)),
                                ConvNeXt_Block(features = 96))) if self.model_spatial == "ConvNeXt_Block" \
            else nn.Sequential((Conv(features = 96, kernel_size = (1,1), strides = (1,1)),
                                ConvNeXt_Block(features = 96),
                                ConvNeXt_Block(features = 96),
                                ConvNeXt_Block(features = 96))) if self.model_spatial == "3_ConvNeXt_Block" \
            else Identity()
        
        Temporal_NN = Dense(features = 32) if self.model_temporal == "Dense32" \
            else Dense(features = 64) if self.model_temporal == "Dense64" \
            else Identity()
        
        DDNN = DDNN_GEV(n_clusters = self.n_clusters) if self.model_distributional == "DDNN" \
            else SimpleBaseline(n_clusters = self.n_clusters)
        
        model_final = AlpTh_NN(
            n_clusters = self.n_clusters,
            Spatial_NN = Spatial_NN,
            Temporal_NN = Temporal_NN,
            DDNN = DDNN
        )
                    
        state_final = createTrainState(model_final,
                                    self.rnginit,
                                    self.learning_rate,
                                    ntimes,
                                    len(self.features))
        
        state_final = state_final.replace(params = output_params)
        
        param_pred = jnp.concatenate([model_final.apply(state_final.params, set_s[ntimes*i:ntimes*(i+1)], set_t[ntimes*i:ntimes*(i+1)]) for i in range(len(set_s)//ntimes)], axis = 0)
        # Error if I try a convolution on the whole batch, I don't know why... but this works soooo
        PIT_histogram(set_l,
                      param_pred,
                      os.path.join(self.plot_folder, "PIT_histogram.png"),
                      title = "PIT histogram", leadtime = leadtime)
        
        return gevCRPSLoss(param_pred, set_l, self.n_stations, len(set_s), self.n_clusters) - set_corr.mean()
    
    
    def plotGEV(self):
        assert os.path.exists(os.path.join(self.plot_folder, 'best_states.pkl')), "Run the experiment first."
        if not hasattr(self, 'test_s'):
            self.load_data()
        
        with open(os.path.join(self.plot_folder, 'best_states.pkl'), 'rb') as f:
            output_params, best_states = pickle.load(f)
        
        Spatial_NN = Conv_NN(features = 16, kernel_size = (2,2), strides = (1,1)) if self.model_spatial == "Conv" \
            else ConvDropout(features = 16, kernel_size = (2,2), strides = (1,1)) if self.model_spatial == "ConvDrop" \
            else ConvNeXt_NN(width = 20, height = 34) if self.model_spatial == "ConvNeXt" \
            else nn.Sequential((Conv(features = 96, kernel_size = (1,1), strides = (1,1)),
                                ConvNeXt_Block(features = 96))) if self.model_spatial == "ConvNeXt_Block" \
            else nn.Sequential((Conv(features = 96, kernel_size = (1,1), strides = (1,1)),
                                ConvNeXt_Block(features = 96),
                                ConvNeXt_Block(features = 96),
                                ConvNeXt_Block(features = 96))) if self.model_spatial == "3_ConvNeXt_Block" \
            else Identity()
        
        Temporal_NN = Dense(features = 32) if self.model_temporal == "Dense32" \
            else Dense(features = 64) if self.model_temporal == "Dense64" \
            else Identity()
        
        DDNN = DDNN_GEV(n_clusters = self.n_clusters) if self.model_distributional == "DDNN" \
            else SimpleBaseline(n_clusters = self.n_clusters)
        
        model_final = AlpTh_NN(
            n_clusters = self.n_clusters,
            Spatial_NN = Spatial_NN,
            Temporal_NN = Temporal_NN,
            DDNN = DDNN
        )
                    
        state_final = createTrainState(model_final,
                                    self.rnginit,
                                    self.learning_rate,
                                    1,
                                    len(self.features))
        
        state_final = state_final.replace(params = output_params)
        
        
        param_pred = model_final.apply(state_final.params, jnp.expand_dims(self.test_s[0],0), jnp.expand_dims(self.test_t[0],0))
        mu,sigma,xi = param_pred[0,0], param_pred[0,5], param_pred[0,10]
        ys = self.test_l[0][0]
        visualise_GEV(mu, sigma, xi, ys, os.path.join(self.plot_folder, "GEV.png"))
        
    
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
            'model_spatial': self.model_spatial,
            'model_temporal': self.model_temporal,
            'model_distributional': self.model_distributional,
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
        
        if hasattr(self, 'output_CRPS'):
            experiment_dict['output_CRPS'] = self.output_CRPS
        if hasattr(self, 'test_CRPS'):
            experiment_dict['test_CRPS'] = self.test_CRPS
        
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
            
        result += f"\nSpatial model: {self.model_spatial}"
        result += f"\nTemporal model: {self.model_temporal}"
        result += f"\nDistributional model: {self.model_distributional}"
        
        result += f"\nNumber of states kept to create output state: {self.n_best_states}"
        
        if hasattr(self, 'output_CRPS'):
            result += f"\n\n --- CRPS of saved model on validation set: {self.output_CRPS} ---"
        if hasattr(self, 'test_CRPS'):
            result += f"\n\n --- CRPS of saved model on test set: {self.test_CRPS} ---"
        
        return result
            
    
    def saveSummary(self):
        assert hasattr(self, 'output_CRPS'), "Run the experiment first."
        if os.path.exists(os.path.join(self.plot_dir, 'Summary.csv')):
            summary = pd.read_csv(os.path.join(self.plot_dir, 'Summary.csv'), index_col = 0)
            # Add line with experience number, output CRPS, and test CRPS
            if int(self.expnumber) in summary.index:
                summary.loc[int(self.expnumber)] = [self.output_CRPS, self.test_CRPS]
            else:
                summary = pd.concat((summary, pd.DataFrame({'Output CRPS': self.output_CRPS, 'Test CRPS': self.test_CRPS}, index = [int(self.expnumber)])))
            # Sort by experience number
            summary.sort_index(inplace = True)
            # Save to file
            summary.to_csv(os.path.join(self.plot_dir, 'Summary.csv'), index_label = "Experiment")
        else:
            summary = pd.DataFrame({'Output CRPS': [self.output_CRPS], 'Test CRPS': [self.test_CRPS]}, index = [int(self.expnumber)])
            summary.to_csv(os.path.join(self.plot_dir, 'Summary.csv'), index_label = "Experiment")
    
    
    @staticmethod
    def createExperimentFile(train_files, val_files, test_files,
                             train_label_files, val_label_files, test_label_files,
                             scratch_dir, plot_dir,
                             features, label,
                             batch_size,
                             model_spatial, model_temporal, model_distributional,
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
            'model_spatial': model_spatial,
            'model_temporal': model_temporal,
            'model_distributional': model_distributional,
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
    
