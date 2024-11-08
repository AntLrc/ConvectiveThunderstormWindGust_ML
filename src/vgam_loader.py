import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import geopandas as gpd

import pickle

from multiprocessing import Pool

from scipy.special import gammainc, expi, gammaln

import os

from storm import Storm, Storms

from utils import stormplot, Metricsevolution


def doubleexp(mu,sigma,y):
    return np.exp(-np.exp(-(y-mu)/sigma))

def lgamma(x):
    return gammaln(x)

def GEV(mu, sigma, xi, y):
    """
    Computes the Generalized Extreme Value CDF.
    """
    
    yred = (y - mu)/sigma
    xiNullMask = (xi == 0)
    xival = np.where(xiNullMask, 0.5, xi)
    
    y0 = np.logical_and(xi > 0, yred <= -1/xival)
    y1 = np.logical_and(xi < 0, yred >= -1/xival)
    
    yInBoundary = np.logical_or(
        np.logical_and(xi < 0, yred < -1/xival),
        np.logical_and(xi > 0, yred > -1/xival)
    )
    
    yredval = np.where(np.logical_or(y0,y1), (np.log(2)**(-xival) - 1)/xival, yred)
    
    return np.where(yInBoundary,
                    np.exp(-(1+xival*yredval)**(-1/xival)),
                    np.where(xiNullMask,
                              doubleexp(mu,sigma,y),
                              np.where(y1,
                                        1.,
                                        0.)))
    
def GEVpdf(mu, sigma, xi, y):
    yred = (y - mu)/sigma
    y0 = np.logical_and(xi > 0, yred <= -1/xi)
    y1 = np.logical_and(xi < 0, yred >= -1/xi)
    
    xival = np.where(xi == 0, 0.5, xi)
    yredval = np.where(np.logical_or(y0,y1), (np.log(2)**(-xival) - 1)/xival, yred)
    
    yInBoundary = np.logical_or(
        np.logical_and(xi < 0, yred < -1/xival),
        np.logical_and(xi > 0, yred > -1/xival)
    )
    
    yInBoundary = np.logical_or(yInBoundary, xi == 0)
    
    ty = np.where(xi ==0, np.exp(-yredval), (1+xival*yredval)**(-1/xi))
    
    return np.where(yInBoundary,
                     (1/sigma)*ty**(xi+1)*np.exp(-ty),
                     0.)                            
    
def gevCRPS(mu, sigma, xi, y):
    """
    Compute the closed form of the Continuous Ranked Probability Score (CRPS) for the Generalized Extreme Value distribution.
    Based on Friedrichs and Thorarinsdottir (2012).
    """
    
    yred = (y - mu)/sigma
    xiNullMask = (xi == 0)
    xival = np.where(xiNullMask, 0.5, xi)
    
    gevval = GEV(mu, sigma, xi, y)
    
    y0 = np.logical_and(xi > 0, yred <= -1/xival)
    y1 = np.logical_and(xi < 0, yred >= -1/xival)

    yInBoundary = np.logical_and(
        np.logical_not(xiNullMask),
        np.logical_not(np.logical_or(y1, y0))
    )
    
    yredval = np.where(np.logical_or(y0,y1), (np.log(2)**(-xival) - 1)/xival, yred)
    
    expyrednull = -np.exp(np.where(xiNullMask, -yred, 0.))
    
    return np.where(yInBoundary,
                     sigma*(-yredval - 1/xival)*(1- 2*gevval) - sigma/xival*np.exp(lgamma(1-xival)) * (2**xival - 2*gammainc(1-xival,(1+xival*yredval)**(-1/xival))),
                    np.where(xiNullMask,
                              mu - y + sigma*(np.euler_gamma - np.log(2)) - 2 * sigma * expi(expyrednull),
                              np.where(y1,
                                        sigma*(-yred - 1/xival)*(1- 2*gevval) - sigma/xival*np.exp(lgamma(1-xival)) * 2**xival,
                                        sigma*(-yred - 1/xival)*(1- 2*gevval) - sigma/xival*np.exp(lgamma(1-xival)) * (2**xival - 2))))

def visualise_GEV(mu, sigma, xi, ys, save_path):
    """
    Visualise the Generalized Extreme Value distribution.
    """
    fig, axs = plt.subplots(2,1, figsize = (6.4, 9.6))
    sns.set_theme()
    x = np.linspace(-10, 50, 300)
    ypdf = GEVpdf(np.repeat(mu, 300), np.repeat(sigma,300), np.repeat(xi,300), x)
    ycdf = GEV(np.repeat(mu, 300), np.repeat(sigma,300), np.repeat(xi,300), x)
    
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

class RExperiment:
    """
    Class to load the data and preprocess it for the VGLM / VGAM model.
    """

    def __init__(self, experiment_file):
        with open(experiment_file, 'rb') as f:
            experiment = pickle.load(f)
        self.expnumber = experiment_file.split('_')[-1].split('.')[0]
        self.files = experiment['files']
        default_files = {
            'storms': None,
            'clusters': None,
            'experiment': experiment_file,
            'train':{
                'inputs': [],
                'labels': None
            },
            'test':{
                'inputs': [],
                'labels': None
            },
            'R': {
                'script': None,
                'source': None,
                'predict':None
            }
        }
        for k,v in default_files.items():
            if not k in self.files.keys():
                    self.files[k] = v
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if not subk in self.files[k].keys():
                        self.files[k][subk] = subv
        self.folders = experiment['folders']
        default_folders = {
            'scratch':{
                'folder': None,
                'dir': None
            },
            'plot':{
                'folder': None,
                'dir': None,
                'model': None
            }
        }
        for k,v in default_folders.items():
            if not k in self.folders.keys():
                    self.folders[k] = v
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if not subk in self.folders[k].keys():
                        self.folders[k][subk] = subv
        self.features = experiment['features']
        self.label = experiment['label']
        self.vgam_kwargs = experiment.get('vgam_kwargs', {})
        default_vgam_kwargs = {
            'model': 'vglm',
            'spline_df': 3
        }
        for k,v in default_vgam_kwargs.items():
            if not k in self.vgam_kwargs.keys():
                self.vgam_kwargs[k] = v
        self.model_kwargs = experiment.get('model_kwargs',{})
        default_model_kwargs = {
            'data':'normal',
            'target': 'GEV',
            'time_encoding': 'sinusoidal',
            'n_folds': len(self.files['train']['inputs']),
            'seqfeatsel': False
        }
        for k,v in default_model_kwargs.items():
            if not k in self.model_kwargs.keys():
                self.model_kwargs[k] = v
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if not subk in self.model_kwargs[k].keys():
                        self.model_kwargs[k][subk] = subv
        self.target = 0 if self.model_kwargs['target'] == 'GEV' else 1 #encoding
        self.filter = experiment.get('filter',{})
        default_filter = {
            'lead_times': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,27,30,33,36,42,48,60,72],
            'storm_part': {'train': None,
                           'test': None},
        }
        for k,v in default_filter.items():
            if not k in self.filter.keys():
                    self.filter[k] = v
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if not subk in self.filter[k].keys():
                        self.filter[k][subk] = subv
        with open(self.files['storms'], 'rb') as f:
            self.filter['storms'] = pickle.load(f)
        self.filter['dates'] = pd.DatetimeIndex(self.filter['storms'].dates.keys()).unique()
        self.clusters = {
            'stations': experiment.get('stations',
                                        xr.open_dataset(self.files['train']['labels'][0]).station.values),
            'groups': pd.read_csv(self.files['clusters'], header = None)
        }
        self.clusters['n'] = len(self.clusters['groups'])
        self.clusters['groups'] = [list(np.intersect1d(self.clusters['groups'].iloc[i,:].dropna(), self.clusters['stations'])) for i in range(self.clusters['n'])]
        self.CRPS = experiment.get('CRPS', {})
        default_CRPS = {
            'mean': None,
            'std': None,
            'values': None
        }
        for k,v in default_CRPS.items():
            if not k in self.CRPS.keys():
                    self.CRPS[k] = v
        self.LogLik = experiment.get('LogLik', {})
        default_LogLik = {
            'mean': None,
            'std': None,
            'values': None
        }
        for k,v in default_LogLik.items():
            if not k in self.LogLik.keys():
                    self.LogLik[k] = v
        self.Data = experiment.get('Data', {})
        default_Data = {
            'mean': None,
            'std': None
        }
        for k,v in default_Data.items():
            if not k in self.Data.keys():
                    self.Data[k] = v
        os.makedirs(self.folders['scratch']['folder'], exist_ok = True)
        os.makedirs(self.folders['plot']['folder'], exist_ok = True)
        if self.folders['plot']['model'] is None:
            self.folders['plot']['model'] = os.path.join(self.folders['plot']['folder'], 'models')
        os.makedirs(self.folders['plot']['model'], exist_ok = True)
        self.plot = self.Plotter(self)
        self.save = self.Saver(self)
        self.diag = self.Diagnostics(self)

    def load_mean_std(self):
        """
        Preparation to feature normalisation through review of training set.
        """
        print("Computing mean and std of features...", end = "\n", flush = True)
        mean_ = np.zeros((len(self.features)))
        mean_sq = np.zeros((len(self.features)))
        count = np.zeros((len(self.features)))
        
        for file in self.files['train']['inputs']:
            print(file)
            inputs = xr.open_dataset(file, engine = "netcdf4").sel(lead_time = self.filter['lead_times'])
            loc_dates = pd.DatetimeIndex(inputs.time)
            if self.filter['storm_part']['train'] is not None:
                seed, ratio = self.filter['storm_part']['train']
                rng = np.random.default_rng(seed)
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter['dates'])
                nostorm_dates = input_dates.difference(self.filter['dates'])
                if len(storm_dates) >= len(input_dates)*ratio:
                    storm_dates = storm_dates[rng.permutation(len(storm_dates))[:int(len(input_dates)*ratio)]]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[rng.permutation(len(nostorm_dates))[:int(len(input_dates)*(1-ratio))]]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
            for ivar in range(len(self.features)):
                var = self.features[ivar]
                if len(var.split('_')) == 2:
                    var, var_level = var.split('_')
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(inputs.sel(isobaricInhPa = var_level)['u']**2 + inputs.sel(isobaricInhPa = var_level)['v']**2)
                    else:
                        tmp = inputs.sel(isobaricInhPa = var_level)[var]
                elif var == 'wind':
                    tmp = np.sqrt(inputs['u10']**2 + inputs['v10']**2)
                elif var == 'date':
                    tmp = inputs.time.dt.dayofyear.expand_dims({'station':inputs.station,'lead_time':inputs.lead_time}, axis = (-1,0))
                elif var == 'hour':
                    tmp = inputs.time.dt.hour.expand_dims({'station':inputs.station,'lead_time':inputs.lead_time}, axis = (-1,0))
                else:
                    tmp = inputs[var]
                tmp = tmp.sel(time = loc_dates)
                tmp = tmp.values
                count[ivar] += len(tmp)
                mean_[ivar] += tmp.mean()*len(tmp)
                mean_sq[ivar] += (tmp**2).mean()*len(tmp)
            inputs.close()
        self.Data['mean'] = mean_/count
        self.Data['std'] = np.sqrt(mean_sq/count - self.Data['mean']**2)
        self.save.experimentfile()
        print("Done.", flush = True)

    def create_inputs(self):
        """
        Creating  the inputs for the VGAM / VGLM model from netcdf files."""
        print("Creating inputs...", flush = True)
        if not self.model_kwargs['data'] in ['normal', 'mean', 'collective']:
            raise ValueError("Mode must be 'normal', 'mean' or 'collective'.")
        # Creating the folds
        for file, labels, fold in zip(self.files['train']['inputs'], self.files['train']['labels'], range(len(self.files['train']['inputs']))):
            #Each file will be use as a fold, ass it corresponds to a year
            inputs = xr.open_dataset(file, engine = "netcdf4").sel(lead_time = self.filter['lead_times'])
            loc_dates = pd.DatetimeIndex(inputs.time)
            if self.filter['storm_part']['train'] is not None:
                seed, ratio = self.filter['storm_part']['train']
                rng = np.random.default_rng(seed)
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter['dates'])
                nostorm_dates = input_dates.difference(self.filter['dates'])
                if (ratio != 1) and len(storm_dates) >= len(nostorm_dates)*ratio/(1-ratio):
                    storm_dates = storm_dates[rng.permutation(len(storm_dates))[:int(len(nostorm_dates)*ratio/(1-ratio))]]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[rng.permutation(len(nostorm_dates))[:int(len(storm_dates)*(1-ratio)/ratio)]]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
                storm_presence = np.zeros(len(loc_dates))
                storm_presence[loc_dates.isin(storm_dates)] = 1
                npstorms = [storm_presence]*self.clusters['n']
            # Inputs
            npinputs_s = [dict(zip(self.filter['lead_times'], [None]*len(self.filter['lead_times']))) for i in range(self.clusters['n'])]
            for ivar in range(len(self.features)):
                # First obtaining corresponding data array
                var = self.features[ivar]
                if len(var.split('_')) == 2:
                    var, var_level = var.split('_')
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(inputs.sel(isobaricInhPa = var_level)['u']**2 + inputs.sel(isobaricInhPa = var_level)['v']**2)
                    else:
                        tmp = inputs.sel(isobaricInhPa = var_level)[var]
                else:
                    if var == 'wind':
                        tmp = np.sqrt(inputs['u10']**2 + inputs['v10']**2)
                    elif var == 'date':
                        tmp = inputs.time.dt.dayofyear.expand_dims({'station':inputs.station,'lead_time':inputs.lead_time}, axis = (-1,0))
                    elif var == 'hour':
                        tmp = inputs.time.dt.hour.expand_dims({'station':inputs.station,'lead_time':inputs.lead_time}, axis = (-1,0))
                    else:
                        tmp = inputs[var]
                tmp = tmp.sel(time = loc_dates)
                # Selecting for each cluster, for each leadtime and converting to numpy array
                for icluster in range(self.clusters['n']):
                    for lt in self.filter['lead_times']:
                        tmp_cluster = tmp.sel(station = self.clusters['groups'][icluster], lead_time = lt).values
                        tmp_cluster = (tmp_cluster - self.Data['mean'][ivar])/self.Data['std'][ivar]
                        tmp_cluster = np.expand_dims(tmp_cluster, axis = -1)
                        npinputs_s[icluster][lt] = tmp_cluster if npinputs_s[icluster][lt] is None else np.concatenate([npinputs_s[icluster][lt], tmp_cluster], axis = -1)
            # Labels
            labels = xr.open_dataset(labels, engine = "netcdf4")
            labels = labels.sel(time = loc_dates)[self.label]
            nplabels = [None]*self.clusters['n']
            for icluster in range(self.clusters['n']):
                nplabels[icluster] = labels.sel(station = self.clusters['groups'][icluster]).values
                nplabels[icluster] = nplabels[icluster].astype(np.float32).reshape(nplabels[icluster].shape[0]*nplabels[icluster].shape[1])
                npstorms[icluster] = np.repeat(np.expand_dims(npstorms[icluster], axis = 1), len(self.clusters['groups'][icluster]), axis = 1).reshape(-1)
            for icluster in range(self.clusters['n']):
                dflabels = pd.DataFrame(nplabels[icluster], columns = [self.label])
                dflabels.to_csv(os.path.join(self.folders['scratch']['folder'], f"{icluster}_fold_{fold}_labels.csv"), index = False)
                dfstorms = pd.DataFrame(npstorms[icluster], columns = ['storm'])
                dfstorms.to_csv(os.path.join(self.folders['scratch']['folder'], f"{icluster}_fold_{fold}_storm.csv"), index = False)
                for lt in self.filter['lead_times']:
                    if self.model_kwargs['data'] == 'normal':
                        npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).reshape(-1, npinputs_s[icluster][lt].shape[2])
                        dfinputs = pd.DataFrame(npinputs_s[icluster][lt], columns = self.features)
                    elif self.model_kwargs['data'] == 'mean':
                        # For each cluster, for each feature, mean of the feature over the stations
                        npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).mean(axis = 1, keepdims = True)
                        npinputs_s[icluster][lt] = np.repeat(npinputs_s[icluster][lt], len(self.clusters['groups'][icluster]), axis = 1).reshape(-1, npinputs_s[icluster][lt].shape[2])
                        dfinputs = pd.DataFrame(npinputs_s[icluster][lt], columns = self.features)
                    elif self.model_kwargs['data'] == 'collective':
                        # For each time step, for each feature, concatenate the features of the stations
                        npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).transpose(1,2,0).reshape(len(self.features)*len(self.clusters['groups'][icluster]), -1).transpose(1,0)
                        npinputs_s[icluster][lt] = np.repeat(npinputs_s[icluster][lt], len(self.clusters['groups'][icluster]), axis = 0)
                        columns = [f"{feature}_{station}" for feature in self.features for station in self.clusters['groups'][icluster]]
                        dfinputs = pd.DataFrame(npinputs_s[icluster][lt], columns = columns)
                    dfinputs.to_csv(os.path.join(self.folders['scratch']['folder'], f"{icluster}_{lt}_fold_{fold}_features.csv"), index = False)
        # Creating the test files
        npinputs_s = [dict(zip(self.filter['lead_times'], [None]*len(self.filter['lead_times']))) for i in range(self.clusters['n'])]
        nplabels = [None]*self.clusters['n']
        for file, labels in zip(self.files['test']['inputs'], self.files['test']['labels']):
            inputs = xr.open_dataset(file, engine = "netcdf4").sel(lead_time = self.filter['lead_times'])
            loc_dates = pd.DatetimeIndex(inputs.time)
            if self.filter['storm_part']['test'] is not None:
                seed, ratio = self.filter['storm_part']['test']
                rng = np.random.default_rng(seed)
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter['dates'])
                nostorm_dates = input_dates.difference(self.filter['dates'])
                if (ratio != 1) and len(storm_dates) >= len(nostorm_dates)*ratio/(1-ratio):
                    storm_dates = storm_dates[rng.permutation(len(storm_dates))[:int(len(nostorm_dates)*ratio/(1-ratio))]]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[rng.permutation(len(nostorm_dates))[:int(len(storm_dates)*(1-ratio)/ratio)]]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
            # Inputs
            tmp_var = [dict(zip(self.filter['lead_times'], [None]*len(self.filter['lead_times']))) for i in range(self.clusters['n'])]
            for ivar, var in enumerate(self.features):
                if len(var.split('_')) == 2:
                    var, var_level = var.split('_')
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(inputs.sel(isobaricInhPa = var_level)['u']**2 + inputs.sel(isobaricInhPa = var_level)['v']**2)
                    else:
                        tmp = inputs.sel(isobaricInhPa = var_level)[var]
                elif var == 'wind':
                    tmp = np.sqrt(inputs['u10']**2 + inputs['v10']**2)
                elif var == 'date':
                    tmp = inputs.time.dt.dayofyear.expand_dims({'station':inputs.station,'lead_time':inputs.lead_time}, axis = (-1,0))
                elif var == 'hour':
                    tmp = inputs.time.dt.hour.expand_dims({'station':inputs.station,'lead_time':inputs.lead_time}, axis = (-1,0))
                else:
                    tmp = inputs[var]
                tmp = tmp.sel(time = loc_dates)
                for icluster in range(self.clusters['n']):
                    for lt in self.filter['lead_times']:
                        tmp_cluster = tmp.sel(station = self.clusters['groups'][icluster], lead_time = lt).values
                        tmp_cluster = (tmp_cluster - self.Data['mean'][ivar])/self.Data['std'][ivar]
                        tmp_cluster = np.expand_dims(tmp_cluster, axis = -1)
                        tmp_var[icluster][lt] = tmp_cluster if tmp_var[icluster][lt] is None else np.concatenate([tmp_var[icluster][lt], tmp_cluster], axis = -1)
            for icluster in range(self.clusters['n']):
                for lt in self.filter['lead_times']:
                    npinputs_s[icluster][lt] = tmp_var[icluster][lt] if npinputs_s[icluster][lt] is None else np.concatenate([npinputs_s[icluster][lt], tmp_var[icluster][lt]], axis = 0)
            # Labels
            labels = xr.open_dataset(labels, engine = "netcdf4")
            labels = labels.sel(time = loc_dates)[self.label]
            for icluster in range(self.clusters['n']):
                labtmp = labels.sel(station = self.clusters['groups'][icluster]).values
                nplabels[icluster] = labtmp if nplabels[icluster] is None else np.concatenate([nplabels[icluster], labtmp], axis = 0)
        for icluster in range(self.clusters['n']):
            nplabels[icluster] = nplabels[icluster].astype(np.float32).reshape(nplabels[icluster].shape[0]*nplabels[icluster].shape[1])
            df = pd.DataFrame(nplabels[icluster], columns = [self.label])
            df.to_csv(os.path.join(self.folders['scratch']['folder'], f"{icluster}_test_labels.csv"), index = False)
            for lt in self.filter['lead_times']:
                if self.model_kwargs['data'] == 'normal':
                    npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).reshape(-1, npinputs_s[icluster][lt].shape[2])
                    df = pd.DataFrame(npinputs_s[icluster][lt], columns = self.features)
                elif self.model_kwargs['data'] == 'mean':
                    # For each cluster, for each feature, mean of the feature over the stations
                    npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).mean(axis = 1, keepdims = True)
                    npinputs_s[icluster][lt] = np.repeat(npinputs_s[icluster][lt], len(self.clusters['groups'][icluster]), axis = 1).reshape(-1, npinputs_s[icluster][lt].shape[2])
                    df = pd.DataFrame(npinputs_s[icluster][lt], columns = self.features)
                elif self.model_kwargs['data'] == 'collective':
                    # For each time step, for each feature, concatenate the features of the stations
                    npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).transpose(1,2,0).reshape(len(self.features)*len(self.clusters['groups'][icluster]), -1).transpose(1,0)
                    npinputs_s[icluster][lt] = np.repeat(npinputs_s[icluster][lt], len(self.clusters['groups'][icluster]), axis = 0)
                    columns = [f"{feature}_{station}" for feature in self.features for station in self.clusters['groups'][icluster]]
                    df = pd.DataFrame(npinputs_s[icluster][lt], columns = columns)
                df.to_csv(os.path.join(self.folders['scratch']['folder'], f"{icluster}_{lt}_test_features.csv"), index = False)
        print("Done.", flush = True)

    def run(self):
        # Create the inputs if needed
        fold_features_created = all([os.path.exists(os.path.join(self.folders['scratch']['folder'], f"{icluster}_{lt}_fold_{fold}_features.csv")) for fold in range(self.model_kwargs['n_folds']) for lt in self.filter['lead_times'] for icluster in range(self.clusters['n'])])
        fold_labels_created = all([os.path.exists(os.path.join(self.folders['scratch']['folder'], f"{icluster}_fold_{fold}_labels.csv")) for fold in range(self.model_kwargs['n_folds']) for icluster in range(self.clusters['n'])])
        test_features_created = all([os.path.exists(os.path.join(self.folders['scratch']['folder'], f"{icluster}_{lt}_test_features.csv")) for lt in self.filter['lead_times'] for icluster in range(self.clusters['n'])])
        test_labels_created = all([os.path.exists(os.path.join(self.folders['scratch']['folder'], f"{icluster}_test_labels.csv")) for icluster in range(self.clusters['n'])])
        if not all([fold_features_created, fold_labels_created, test_features_created, test_labels_created]):
            self.load_mean_std()
            self.create_inputs()
        # Create a list of tuple containing the cluster and lead time
        args = [(cluster, lead_time, fold, self.model_kwargs['seqfeatsel']) for cluster in range(self.clusters['n']) for lead_time in self.filter['lead_times'] for fold in range(self.model_kwargs['n_folds'])]
        with Pool() as p:
            p.map(self.run_single, args)
        # Computing CRPS
        predsfiles = np.array([[[os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv') for fold in range(self.model_kwargs['n_folds'])] for lead_time in self.filter['lead_times']] for cluster in range(self.clusters['n'])])
        if self.model_kwargs['seqfeatsel']:
            labelsfiles = np.array([[[os.path.join(self.folders['scratch']['folder'], f"{cluster}_{lead_time}_{fold}_test_labels.csv") for fold in range(self.model_kwargs['n_folds'])] for lead_time in self.filter['lead_times']] for cluster in range(self.clusters['n'])])
        else:
            labelsfiles = np.array([[[os.path.join(self.folders['scratch']['folder'], f"{cluster}_test_labels.csv") for fold in range(self.model_kwargs['n_folds'])] for lead_time in self.filter['lead_times']] for cluster in range(self.clusters['n'])])
        def compute_CRPS(predfile, labelfile):
            preds = pd.read_csv(predfile)
            labels = pd.read_csv(labelfile)
            return gevCRPS(preds.values[:,0], preds.values[:,1], preds.values[:,2], labels.values[:,0]).mean()
        compute_CRPS = np.vectorize(compute_CRPS)
        CRPS = compute_CRPS(predsfiles, labelsfiles)
        def compute_LogLik(predfile, labelfile):
            preds = pd.read_csv(predfile)
            labels = pd.read_csv(labelfile)
            return - np.log(GEVpdf(preds.values[:,0], preds.values[:,1], preds.values[:,2], labels.values[:,0])).mean()
        compute_LogLik = np.vectorize(compute_LogLik)
        LogLik = compute_LogLik(predsfiles, labelsfiles)
        with open(os.path.join(self.folders['plot']['folder'], 'CRPS.pkl'), 'wb') as f:
            pickle.dump(CRPS, f)
        with open(os.path.join(self.folders['plot']['folder'], 'LogLik.pkl'), 'wb') as f:
            pickle.dump(LogLik, f)
        CRPS = CRPS.mean(axis = (0,1)) #mean over clusters and lead times
        LogLik = LogLik.mean(axis = (0,1)) #mean over clusters and lead times
        self.CRPS['mean'] = CRPS.mean()
        self.CRPS['std'] = CRPS.std()
        self.CRPS['values'] = CRPS
        self.LogLik['mean'] = LogLik.mean()
        self.LogLik['std'] = LogLik.std()
        self.LogLik['values'] = LogLik
        self.save.information()
        self.save.summary()
        self.save.experimentfile()

    def run_single(self, arg):
        cluster, lead_time, fold, seqfeatsel = arg
        # Check if already trained
        if os.path.exists(os.path.join(self.folders['plot']['model'], f'{cluster}_{lead_time}_{fold}.rds')):
            # Only compute the predictions
            res = os.system(f"Rscript {self.files['R']['predict']} \
                            --test-predictors {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_test_features.csv')} \
                                --output {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv')} \
                                    --model-file {os.path.join(self.folders['plot']['model'], f'{cluster}_{lead_time}_{fold}.rds')} \
                                        --source {self.files['R']['source']}")
        else:
            if not seqfeatsel:
                # Testing will be performed on test set
                if not os.path.exists(os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv')):
                    train_s = pd.concat([pd.read_csv(os.path.join(self.folders['scratch']['folder'], f"{cluster}_{lead_time}_fold_{f}_features.csv"))
                                        for f in range(self.model_kwargs['n_folds']) if f != fold])
                    train_l = pd.concat([pd.read_csv(os.path.join(self.folders['scratch']['folder'], f"{cluster}_fold_{f}_labels.csv"))
                                        for f in range(self.model_kwargs['n_folds']) if f != fold])
                    train_s.to_csv(os.path.join(self.folders['scratch']['folder'], f"TS{cluster}_{lead_time}_{fold}.csv"), index = False)
                    train_l.to_csv(os.path.join(self.folders['scratch']['folder'], f"TL{cluster}_{lead_time}_{fold}.csv"), index = False)
                    storms = pd.read_csv(os.path.join(self.folders['scratch']['folder'], f"{cluster}_fold_{fold}_storm.csv"))
                    res = os.system(f"Rscript {self.files['R']['script']} \
                        --predictors {os.path.join(self.folders['scratch']['folder'], f'TS{cluster}_{lead_time}_{fold}.csv')} \
                            --response {os.path.join(self.folders['scratch']['folder'], f'TL{cluster}_{lead_time}_{fold}.csv')} \
                                --test-predictors {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_test_features.csv')} \
                                    --output {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv')} \
                                        --model-file {os.path.join(self.folders['plot']['model'], f'{cluster}_{lead_time}_{fold}.rds')} \
                                            --model {self.vgam_kwargs['model']} --source {self.files['R']['source']}")
                    os.remove(os.path.join(self.folders['scratch']['folder'], f"TS{cluster}_{lead_time}_{fold}.csv"))
                    os.remove(os.path.join(self.folders['scratch']['folder'], f"TL{cluster}_{lead_time}_{fold}.csv"))
                    return res
                else:
                    return 0
            else:
                # Testing will be performed on remaining fold
                if not os.path.exists(os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv')):
                    train_s = pd.concat([pd.read_csv(os.path.join(self.folders['scratch']['folder'],
                                                                f"{cluster}_{lead_time}_fold_{f}_features.csv"))
                                        for f in range(self.model_kwargs['n_folds']) if f != fold])
                    train_l = pd.concat([pd.read_csv(os.path.join(self.folders['scratch']['folder'],
                                                                f"{cluster}_fold_{f}_labels.csv"))
                                        for f in range(self.model_kwargs['n_folds']) if f != fold])
                    train_s.to_csv(os.path.join(self.folders['scratch']['folder'],
                                                f"TS{cluster}_{lead_time}_{fold}.csv"),
                                index = False)
                    train_l.to_csv(os.path.join(self.folders['scratch']['folder'],
                                                f"TL{cluster}_{lead_time}_{fold}.csv"),
                                index = False)
                    storms = pd.read_csv(os.path.join(self.folders['scratch']['folder'],
                                                    f"{cluster}_fold_{fold}_storm.csv"))
                    test_s = pd.read_csv(os.path.join(self.folders['scratch']['folder'],
                                                    f'{cluster}_{lead_time}_fold_{fold}_features.csv'))
                    test_s = test_s[storms.values == 1]
                    test_s.to_csv(os.path.join(self.folders['scratch']['folder'],
                                            f"{cluster}_{lead_time}_{fold}_test_features.csv"),
                                index=False)
                    test_l = pd.read_csv(os.path.join(self.folders['scratch']['folder'],
                                                        f"{cluster}_fold_{fold}_labels.csv"))
                    test_l = test_l[storms.values == 1]
                    test_l.to_csv(os.path.join(self.folders['scratch']['folder'],
                                            f"{cluster}_{lead_time}_{fold}_test_labels.csv"),
                                index=False)
                    res = os.system(f"Rscript {self.files['R']['script']} \
                        --predictors {os.path.join(self.folders['scratch']['folder'], f'TS{cluster}_{lead_time}_{fold}.csv')} \
                            --response {os.path.join(self.folders['scratch']['folder'], f'TL{cluster}_{lead_time}_{fold}.csv')} \
                                --test-predictors {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_features.csv')} \
                                    --output {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv')} \
                                        --model-file {os.path.join(self.folders['plot']['model'], f'{cluster}_{lead_time}_{fold}.rds')} \
                                            --model {self.vgam_kwargs['model']} --source {self.files['R']['source']}")
                    os.remove(os.path.join(self.folders['scratch']['folder'], f"TS{cluster}_{lead_time}_{fold}.csv"))
                    os.remove(os.path.join(self.folders['scratch']['folder'], f"TL{cluster}_{lead_time}_{fold}.csv"))
                    return res
                else:
                    return 0

    def copy(self, **kwargs):
        """
        Copy the experiment with new parameters.
        """
        new_exp = RExperiment(self.files['experiment'])
        for key, value in kwargs.items():
            if not isinstance(value, dict):
                setattr(new_exp, key, value)
            else:
                for subkey, subvalue in value.items():
                    if not isinstance(subvalue, dict):
                        tmp = getattr(new_exp, key)
                        tmp[subkey] = subvalue
                        setattr(new_exp, key, tmp)
                    else:
                        for subsubkey, subsubvalue in subvalue.items():
                            tmp = getattr(new_exp, key)
                            tmp[subkey][subsubkey] = subsubvalue
                            setattr(new_exp, key, tmp)
        save_path = os.path.dirname(new_exp.files['experiment'])
        new_exp.expnumber = len(os.listdir(save_path))
        new_exp.folders['scratch']['folder'] = os.path.join(new_exp.folders['scratch']['dir'], f"Experiment_{new_exp.expnumber}")
        new_exp.folders['plot']['folder'] = os.path.join(new_exp.folders['plot']['dir'], f"Experiment_{new_exp.expnumber}")
        new_exp.folders['plot']['model'] = os.path.join(new_exp.folders['plot']['folder'], 'models')
        new_exp.files['experiment'] = "_".join(self.files['experiment'].split('_')[:-1]) + f'_{new_exp.expnumber}.pkl'
        return new_exp

    def __str__(self):
        result = f"R Experiment n°{self.expnumber}\n\n"
        result += "Training files:\n"
        for file in self.files['train']['inputs']:
            result += f"{file}\n"
            
        result += "\nTest files:\n"
        for file in self.files['test']['inputs']:
            result += f"{file}\n"
            
        result += "\nTraining labels:\n"
        for file in self.files['train']['labels']:
            result += f"{file}\n"
        
        result += "\nTest labels:\n"
        for file in self.files['test']['labels']:
            result += f"{file}\n"
        
        result += f"\nScratch directory: {self.folders['scratch']['folder']}\n"
        
        result += f"\nPlotting directory: {self.folders['plot']['folder']}\n"
        
        result += "\nFeatures: "
        for feature in self.features:
            result += f"{feature} "
        result += "\n"
        
        result += f"\nLabel: {self.label}\n"
        
        result += f"\nData: {self.model_kwargs['data']}\n"
        
        result += f"\nLead times: {self.filter['lead_times']}\n"
              
        result += f"\nPart of storm in dataset: {self.filter['storm_part']}\n"
        
        result += f"\nTarget: {self.model_kwargs['target']}"
               
        result += f"\nModel:\n{self.vgam_kwargs['model']}\n"
        
        result += f"\n\n--- CRPS: {self.CRPS['mean']} +/- {self.CRPS['std']} m/s ---"
        
        result += f"\n\n--- LogLik: {self.LogLik['mean']} +/- {self.LogLik['std']} ---"
        
        return result

    class Diagnostics:
        def __init__(self, experiment):
            self.experiment = experiment
        
        def predict(self, inputs, label,
                    features_filename, preds_filename, label_filename,
                    lead_times = None, clusters = None, folds = None):
            lead_times = self.experiment.filter['lead_times'][0] if lead_times is None else lead_times
            if not isinstance(lead_times, list):
                lead_times = [lead_times]
            clusters = list(range(self.experiment.clusters['n'])) if clusters is None else clusters
            if not isinstance(clusters, list):
                clusters = [clusters]
            folds = list(range(self.experiment.model_kwargs['n_folds'])) if folds is None else folds
            if not isinstance(folds, list):
                folds = [folds]
            # Creating the input file
            npinputs_s = [dict(zip(self.filter['lead_times'], [None]*len(self.filter['lead_times']))) for i in range(self.clusters['n'])]
            nplabels = [None]*self.clusters['n']
            for ivar, var in enumerate(self.features):
                if len(var.split('_')) == 2:
                    var, var_level = var.split('_')
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(inputs.sel(isobaricInhPa = var_level)['u']**2 + inputs.sel(isobaricInhPa = var_level)['v']**2)
                    else:
                        tmp = inputs.sel(isobaricInhPa = var_level)[var]
                elif var == 'wind':
                    tmp = np.sqrt(inputs['u10']**2 + inputs['v10']**2)
                elif var == 'date':
                    tmp = inputs.time.dt.dayofyear.expand_dims({'station':inputs.station,'lead_time':inputs.lead_time}, axis = (-1,0))
                elif var == 'hour':
                    tmp = inputs.time.dt.hour.expand_dims({'station':inputs.station,'lead_time':inputs.lead_time}, axis = (-1,0))
                else:
                    tmp = inputs[var]
                for icluster in range(self.clusters['n']):
                    for lt in self.filter['lead_times']:
                        tmp_cluster = tmp.sel(station = self.clusters['groups'][icluster], lead_time = lt).values
                        tmp_cluster = (tmp_cluster - self.Data['mean'][ivar])/self.Data['std'][ivar]
                        tmp_cluster = np.expand_dims(tmp_cluster, axis = -1)
                        npinputs_s[icluster][lt] = tmp_cluster if npinputs_s[icluster][lt] is None else np.concatenate([npinputs_s[icluster][lt], tmp_cluster], axis = -1)
            # Labels
            labels = xr.open_dataset(labels, engine = "netcdf4")
            labels = labels[self.label]
            for icluster in range(self.clusters['n']):
                labtmp = labels.sel(station = self.clusters['groups'][icluster]).values
                nplabels[icluster] = labtmp
            for icluster in range(self.clusters['n']):
                nplabels[icluster] = nplabels[icluster].astype(np.float32).reshape(nplabels[icluster].shape[0]*nplabels[icluster].shape[1])
                df = pd.DataFrame(nplabels[icluster], columns = [self.label])
                df.to_csv(label_filename + f"{icluster}.csv", index = False)
                for lt in self.filter['lead_times']:
                    if self.model_kwargs['data'] == 'normal':
                        npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).reshape(-1, npinputs_s[icluster][lt].shape[2])
                        df = pd.DataFrame(npinputs_s[icluster][lt], columns = self.features)
                    elif self.model_kwargs['data'] == 'mean':
                        # For each cluster, for each feature, mean of the feature over the stations
                        npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).mean(axis = 1, keepdims = True)
                        npinputs_s[icluster][lt] = np.repeat(npinputs_s[icluster][lt], len(self.clusters['groups'][icluster]), axis = 1).reshape(-1, npinputs_s[icluster][lt].shape[2])
                        df = pd.DataFrame(npinputs_s[icluster][lt], columns = self.features)
                    elif self.model_kwargs['data'] == 'collective':
                        # For each time step, for each feature, concatenate the features of the stations
                        npinputs_s[icluster][lt] = npinputs_s[icluster][lt].astype(np.float32).transpose(1,2,0).reshape(len(self.features)*len(self.clusters['groups'][icluster]), -1).transpose(1,0)
                        npinputs_s[icluster][lt] = np.repeat(npinputs_s[icluster][lt], len(self.clusters['groups'][icluster]), axis = 0)
                        columns = [f"{feature}_{station}" for feature in self.features for station in self.clusters['groups'][icluster]]
                        df = pd.DataFrame(npinputs_s[icluster][lt], columns = columns)
                    df.to_csv(features_filename + f"{icluster}_{lt}.csv", index = False)
            # Computing the predictions
            for icluster in clusters:
                for lt in lead_times:
                    for fold in folds:
                        os.system(f"Rscript {self.files['R']['predict']} \
                            --test-predictors {features_filename + f'{icluster}_{lt}.csv'} \
                                --output {preds_filename + f'{icluster}_{lt}_{fold}.csv'} \
                                    --model-file {os.path.join(self.folders['plot']['model'], f'{icluster}_{lt}_{fold}.rds')} \
                                        --source {self.files['R']['source']}")
                        
        def compute_CRPS(self, preds_filename, label_filename,
                            lead_times = None, clusters = None, folds = None):
                lead_times = self.experiment.filter['lead_times'][0] if lead_times is None else lead_times
                if not isinstance(lead_times, list):
                    lead_times = [lead_times]
                clusters = list(range(self.experiment.clusters['n'])) if clusters is None else clusters
                if not isinstance(clusters, list):
                    clusters = [clusters]
                folds = list(range(self.experiment.model_kwargs['n_folds'])) if folds is None else folds
                if not isinstance(folds, list):
                    folds = [folds]
                predsfiles = np.array([[[preds_filename + f'{cluster}_{lead_time}_{fold}.csv' for fold in folds] for lead_time in lead_times] for cluster in clusters])
                labelsfiles = np.array([[label_filename + f"{cluster}.csv" for cluster in clusters] for lead_time in lead_times])
                def compute_CRPS(predfile, labelfile):
                    preds = pd.read_csv(predfile)
                    labels = pd.read_csv(labelfile)
                    return gevCRPS(preds.values[:,0], preds.values[:,1], preds.values[:,2], labels.values[:,0]).mean()
                compute_CRPS = np.vectorize(compute_CRPS)
                CRPS = compute_CRPS(predsfiles, labelsfiles)
                return CRPS
        
        def allCRPS(self, save = False):
            # Compute the CRPS for all clusters, time steps and folds
            # First recreate the inputs (which won't be used to compute CRPS as predicitons were already made when the model was run)
            # Same calculations as cnn_loader.Experiment.create_inputs
            # Creating the test files
            if not os.path.exists(os.path.join(self.experiment.folders['scratch']['folder'], "test_set.pkl")):
                npinputs_t = None
                npinputs_s = None
                nplabels = [None]*self.experiment.clusters['n']
                for file, labels in zip(self.experiment.files['test']['inputs'], self.experiment.files['test']['labels']):
                    inputs = xr.open_dataset(file, engine = "netcdf4").sel(lead_time = self.experiment.filter['lead_times'])
                    loc_dates = pd.DatetimeIndex(inputs.time)
                    if self.experiment.filter['storm_part']['test'] is not None:
                        seed, ratio = self.experiment.filter['storm_part']['test']
                        rng = np.random.default_rng(seed)
                        input_dates = pd.DatetimeIndex(inputs.time)
                        storm_dates = input_dates.intersection(self.experiment.filter['dates'])
                        nostorm_dates = input_dates.difference(self.experiment.filter['dates'])
                        if (ratio != 1) and len(storm_dates) >= len(nostorm_dates)*ratio/(1-ratio):
                            storm_dates = storm_dates[rng.permutation(len(storm_dates))[:int(len(nostorm_dates)*ratio/(1-ratio))]]
                            loc_dates = storm_dates.union(nostorm_dates)
                        else:
                            nostorm_dates = nostorm_dates[rng.permutation(len(nostorm_dates))[:int(len(storm_dates)*(1-ratio)/ratio)]]
                            loc_dates = storm_dates.union(nostorm_dates)
                        loc_dates = loc_dates.sort_values()
                    # Encoding of time
                    if self.experiment.model_kwargs['time_encoding'] == 'sinusoidal':
                        tmp_t = np.array([[[(time.day_of_year - 242)/107, # Encoding of day of year between -1 and +1
                                            np.cos(time.hour*np.pi/12),
                                            np.sin(time.hour*np.pi/12),
                                            lt/72] for time in loc_dates] for lt in inputs.lead_time],
                                        dtype = np.float32).reshape(-1,4)
                    elif self.experiment.model_kwargs['time_encoding'] == 'rbf':
                        pass
                    npinputs_t = tmp_t if npinputs_t is None else np.concatenate([npinputs_t, tmp_t], axis = 0)
                    # Inputs
                    tmp_var = None
                    for ivar in range(len(self.experiment.features)):
                        # First obtaining corresponding data array
                        var = self.experiment.features[ivar]
                        if len(var.split('_')) == 2:
                            var, var_level = var.split('_')
                            var_level = int(var_level[:-3])
                            if var == "wind":
                                tmp = np.sqrt(inputs.sel(isobaricInhPa = var_level)['u']**2 + inputs.sel(isobaricInhPa = var_level)['v']**2)
                            else:
                                tmp = inputs.sel(isobaricInhPa = var_level)[var]
                        elif var == 'wind':
                            tmp = np.sqrt(inputs['u10']**2 + inputs['v10']**2)
                        elif var == 'CAPE':
                            tmp = inputs['CAPE'].fillna(0.)
                        else:
                            tmp = inputs[var]
                        tmp = tmp.sel(time = loc_dates)
                        tmp = tmp.values
                        sh = tmp.shape
                        tmp = tmp.reshape((sh[0]*sh[1], sh[2]))
                        tmp = (tmp - self.experiment.Data['mean'][ivar])/self.experiment.Data['std'][ivar]
                        tmp = np.expand_dims(tmp, axis = -1)
                        tmp_var = tmp if tmp_var is None else np.concatenate([tmp_var, tmp], axis = -1)
                    npinputs_s = tmp_var if npinputs_s is None else np.concatenate([npinputs_s, tmp_var], axis = 0)
                    # Labels
                    labels = xr.open_dataset(labels, engine = "netcdf4")
                    labels = labels.sel(time = loc_dates)[self.experiment.label]
                    for i in range(self.experiment.clusters['n']):
                        labtmp = labels.sel(station = self.experiment.clusters['groups'][i]).values
                        labtmp = np.repeat(np.expand_dims(labtmp, axis = 0), len(inputs.lead_time), axis = 0)
                        labtmp = labtmp.reshape(labtmp.shape[0]*labtmp.shape[1], labtmp.shape[2])
                        nplabels[i] = labtmp if nplabels[i] is None else np.concatenate([nplabels[i], labtmp], axis = 0)
                with open(os.path.join(self.experiment.folders['scratch']['folder'], "test_set.pkl"), 'wb') as f:
                    pickle.dump((npinputs_s, npinputs_t, nplabels), f)
            # Get the test set
            with open(os.path.join(self.experiment.folders['scratch']['folder'], f"test_set.pkl"), 'rb') as f:
                s, t, l = pickle.load(f)
            CRPS = np.zeros((self.experiment.model_kwargs['n_folds'],
                             self.experiment.clusters['n'],
                             t.shape[0]))
            for ifold, fold in enumerate(range(self.experiment.model_kwargs['n_folds'])):
                for icluster, cluster in enumerate(range(self.experiment.clusters['n'])):
                    for ilt, lt in enumerate(self.experiment.filter['lead_times']):
                        preds = pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"{cluster}_{lt}_{fold}_test_preds.csv"))
                        labels = pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"{cluster}_test_labels.csv"))
                        crps = gevCRPS(preds.values[:,0], preds.values[:,1], preds.values[:,2], labels.values[:,0])
                        # Reshaping crps to account for the different stations
                        crps = crps.reshape(-1, len(self.experiment.clusters['groups'][cluster])).mean(axis = 1)
                        CRPS[ifold,icluster,t[:,3] == lt/72] = crps
            if save:
                with open(os.path.join(self.experiment.folders['plot']['folder'], 'allCRPSs.pkl'), 'wb') as f:
                    pickle.dump(CRPS,f)
            return s,t,l,CRPS
                        
    # Plotting
    class Plotter:
        def __init__(self, experiment):
            self.experiment = experiment
        
        def casestudy(self, date, lead_time, weathermapfile = None, alllts = False, cluster_nb = None, precomputed = False):
            strdate = date.strftime('%Y%m%d%H%M')
            stations = xr.open_dataset(self.experiment.files['test']['labels'][0]).isel(time = 0).to_dataframe()[["latitude", "longitude"]]
            stations = gpd.GeoDataFrame(stations, geometry = gpd.points_from_xy(stations.longitude, stations.latitude), crs = "EPSG:4326")
            file = None
            for f in self.experiment.files['test']['inputs']:
                if date in xr.open_dataset(f).time.values:
                    file = f
                    break
            if file is None:
                raise ValueError(f"No test file found for date {date}")
            l_file = None
            for f in self.experiment.files['test']['labels']:
                if date in xr.open_dataset(f).time.values:
                    l_file = f
                    break
            if l_file is None:
                raise ValueError(f"No test label file found for date {date}")
            #Create inputs
            ds0 = xr.open_dataset(file, engine = "netcdf4").sel(time = date)
            ds2d = xr.open_dataset(weathermapfile, engine = "netcdf4").sel(time = date, lead_time = lead_time)
            temperature = ds2d.t2m
            pressure = ds2d.msl
            wind = ds2d[['u10', 'v10']]
            npinputs_s = [None]*self.experiment.clusters['n']
            npinputs_l = [None]*self.experiment.clusters['n']
            if self.experiment.Data['mean'] is None or self.experiment.Data['std'] is None:
                self.experiment.load_mean_std()
            if not precomputed:
                if not alllts:
                    ds = ds0.sel(lead_time = lead_time)
                    for ivar in range(len(self.experiment.features)):
                        # First obtaining corresponding data array
                        var = self.experiment.features[ivar]
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
                        # Selecting for each cluster, for each leadtime and converting to numpy array
                        for icluster in range(self.experiment.clusters['n']):
                            tmp_cluster = tmp.sel(station = self.experiment.clusters['groups'][icluster]).values
                            tmp_cluster = (tmp_cluster - self.experiment.Data['mean'][ivar])/self.experiment.Data['std'][ivar]
                            tmp_cluster = np.expand_dims(tmp_cluster, axis = len(tmp_cluster.shape))
                            npinputs_s[icluster] = tmp_cluster if npinputs_s[icluster] is None else np.concatenate([npinputs_s[icluster], tmp_cluster], axis = -1)
                    for icluster in range(self.experiment.clusters['n']):
                        if self.experiment.model_kwargs['data'] == 'normal':
                            npinputs_s[icluster] = npinputs_s[icluster].astype(np.float32).reshape(-1, npinputs_s[icluster].shape[-1])
                            df = pd.DataFrame(npinputs_s[icluster], columns = self.experiment.features)
                        elif self.experiment.model_kwargs['data'] == 'mean':
                            # For each cluster, for each feature, mean of the feature over the stations
                            npinputs_s[icluster] = npinputs_s[icluster].astype(np.float32).mean(axis = 0, keepdims = True)
                            npinputs_s[icluster] = np.repeat(npinputs_s[icluster], len(self.experiment.clusters['groups'][icluster]), axis = 0)
                            df = pd.DataFrame(npinputs_s[icluster], columns = self.experiment.features)
                        df.to_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{strdate}_{lead_time}.csv"), index = False)
                    for fold in range(self.experiment.model_kwargs['n_folds']):
                        for icluster in range(self.experiment.clusters['n']):
                            os.system(f"Rscript {self.experiment.files['R']['predict']} --test-predictors {os.path.join(self.experiment.folders['scratch']['folder'], f'CASESTUDY_{icluster}_{strdate}_{lead_time}.csv')} --output {os.path.join(self.experiment.folders['scratch']['folder'], f'CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv')} --model-file {os.path.join(self.experiment.folders['plot']['model'], f'{icluster}_{lead_time}_{fold}.rds')} --source {self.experiment.files['R']['source']}")
                else:
                    for lt in self.experiment.filter['lead_times']:
                        ds = ds0.sel(lead_time = lt)
                        npinputs_s = [None]*self.experiment.clusters['n']
                        npinputs_l = [None]*self.experiment.clusters['n']
                        for ivar in range(len(self.experiment.features)):
                            # First obtaining corresponding data array
                            var = self.experiment.features[ivar]
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
                            # Selecting for each cluster, for each leadtime and converting to numpy array
                            for icluster in range(self.experiment.clusters['n']):
                                tmp_cluster = tmp.sel(station = self.experiment.clusters['groups'][icluster]).values
                                tmp_cluster = (tmp_cluster - self.experiment.Data['mean'][ivar])/self.experiment.Data['std'][ivar]
                                tmp_cluster = np.expand_dims(tmp_cluster, axis = len(tmp_cluster.shape))
                                npinputs_s[icluster] = tmp_cluster if npinputs_s[icluster] is None else np.concatenate([npinputs_s[icluster], tmp_cluster], axis = -1)
                        for icluster in range(self.experiment.clusters['n']):
                            if self.experiment.model_kwargs['data'] == 'normal':
                                npinputs_s[icluster] = npinputs_s[icluster].astype(np.float32).reshape(-1, npinputs_s[icluster].shape[-1])
                                df = pd.DataFrame(npinputs_s[icluster], columns = self.experiment.features)
                            elif self.experiment.model_kwargs['data'] == 'mean':
                                # For each cluster, for each feature, mean of the feature over the stations
                                npinputs_s[icluster] = npinputs_s[icluster].astype(np.float32).mean(axis = 0, keepdims = True)
                                npinputs_s[icluster] = np.repeat(npinputs_s[icluster], len(self.experiment.clusters['groups'][icluster]), axis = 0)
                                df = pd.DataFrame(npinputs_s[icluster], columns = self.experiment.features)
                            df.to_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{strdate}_{lt}.csv"), index = False)
                        for fold in range(self.experiment.model_kwargs['n_folds']):
                            for icluster in range(self.experiment.clusters['n']):
                                os.system(f"Rscript {self.experiment.files['R']['predict']} --test-predictors {os.path.join(self.experiment.folders['scratch']['folder'], f'CASESTUDY_{icluster}_{strdate}_{lt}.csv')} --output {os.path.join(self.experiment.folders['scratch']['folder'], f'CASESTUDY_{icluster}_{fold}_{strdate}_{lt}_preds.csv')} --model-file {os.path.join(self.experiment.folders['plot']['model'], f'{icluster}_{lt}_{fold}.rds')} --source {self.experiment.files['R']['source']}")
            # Labels
            labels = xr.open_dataset(l_file, engine = "netcdf4")
            labels = labels.sel(time = date)[self.experiment.label]
            for icluster in range(self.experiment.clusters['n']):
                labtmp = labels.sel(station = self.experiment.clusters['groups'][icluster]).values
                npinputs_l[icluster] = labtmp if npinputs_l[icluster] is None else np.concatenate([npinputs_l[icluster], labtmp], axis = 0)
            for icluster in range(self.experiment.clusters['n']):
                npinputs_l[icluster] = npinputs_l[icluster].astype(np.float32).reshape(-1)

            
            #Build gev_params and mus, sigmas, xis          
            mus = [[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv")).values[:,0] for icluster in range(self.experiment.clusters['n'])] for fold in range(self.experiment.model_kwargs['n_folds'])]
            sigmas = [[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv")).values[:,1] for icluster in range(self.experiment.clusters['n'])] for fold in range(self.experiment.model_kwargs['n_folds'])]
            xis = [[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv")).values[:,2] for icluster in range(self.experiment.clusters['n'])] for fold in range(self.experiment.model_kwargs['n_folds'])]
            
            gev_params = [
                [
                    pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv")).values for icluster in range(self.experiment.clusters['n'])
                ] for fold in range(self.experiment.model_kwargs['n_folds'])
            ]
            
            # Compute CRPS for each cluster
            CRPSs = np.array([[
                gevCRPS(mus[ifold][icluster],
                        sigmas[ifold][icluster],
                        xis[ifold][icluster],
                        npinputs_l[icluster]).mean()
                for icluster in range(self.experiment.clusters['n'])]
            for ifold in range(self.experiment.model_kwargs['n_folds'])
            ])
            
            if alllts:
                allmus = [[[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{fold}_{strdate}_{lt}_preds.csv")).values[:,0] for icluster in range(self.experiment.clusters['n'])] for fold in range(self.experiment.model_kwargs['n_folds'])] for lt in self.experiment.filter['lead_times']]
                allsigmas = [[[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{fold}_{strdate}_{lt}_preds.csv")).values[:,1] for icluster in range(self.experiment.clusters['n'])] for fold in range(self.experiment.model_kwargs['n_folds'])] for lt in self.experiment.filter['lead_times']]
                allxis = [[[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{icluster}_{fold}_{strdate}_{lt}_preds.csv")).values[:,2] for icluster in range(self.experiment.clusters['n'])] for fold in range(self.experiment.model_kwargs['n_folds'])] for lt in self.experiment.filter['lead_times']]
                allCRPSs = {lt:[
                    [gevCRPS(allmus[ilt][ifold][icluster],
                            allsigmas[ilt][ifold][icluster],
                            allxis[ilt][ifold][icluster],
                            npinputs_l[icluster]).mean()
                    for icluster in range(self.experiment.clusters['n'])]
                for ifold in range(self.experiment.model_kwargs['n_folds'])
                ] for ilt, lt in enumerate(self.experiment.filter['lead_times'])
                }
            stormplot(stations, self.experiment.filter['storms'], gev_params, npinputs_l,
                temperature, pressure, wind,
                self.experiment.clusters['groups'], CRPSs, self.experiment.CRPS['mean'],
                date, os.path.join(self.experiment.folders['plot']['folder'], f'Stormplot_{date}_{lead_time}.png'),
                allCRPSs = allCRPSs, cluster_nb = cluster_nb)
            return allCRPSs

        def ltCRPS(self, keep = False):
            """
            Plot the CRPS for each lead time.
            """
            with open(os.path.join(self.experiment.folders['plot']['folder'], 'CRPS.pkl'), 'rb') as f:
                CRPS = pickle.load(f)
            CRPS = CRPS.mean(axis = 0).T
            Metricsevolution({"CRPS":CRPS}, self.experiment.filter['lead_times'], os.path.join(self.experiment.folders['plot']['folder'], 'CRPS_leadtime.png'))
            with open(os.path.join(self.experiment.folders['plot']['folder'], 'ltCRPS.pkl'), 'wb') as f:
                pickle.dump(CRPS, f)

        def parameters(self):
            ...

        def clusterCRPS(self, cluster, date):
            strdate = date.strftime('%Y%m%d%H%M')
            # Labels
            labels = xr.open_dataset(self.experiment.files['test']['labels'][0], engine = "netcdf4")
            labels = labels.sel(time = date)[self.experiment.label]
            npinputs_l = [None]*self.experiment.clusters['n']
            for icluster in range(self.experiment.clusters['n']):
                labtmp = labels.sel(station = self.experiment.clusters['groups'][icluster]).values
                npinputs_l[icluster] = labtmp if npinputs_l[icluster] is None else np.concatenate([npinputs_l[icluster], labtmp], axis = 0)
            for icluster in range(self.experiment.clusters['n']):
                npinputs_l[icluster] = npinputs_l[icluster].astype(np.float32).reshape(-1)
            mus = [[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{cluster}_{fold}_{strdate}_{lt}_preds.csv")).values[:,0] for fold in range(self.experiment.model_kwargs['n_folds'])] for lt in self.experiment.filter['lead_times']]
            sigmas = [[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{cluster}_{fold}_{strdate}_{lt}_preds.csv")).values[:,1] for fold in range(self.experiment.model_kwargs['n_folds'])] for lt in self.experiment.filter['lead_times']]
            xis = [[pd.read_csv(os.path.join(self.experiment.folders['scratch']['folder'], f"CASESTUDY_{cluster}_{fold}_{strdate}_{lt}_preds.csv")).values[:,2] for fold in range(self.experiment.model_kwargs['n_folds'])] for lt in self.experiment.filter['lead_times']]
            allCRPSs = np.array([[gevCRPS(mus[ilt][ifold][cluster],
                        sigmas[ilt][ifold][cluster],
                        xis[ilt][ifold][cluster],
                        npinputs_l[cluster]).mean()
                                    for ifold in range(self.experiment.model_kwargs['n_folds'])
                                ] for ilt, lt in enumerate(self.experiment.filter['lead_times'])]).T
            print(allCRPSs)
            Metricsevolution({"CRPS":allCRPSs}, self.experiment.filter['lead_times'], os.path.join(self.experiment.folders['plot']['folder'], f'CRPS_{cluster}_{date}.png'))
            return allCRPSs
        
    class Saver:
        def __init__(self, experiment):
            self.experiment = experiment

        def information(self):
            with open(os.path.join(self.experiment.folders['plot']['folder'], 'Information.txt'), 'w') as f:
                f.write(str(self.experiment))

        def summary(self):
            pass

        def experimentfile(self):
            experiment_dict = {
                'files': self.experiment.files,
                'folders': self.experiment.folders,
                'features': self.experiment.features,
                'label': self.experiment.label,
                'vgam_kwargs': self.experiment.vgam_kwargs,
                'model_kwargs': self.experiment.model_kwargs,
                'filter': {k:self.experiment.filter[k] for k in ['lead_times', 'storm_part']},
                'CRPS': self.experiment.CRPS,
                'LogLik': self.experiment.LogLik,
                'Data': self.experiment.Data
            }
            with open(self.experiment.files['experiment'], 'wb') as f:
                pickle.dump(experiment_dict, f)