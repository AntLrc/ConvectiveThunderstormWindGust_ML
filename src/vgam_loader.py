import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

from multiprocessing import Pool

from scipy.special import gammainc, expi, gammaln

import os

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

class Experiment:
    """
    Class to load the data and preprocess it for the VGLM / VGAM model.
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
        
        self.cluster_file = experiment['cluster_file']
        self.storm_file = experiment['storm_file']
        self.storm_only = experiment['storm_only']

        self.Rscript = experiment['Rscript']
        self.Rsource = experiment['Rsource']
        self.model = experiment['model']
        
        self.expnumber = experiment_file.split('_')[-1].split('.')[0]
        
        self.stations = xr.open_dataset(self.train_label_files[0]).station.values
        
        self.clusters = pd.read_csv(self.cluster_file, header = None)
        self.n_clusters = len(self.clusters)
        self.clusters = [list(np.intersect1d(self.clusters.iloc[i,:].dropna(), self.stations)) for i in range(self.n_clusters)]
        self.n_stations = np.array(list(map(len, self.clusters))).sum()
        
        self.clusters_len = np.array(list(map(len, self.clusters)))
        coeffs_corr = np.concatenate(tuple(np.arange(1,self.clusters_len[i]+1) for i in range(len(self.clusters_len))))
        self.clusters_sep = np.cumsum(self.clusters_len) # Add on to work with coeffs_corr on separated clusters
        self.clusters_len = np.repeat(self.clusters_len, self.clusters_len)
        self.coeffs_corr = (2*coeffs_corr - self.clusters_len - 1)/(self.clusters_len**2)
        
        self.lead_times = None
        for files_set in [self.train_files, self.val_files, self.test_files]:
            for file in files_set:
                self.lead_times = np.intersect1d(xr.open_dataset(file).lead_time.values, self.lead_times) if self.lead_times is not None else xr.open_dataset(file).lead_time.values
        
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
        Creating  the inputs for the VGAM / VGLM model from netcdf files."""
        print("Creating inputs...", flush = True)
        sep_coeffs_corr = np.split(self.coeffs_corr, self.clusters_sep)
        for files_set, labels_set, type_set in zip([self.train_files, self.val_files, self.test_files], [self.train_label_files, self.val_label_files, self.test_label_files], ['train', 'val', 'test']):
            npinputs_s = []
            for i in range(self.n_clusters):
                npinputs_s.append(dict(zip(self.lead_times, [None]*len(self.lead_times))))
                
            nplabels = [None]*self.n_clusters
            npcorrs = np.zeros(self.n_clusters)
            for file, labels in zip(files_set, labels_set):
                inputs = xr.open_dataset(file, engine = "netcdf4")
                
                if self.storm_only:
                    loc_dates = pd.DatetimeIndex(np.intersect1d(inputs.time, self.dates))
                else:
                    loc_dates = inputs.time
                
                # Inputs
                tmp_var = []
                for i in range(self.n_clusters):
                    tmp_var.append(dict(zip(self.lead_times, [None]*len(self.lead_times))))
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
                    
                    # Selecting for each cluster, for each leadtime and converting to numpy array
                    for i in range(self.n_clusters):
                        for lt in self.lead_times:
                            tmp_cluster = tmp.sel(station = self.clusters[i], lead_time = lt).values
                            tmp_cluster = (tmp_cluster - self.mean[ivar])/self.std[ivar]
                            tmp_cluster = np.expand_dims(tmp_cluster, axis = -1)
                            tmp_var[i][lt] = tmp_cluster if tmp_var[i][lt] is None else np.concatenate([tmp_var[i][lt], tmp_cluster], axis = -1)
                
                for i in range(self.n_clusters):
                    for lt in self.lead_times:
                        npinputs_s[i][lt] = tmp_var[i][lt] if npinputs_s[i][lt] is None else np.concatenate([npinputs_s[i][lt], tmp_var[i][lt]], axis = 0)
                # Now npinputs_s should have a shape of (time, station, n_features)
                
                # Labels
                labels = xr.open_dataset(labels, engine = "netcdf4")
                labels = labels.sel(time = loc_dates)[self.label]
                for i in range(self.n_clusters):
                    labtmp = labels.sel(station = self.clusters[i]).values
                    nplabels[i] = labtmp if nplabels[i] is None else np.concatenate([nplabels[i], labtmp], axis = 0)
            for i in range(self.n_clusters):
                npcorrs[i] = (nplabels[i]@sep_coeffs_corr[i]).astype(np.float32).mean()                
                nplabels[i] = nplabels[i].astype(np.float32).reshape(nplabels[i].shape[0]*nplabels[i].shape[1])
                df = pd.DataFrame(nplabels[i], columns = [self.label])
                df.to_csv(os.path.join(self.scratch_folder, f"{i}_{type_set}_labels.csv"), index = False)
                for lt in self.lead_times:
                    npinputs_s[i][lt] = npinputs_s[i][lt].astype(np.float32).reshape(npinputs_s[i][lt].shape[0]*npinputs_s[i][lt].shape[1], npinputs_s[i][lt].shape[2])
                    df = pd.DataFrame(npinputs_s[i][lt], columns = self.features)
                    df.to_csv(os.path.join(self.scratch_folder, f"{i}_{lt}_{type_set}_features.csv"), index = False)
            pd.DataFrame(npcorrs).to_csv(os.path.join(self.scratch_folder, f"{type_set}_corrs.csv"), index = False)
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
    
    
    def run(self):
        """
        Run the experiment.
        """
        scratch_train_files_exist = all([os.path.exists(os.path.join(self.scratch_folder, f"{cluster}_{lead_time}_train_features.csv")) for cluster in range(self.n_clusters) for lead_time in self.lead_times])
        scratch_val_files_exist = all([os.path.exists(os.path.join(self.scratch_folder, f"{cluster}_{lead_time}_val_features.csv")) for cluster in range(self.n_clusters) for lead_time in self.lead_times])
        scratch_test_files_exist = all([os.path.exists(os.path.join(self.scratch_folder, f"{cluster}_{lead_time}_test_features.csv")) for cluster in range(self.n_clusters) for lead_time in self.lead_times])
        scratch_train_labels_exist = all([os.path.exists(os.path.join(self.scratch_folder, f"{cluster}_train_labels.csv")) for cluster in range(self.n_clusters)])
        scratch_val_labels_exist = all([os.path.exists(os.path.join(self.scratch_folder, f"{cluster}_val_labels.csv")) for cluster in range(self.n_clusters)])
        scratch_test_labels_exist = all([os.path.exists(os.path.join(self.scratch_folder, f"{cluster}_test_labels.csv")) for cluster in range(self.n_clusters)])
        
        files_exist = scratch_train_files_exist and scratch_val_files_exist and scratch_test_files_exist and scratch_train_labels_exist and scratch_val_labels_exist and scratch_test_labels_exist
        
        if not files_exist:
            self.load_mean_std()
            self.create_inputs()
        

        # Create a list of tuple containing the cluster and lead time
        args = [(cluster, lead_time) for cluster in range(self.n_clusters) for lead_time in self.lead_times]
        
        with Pool() as p:
            p.map(self.run_single, args)
        
        # Computing CRPS
        
        CRPS = np.zeros((self.n_clusters, len(self.lead_times)))
        self.corrs = pd.read_csv(os.path.join(self.scratch_folder, "test_corrs.csv")).values
        for cluster in range(self.n_clusters):
            for i, lead_time in enumerate(self.lead_times):
                preds = pd.read_csv(os.path.join(self.scratch_folder, f"{cluster}_{lead_time}_preds.csv"))
                labels = pd.read_csv(os.path.join(self.scratch_folder, f"{cluster}_test_labels.csv"))
                CRPS[cluster, i] = gevCRPS(preds.values[:,0], preds.values[:,1], preds.values[:,2], labels.values[:,0]).mean() - self.corrs[cluster]
        
        self.final_CRPS = CRPS.mean()
        
        self.saveInformation()
        self.saveExperimentFile()
        
        self.PIT(lead_time=[0,1,2,3,4,5,6,12,24,36,48,72])
        
        self.plotGEV()
        
        self.saveInformation()
        self.saveExperimentFile()
    
    def run_single(self, arg):
        cluster, lead_time = arg
        return os.system(f"Rscript {self.Rscript} --predictors {os.path.join(self.scratch_folder, f'{cluster}_{lead_time}_train_features.csv')} --response {os.path.join(self.scratch_folder, f'{cluster}_train_labels.csv')} --test-predictors {os.path.join(self.scratch_folder, f'{cluster}_{lead_time}_test_features.csv')} --output {os.path.join(self.scratch_folder, f'{cluster}_{lead_time}_preds.csv')} --model {self.model} --source {self.Rsource}")

    
    def PIT(self, lead_time = None, ensemble = 'test'):
        
        set_l = None
        param_pred = None
        
        if lead_time is None:
            set_l = pd.concat(
                [pd.read_csv(os.path.join(self.scratch_folder, f"{cluster}_{ensemble}_labels.csv")) for lt in self.lead_times for cluster in range(self.n_clusters)]
            )
            param_pred = pd.concat(
                [pd.read_csv(os.path.join(self.scratch_folder, f"{cluster}_{lt}_preds.csv")) for lt in self.lead_times for cluster in range(self.n_clusters)]
            )
            
            PIT = GEV(param_pred.values[:,0], param_pred.values[:,1], param_pred.values[:,2], set_l.values[:,0])
            sns.histplot(PIT, stat = 'density', bins = 50)
            plt.plot([0,1], [1,1], color = "black")
            plt.vlines(x=[0.,1.], ymin=0, ymax=1, color = 'black')
        else:
            if not isinstance(lead_time, list):
                lead_time = [lead_time]
            ncols = (len(lead_time) + 1)//2
            fig, axs = plt.subplots(2, ncols, figsize = (5*ncols, 10))
            plt.tight_layout()
            for i, lt in enumerate(lead_time):
                set_l = pd.concat(
                    [pd.read_csv(os.path.join(self.scratch_folder, f"{cluster}_{ensemble}_labels.csv")) for cluster in range(self.n_clusters)]
                )
                param_pred = pd.concat(
                    [pd.read_csv(os.path.join(self.scratch_folder, f"{cluster}_{lt}_preds.csv")) for cluster in range(self.n_clusters)]
                )
                PIT = GEV(param_pred.values[:,0], param_pred.values[:,1], param_pred.values[:,2], set_l.values[:,0])
                dataCRPS = gevCRPS(param_pred.values[:,0], param_pred.values[:,1], param_pred.values[:,2], set_l.values[:,0]).mean() - self.corrs.mean()
                sns.histplot(PIT, stat = 'density', bins = 50, ax = axs[i//ncols, i%ncols], label = f'CRPS: {dataCRPS:.3f}')
                axs[i//ncols, i%ncols].plot([0,1], [1,1], color = "black")
                axs[i//ncols, i%ncols].vlines(x=[0.,1.], ymin=0, ymax=1, color = 'black')
                axs[i//ncols, i%ncols].set_title(f'Rank histogram for lead time {lt} hours')
                axs[i//ncols, i%ncols].legend()
                
        plt.savefig(os.path.join(self.plot_folder, f'PIT_histogram.png'))
        plt.close()
    
    def plotGEV(self):
        mu, sigma, xi = pd.read_csv(os.path.join(self.scratch_folder, "0_0_preds.csv")).values[0]
        ys = pd.read_csv(os.path.join(self.scratch_folder, "0_test_labels.csv")).values[0]
        
        visualise_GEV(mu, sigma, xi, ys, os.path.join(self.plot_folder, "GEV.png"))
                
    def saveInformation(self):
        """
        Save the information of the experiment by printing it to a text file in self.plot_folder.
        """
        with open(os.path.join(self.plot_folder, 'Information.txt'), 'w') as f:
            f.write(str(self))
            
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
        
        result += f"\nModel: {self.model}\n"
        
        if hasattr(self, 'output_CRPS'):
            result += f"\n\n --- CRPS of saved model on validation set: {self.output_CRPS} ---"
        if hasattr(self, 'test_CRPS'):
            result += f"\n\n --- CRPS of saved model on test set: {self.test_CRPS} ---"
        
        return result
    
    
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
            'cluster_file': self.cluster_file,
            'storm_file': self.storm_file,
            'storm_only': self.storm_only,
            'Rscript': self.Rscript,
            'Rsource': self.Rsource,
            'model': self.model
        }
        
        if hasattr(self, 'output_CRPS'):
            experiment_dict['output_CRPS'] = self.output_CRPS
        
        if hasattr(self, 'test_CRPS'):
            experiment_dict['test_CRPS'] = self.test_CRPS
        
        with open(self.experiment_file, 'wb') as f:
            pickle.dump(experiment_dict, f)
        
        print(f"Experiment file saved at {self.experiment_file}")
        return
        
        
    @staticmethod
    def createExperimentFile(train_files, val_files, test_files,
                             train_label_files, val_label_files, test_label_files,
                             scratch_dir, plot_dir,
                             features, label,
                             cluster_file, storm_file, storm_only,
                             Rscript, Rsource, model,
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
            'cluster_file': cluster_file,
            'storm_file': storm_file,
            'storm_only': storm_only,
            'Rscript': Rscript,
            'Rsource': Rsource,
            'model': model,
        }
        
        # Add remaining arguments from kwargs
        for key, value in kwargs.items():
            experiment[key] = value
        
        with open(save_path, 'wb') as f:
            pickle.dump(experiment, f)
        
        print(f'Experiment file saved at {save_path}')
        
        return