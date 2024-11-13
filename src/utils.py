import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import jax.numpy as jnp
import jax
import xarray as xr
import geopandas as gpd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines

import os

from nn_losses import GEV, gevCRPSLoss, GEVpdf

from storm import Storm, Storms

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

def stormplot(stations, storms, gev_params, windgusts,
              temperature, pressure, wind,
              clusters, CRPSs, expCRPS,
              date, save_path,
              allCRPSs = None, cluster_nb = None):
    """
    Plot the storms on a map.
    """
    
    n_clusters = len(clusters)
    fig = plt.figure(figsize=(15,15))
    fig.suptitle(date, fontsize = 20)
    
    # Plot the map
    gs = GridSpec(2,2,top = 0.95, bottom = 0.55)    
    crsOrtho = ccrs.Orthographic(central_longitude = 8.3, central_latitude = 46.8)
    ax = fig.add_subplot(gs[0,0], projection = crsOrtho)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([5.8, 10.7, 45.7, 47.8])
    colors = ["tab:blue", "tab:brown", "tab:orange", "tab:pink", "tab:green", "tab:gray", "tab:olive", "tab:purple", "tab:cyan"]
    if not stations.crs == crsOrtho:
        stations = stations.to_crs(crsOrtho)
    stations['cluster'] = "0."
    for i in range(len(clusters)):
        for station in clusters[i]:
            if station in stations.index:
                stations.loc[station, 'cluster'] = f"Cluster {int(i + 1)}"
    for i in range(len(clusters)):
        stations.where(stations.cluster == f"Cluster {int(i + 1)}").plot(ax = ax, color = colors[i], legend = True, legend_kwds = {'label': f"Cluster {int(i + 1)}"})
    if date in storms.dates:
        geom = stations.loc[storms.dates[date]['stations']].geometry
        if not geom.crs == crsOrtho:
            geom = geom.to_crs(crsOrtho)
        geom.plot(ax = ax, color = "red", marker = 'x')
        stormsid = storms.dates[date]['storms']
        for stormid in stormsid:
            geom = storms.storms[stormid].track.iloc[(storms.storms[stormid].track.index + pd.Timedelta("55min")).floor("h") == date].geometry
            if not geom.crs == crsOrtho:
                geom = geom.to_crs(crsOrtho)
            chxstorm, chystorm = geom.x, geom.y
            ax.plot(chxstorm, chystorm, color = "red")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Stations and storms')
    colrs = colors[:len(clusters)]
    labels = [f"Cluster {int(i + 1)}" for i in range(len(clusters))]
    handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=label) for color, label in zip(colrs, labels)]
    ax.legend(handles = handles)
    
    ax = fig.add_subplot(gs[0,1], projection = crsOrtho)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([5.8, 10.7, 45.7, 47.8])
    ts = temperature.plot.contourf(ax = ax, cmap = 'coolwarm', transform = ccrs.PlateCarree(), cbar_kwargs = {'label': 'Temperature (K)', 'shrink':0.6})
    ax.clabel(ts, inline = True, colors = 'k')
    ax.set_title('Temperature')
    if date in storms.dates:
        stormsid = storms.dates[date]['storms']
        for stormid in stormsid:
            geom = storms.storms[stormid].track.iloc[(storms.storms[stormid].track.index + pd.Timedelta("55min")).floor("h") == date].geometry
            if not geom.crs == crsOrtho:
                geom = geom.to_crs(crsOrtho)
            chxstorm, chystorm = geom.x, geom.y
            ax.plot(chxstorm, chystorm, color = "red")
    
    ax = fig.add_subplot(gs[1,0], projection = crsOrtho)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([5.8, 10.7, 45.7, 47.8])
    ps = pressure.plot.contour(ax = ax, cmap = 'coolwarm', transform = ccrs.PlateCarree(), add_colorbar = True, cbar_kwargs = {'label': 'Pressure (Pa)', 'shrink':0.6})
    ax.clabel(ps, inline = True)
    ax.set_title('Pressure')
    if date in storms.dates:
        stormsid = storms.dates[date]['storms']
        for stormid in stormsid:
            geom = storms.storms[stormid].track.iloc[(storms.storms[stormid].track.index + pd.Timedelta("55min")).floor("h") == date].geometry
            if not geom.crs == crsOrtho:
                geom = geom.to_crs(crsOrtho)
            chxstorm, chystorm = geom.x, geom.y
            ax.plot(chxstorm, chystorm, color = "red")
    
    ax = fig.add_subplot(gs[1,1], projection = crsOrtho)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([5.8, 10.7, 45.7, 47.8])
    np.sqrt(wind.u10**2 + wind.v10**2).sel(lon = slice(5.5,11), lat = slice(48, 45.5)).plot(ax = ax, cmap = 'viridis', transform = ccrs.PlateCarree(), cbar_kwargs = {'label': 'Wind speed (m/s)', 'shrink':0.6})
    wind.plot.quiver(x = 'lon', y = 'lat', u = 'u10', v = 'v10', ax = ax, transform = ccrs.PlateCarree())
    ax.set_title('Wind')
    if date in storms.dates:
        stormsid = storms.dates[date]['storms']
        for stormid in stormsid:
            geom = storms.storms[stormid].track.iloc[(storms.storms[stormid].track.index + pd.Timedelta("55min")).floor("h") == date].geometry
            if not geom.crs == crsOrtho:
                geom = geom.to_crs(crsOrtho)
            chxstorm, chystorm = geom.x, geom.y
            ax.plot(chxstorm, chystorm, color = "red")
    
    gs = GridSpec(2,n_clusters,top = 0.5, bottom = 0.25)
    
    for icluster in range(n_clusters):
        axpdf = fig.add_subplot(gs[0,icluster], sharex = axpdf if icluster > 0 else None, sharey = axpdf if icluster > 0 else None)
        pdf = []
        cdf = []
        for ifold in range(len(gev_params)):
            if gev_params[ifold][icluster].shape == (3,):
                mu, sigma, xi = tuple(gev_params[ifold][icluster])
                x = np.linspace(0., 50., 300)
                pdf.append(GEVpdf(np.repeat(mu, 300), np.repeat(sigma, 300), np.repeat(xi, 300), x))
                cdf.append(GEV(np.repeat(mu, 300), np.repeat(sigma, 300), np.repeat(xi, 300), x))
            else:
                for istation in range(gev_params[ifold][icluster].shape[0]):
                    mu, sigma, xi = tuple(gev_params[ifold][icluster][istation])
                    x = np.linspace(0., 50., 300)
                    pdf.append(GEVpdf(np.repeat(mu, 300), np.repeat(sigma, 300), np.repeat(xi, 300), x))
                    cdf.append(GEV(np.repeat(mu, 300), np.repeat(sigma, 300), np.repeat(xi, 300), x))
        xs = np.repeat(np.expand_dims(x,0), len(pdf), axis = 0)
        pdf = pd.DataFrame(np.stack((xs, pdf)).reshape(2,-1).T, columns = ["Wind gust", "Density"], dtype = np.float32)
        cdf = pd.DataFrame(np.stack((xs, cdf)).reshape(2,-1).T, columns = ["Wind gust", "Probability"])
        sns.histplot(windgusts[icluster], stat = 'density', ax = axpdf, color = "lightgrey", label = 'Empirical')
        sns.lineplot(data = pdf, x = "Wind gust", y = "Density",
                     ax = axpdf, label = 'Theoretical distribution',
                     errorbar = lambda x: (x.min(), x.max()),
                     err_kws = {'alpha': 0.5})
        axpdf.set_xlabel(None)
        axpdf.set_ylabel(None)
        axpdf.set_title(f'Cluster {icluster+1}')
        axpdf.legend(fontsize = 'x-small')
        axpdf.grid()
        axcdf = fig.add_subplot(gs[1,icluster], sharex = axcdf if icluster > 0 else axpdf, sharey = axcdf if icluster > 0 else None)
        sns.ecdfplot(windgusts[icluster], ax = axcdf, stat = 'proportion', color = 'black', label = 'Empirical')
        sns.lineplot(data = cdf, x = "Wind gust", y = "Probability",
                     ax = axcdf, label = 'Theoretical CDF',
                     errorbar = lambda x: (x.min(), x.max()),
                     err_kws = {'alpha': 0.5})
        axcdf.set_xlabel(None)
        axcdf.set_ylabel(None)
        axcdf.legend(fontsize = 'x-small')
        axcdf.grid()
    
    gs = GridSpec(1,1,top = 0.2, bottom = 0.05)
    ax = fig.add_subplot(gs[0,0])
    if allCRPSs is None or cluster_nb is None:
        CRPSdf = pd.DataFrame(CRPSs, columns = [f"Cluster {i}" for i in range(1, n_clusters+1)])
        sns.boxplot(CRPSdf, ax=ax, palette = colors[:n_clusters], whis=(0, 100), fill=False)
        ax.scatter([f"Cluster {i}" for i in range(1, n_clusters+1)], [expCRPS]*n_clusters, color = 'black', label = 'Model CRPS')
        ax.set_ylabel('CRPS [m/s]')
        ax.legend()
        ax.grid()
    else:
        allCRPSsdf = None
        for lt in allCRPSs.keys():
            CRPSdf = pd.DataFrame(allCRPSs[lt], columns = [f"Cluster {i}" for i in range(1, n_clusters+1)])
            CRPSdf['Lead time'] = lt
            allCRPSsdf = CRPSdf if allCRPSsdf is None else pd.concat([allCRPSsdf, CRPSdf], ignore_index=True)
        sns.lineplot(data = allCRPSsdf, x = 'Lead time', y = f'Cluster {cluster_nb}', ax = ax, label = f'Cluster {cluster_nb}', color = colors[cluster_nb-1])
        CRPSdf = pd.DataFrame(CRPSs, columns = [f"Cluster {i}" for i in range(1, n_clusters+1)])
    
    fig.savefig(save_path)
    
    plt.close()
    
def plotTemporalVars(data, var, filtr, save_path, stations = None):
    """
    data: list of path to files containing the data.
    save_path: path where to dir where to save the plots.
    """
    
    if not isinstance(filtr, dict):
        if not stations is None and not isinstance(stations, list):
            stations = [stations]
        
        df = None
        for file in data:
            ds = xr.open_dataset(file)[var]
            x = ds.sel(time = pd.DatetimeIndex(np.intersect1d(ds.time, filtr))).to_dataframe().reset_index()
            x['month'] = x.time.dt.month
            x['hour'] = x.time.dt.hour
            x['Filtered'] = "Storm"
            x = x.drop(labels = ['time', 'latitude', 'longitude'], axis = 1)
            if not stations is None:
                x = x[x.station.isin(stations)]
            else:
                x = x.drop(labels = 'station', axis = 1)
            df = x if df is None else pd.concat([df, x], ignore_index=True)
            
            y = ds.to_dataframe().reset_index()
            y['month'] = y.time.dt.month
            y['hour'] = y.time.dt.hour
            y['Filtered'] = "Unfiltered"
            y = y.drop(labels = ['time', 'latitude', 'longitude'], axis = 1)
            if not stations is None:
                y = y[y.station.isin(stations)]
            else:
                y = y.drop(labels = 'station', axis = 1)
            df = pd.concat([df, y], ignore_index=True)
    else:
        # filtr is structured as a dict with time steps for keys and a set of stations for values
        df = None
        for file in data:
            ds = xr.open_dataset(file)[var]
            x = ds.sel(time = pd.DatetimeIndex(np.intersect1d(ds.time, pd.DatetimeIndex(filtr.keys())))).to_dataframe().reset_index()
            x['month'] = x.time.dt.month
            x['hour'] = x.time.dt.hour
            x['Filtered'] = "0"
            for t in filtr.keys():
                # Set the Filtered value to "Storm" for the rows for which time is t and station is in filtr[t]
                x.loc[(x.time == t) & (x.station.isin(filtr[t])), 'Filtered'] = "Storm"
            x = x.drop(labels = ['time', 'latitude', 'longitude'], axis = 1)
            x = x[x.Filtered != "0"]
            df = x if df is None else pd.concat([df, x], ignore_index=True)
            
            y = ds.to_dataframe().reset_index()
            y['month'] = y.time.dt.month
            y['hour'] = y.time.dt.hour
            y['Filtered'] = "Unfiltered"
            y = y.drop(labels = ['time', 'latitude', 'longitude'], axis = 1)
            df = pd.concat([df, y], ignore_index=True)
            
        
    # Plot the data
    # Relplot
    sns.set_theme()
    sns.relplot(
        data = df,
        x = 'hour',
        y = 'wind_speed_of_gust',
        hue = 'Filtered',
        col = 'month',
        row = 'station' if not stations is None else None,
        kind = 'line',
        errorbar = 'sd'
    )
    plt.savefig(os.path.join(save_path, '_'.join(stations if not stations is None else []) + 'RelPlotWindSpeed.png'))
    plt.close()
    
    # Catplot
    sns.set_theme()
    sns.catplot(
    data = df,
    kind = 'violin',
    x = 'hour',
    y = 'wind_speed_of_gust',
    row = 'station' if not stations is None else None,
    hue = 'Filtered',
    split = True,
    height = 5,
    aspect = 4
    )
    plt.savefig(os.path.join(save_path, '_'.join(stations if not stations is None else []) + 'CatPlotWindSpeed.png'))
    plt.close()

def Metricsevolution(modelCRPS, lead_times, save_path, ptype = 'lines', metricsname = 'Metrics', style = None, stylename = 'Data'):
    """
    Plot the evolution of the CRPS during the training for different models
    
    modelCRPS: dict with the model name as key and the CRPSs as values.
    """
    df = None
    for model in modelCRPS.keys():
        x = pd.DataFrame(modelCRPS[model], columns = lead_times)
        x['Fold'] = [f"Fold {i}" for i in range(len(modelCRPS[model]))]
        x = x.melt(id_vars = 'Fold', var_name = 'Lead time', value_name = metricsname)
        x['Model'] = model
        if style is not None:
            x[stylename] = style[model]
            x['Model'] = model.split(' ')[0]
        df = x if df is None else pd.concat([df, x], ignore_index=True)
    fig = plt.figure(figsize = (7, 4))
    if ptype == 'lines':
        sns.lineplot(data = df, x = 'Lead time', y = metricsname, hue = 'Model', style = 'Fold')
    else:
        sns.lineplot(data = df, x = 'Lead time', y = metricsname, hue = 'Model', style = stylename)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def Paramdistribution(param, save_path):
    """
    Plot the distribution of the parameters.
    
    param: dict with the model name as key and the parameters as values.
    """
    n_clusters = param.shape[1]//3
    df = pd.DataFrame(param,
                      columns = [f"mu_{i}" for i in range(n_clusters)] +\
                          [f"sigma_{i}" for i in range(n_clusters)] +\
                              [f"xi_{i}" for i in range(n_clusters)])
    df = df.melt(var_name = 'Cluster', value_name = 'Value')
    df['Parameter'] = df.Cluster.apply(lambda x: x.split('_')[0])
    df['Cluster'] = df.Cluster.apply(lambda x: x.split('_')[1])
    sns.displot(data = df, x = 'Value', col = 'Cluster', row = 'Parameter', kind = 'hist', stat = 'density')
    plt.savefig(save_path)
    plt.close()
    
