# to be run with AlpthShape env from /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/src

import xarray as xr
import numpy as np
import pickle
import pandas as pd

from nn_loader import Experiment
from baselines import compute_ecdf_crps

# Load the data
exp = Experiment('/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/test/experiments/experiment_67.txt')

icon_ds = xr.open_dataset("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/treated_data/icon-ch2-eps-wind-speed-of-gust-2021-smn-v1.nc")
obs_da = xr.open_dataset("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/treated_data/NN_preinputs/new/2021_labels.nc").wind_speed_of_gust
icon_da = icon_ds.sel(station = obs_da.station.values).wind_speed_of_gust

with open('/scratch/alecler1/downscaling/JAX_NN/Experiment_54/test_set.pkl', 'rb') as f:
    _,t_array,_ = pickle.load(f)
    
lead_times = np.array(exp.filter['lead_times'])
clusters = exp.clusters['groups']
stations = icon_da.station.values
i_clusters = [
    [np.argwhere(stations == s)[0][0] for s in cluster] for cluster in clusters
]

with open('/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/NN/Experiment_67/allCRPSs.pkl', 'rb') as f:
    long_crps_cnn = pickle.load(f)

with open('/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/NN/Experiment_69/allCRPSs.pkl', 'rb') as f:
    long_crps_vit = pickle.load(f)

with open('/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/NN/Experiment_68/allCRPSs.pkl', 'rb') as f:
    long_crps_ann = pickle.load(f)

with open('/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/Experiment_66/allCRPSs.pkl', 'rb') as f:
    long_crps_vgam = pickle.load(f)

with open('/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/Baselines/EmpiricalCRPS.pkl', 'rb') as f:
    long_crps_baseline = pickle.load(f)

with open('/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/Baselines/PersistantCRPS.pkl', 'rb') as f:
    long_crps_persistant = pickle.load(f)
    
long_crps_best_possible = np.tile(long_crps_persistant[:,:,:339], (1,1,33))

year = 2021
days = t_array[:, 0] * 107 + 242
hour = (
    np.angle([complex(t_array[i, 1], t_array[i, 2]) for i in range(len(t_array))])
    * 12
    / np.pi
)
hour += 24 * (hour < 0)
target_dates = pd.DatetimeIndex(
    [
        pd.Timestamp(f"{year}-01-01")
        + pd.Timedelta(days=round(days[i]) - 1, hours=round(hour[i]))
        for i in range(len(t_array))
    ]
).unique()

ref_dates = icon_da.forecast_reference_time.values

# shape : (n_fold, n_clusters, n_ref_dates, n_lead_times)
crps_cnn_ref = np.full((5,5,len(ref_dates),len(lead_times)), np.nan)
crps_vgam_ref = np.full((5,5,len(ref_dates),len(lead_times)), np.nan)
crps_ann_ref = np.full((5,5,len(ref_dates),len(lead_times)), np.nan)
crps_vit_ref = np.full((5,5,len(ref_dates),len(lead_times)), np.nan)
crps_baseline_ref = np.full((5,5,len(ref_dates),len(lead_times)), np.nan)
crps_best_possible_ref = np.full((5,5,len(ref_dates),len(lead_times)), np.nan)

for i_lt, lead_time in enumerate(lead_times):
    for i_rd, ref_date in enumerate(icon_ds.forecast_reference_time.values):
        target_date = ref_date + pd.to_timedelta(lead_time, unit = 'h')
        if target_date in target_dates:
            
            i_d = np.argwhere(target_dates == target_date)[0][0]+339*i_lt
            crps_cnn_ref[:,:,i_rd,i_lt] = long_crps_cnn[:,:,i_d]
            crps_ann_ref[:,:,i_rd,i_lt] = long_crps_ann[:,:,i_d]
            crps_vit_ref[:,:,i_rd,i_lt] = long_crps_vit[:,:,i_d]
            crps_vgam_ref[:,:,i_rd,i_lt] = long_crps_vgam[:,:,i_d]
            crps_baseline_ref[:,:,i_rd,i_lt] = long_crps_baseline[:,:,i_d]
            crps_best_possible_ref[:,:,i_rd,i_lt] = long_crps_best_possible[:,:,i_d]
mask_cnn = np.any(np.isnan(crps_cnn_ref), axis = 0, keepdims = True)
mask_ann = np.any(np.isnan(crps_ann_ref), axis = 0, keepdims = True)
mask_vit = np.any(np.isnan(crps_vit_ref), axis = 0, keepdims = True)
mask_vgam = np.any(np.isnan(crps_vgam_ref), axis = 0, keepdims = True)
mask_baseline = np.any(np.isnan(crps_baseline_ref), axis = 0, keepdims = True)
mask_best_possible = np.any(np.isnan(crps_best_possible_ref), axis = 0, keepdims = True)
mask = np.logical_or(mask_cnn, mask_ann)
mask = np.logical_or(mask, mask_vit)
mask = np.logical_or(mask, mask_vgam)
mask = np.logical_or(mask, mask_baseline)
mask = np.logical_or(mask, mask_best_possible)

mask_5fold = np.tile(mask, (5,1,1,1))
mask_21fold = np.tile(mask, (21,1,1,1))

crps_cnn_ref = np.where(mask, np.nan, crps_cnn_ref)
crps_ann_ref = np.where(mask, np.nan, crps_ann_ref)
crps_vit_ref = np.where(mask, np.nan, crps_vit_ref)
crps_vgam_ref = np.where(mask, np.nan, crps_vgam_ref)
crps_baseline_ref = np.where(mask, np.nan, crps_baseline_ref)
crps_best_possible_ref = np.where(mask, np.nan, crps_best_possible_ref)

crps_cnn_ref = crps_cnn_ref[:,:,:143,:]
crps_ann_ref = crps_ann_ref[:,:,:143,:]
crps_vit_ref = crps_vit_ref[:,:,:143,:]
crps_vgam_ref = crps_vgam_ref[:,:,:143,:]
crps_baseline_ref = crps_baseline_ref[:,:,:143,:]
crps_best_possible_ref = crps_best_possible_ref[:,:,:143,:]


crps_cnn = np.nanmean(crps_cnn_ref, axis = (1,2))
crps_vit = np.nanmean(crps_vit_ref, axis = (1,2))
crps_ann = np.nanmean(crps_ann_ref, axis = (1,2))
crps_vgam = np.nanmean(crps_vgam_ref, axis = (1,2))
crps_baseline = np.nanmean(crps_baseline_ref, axis = (1,2))
crps_best_possible = np.nanmean(crps_best_possible_ref, axis = (1,2))

crpss_cnn = (crps_cnn - crps_baseline)/(crps_best_possible - crps_baseline)
crpss_vit = (crps_vit - crps_baseline)/(crps_best_possible - crps_baseline)
crpss_ann = (crps_ann - crps_baseline)/(crps_best_possible - crps_baseline)
crpss_vgam = (crps_vgam - crps_baseline)/(crps_best_possible - crps_baseline)

def var_icon_crps(icon_ds, obs_da, t_array, year, clusters, lead_times):
    """
    From the ICON forecast, create a crps array with the same format as the
    predictions from post-processing methods.
    
    Parameters
    ----------
    icon_ds: xr.Dataset
        Dataset containing the ICON forecast.
    obs_da: xr.DataArray
        Observations at each station.
    t_array: np.array
        Time array output from Experiment or RExperiment.
    year: int
        Year of the forecast.
    clusters: list of list
        List of list of stations.
    lead_times: list
        List of lead times.
    
    Returns
    -------
    crps: np.array
        CRPS of the forecast. Shape (n_realisations, n_clusters, n_lead_times).
    """
    stations = obs_da.station.values
    icon_da = icon_ds.sel(station = stations,
                          lead_time = pd.to_timedelta(lead_times, unit = 'h')).wind_speed_of_gust
    i_clusters = [
        [np.argwhere(stations == s)[0][0] for s in cluster] for cluster in clusters
    ]
    # Inverting the time array to get the dates
    days = t_array[:, 0] * 107 + 242
    hour = (
        np.angle([complex(t_array[i, 1], t_array[i, 2]) for i in range(len(t_array))])
        * 12
        / np.pi
    )
    hour += 24 * (hour < 0)
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{year}-01-01")
            + pd.Timedelta(days=round(days[i]) - 1, hours=round(hour[i]))
            for i in range(len(t_array))
        ]
    ).unique()
    # Create a new xr.DataArray with coordinates forecast_reference_time and lead_time,
    # its values being the observations. Lead time should be a new coordinate based on lead_times.
    # forecast_reference_time should be equal to time minus lead_time.
    obs_values = np.full((len(icon_ds.forecast_reference_time.values),
                          len(lead_times),
                          len(stations)),
                         np.nan)
    for i_lt, lead_time in enumerate(lead_times):
        for i_rd, ref_date in enumerate(icon_ds.forecast_reference_time.values):
            date = ref_date + pd.to_timedelta(lead_time, unit = 'h')
            if date in dates:
                obs_values[i_rd, i_lt, :] = obs_da.sel(
                    time = date).values
    new_obs_da = xr.DataArray(obs_values,
                          coords = {'forecast_reference_time': icon_ds.forecast_reference_time,
                                    'lead_time': pd.to_timedelta(lead_times, unit = 'h'),
                                    'station': stations},
                          dims = ['forecast_reference_time', 'lead_time', 'station'])
    np_fcst = icon_da.values
    np_obs = new_obs_da.values
    res_fcst = []
    res_obs = []
    for cluster in i_clusters:
        res_fcst.append(np_fcst[:, :, cluster])
        res_obs.append(np_obs[:, :, cluster])
    # res_fcst is a list of array of shape (ref_dates, lead_times, stations, realisation)
    crps = np.full((len(clusters), res_fcst[0].shape[-1], res_fcst[0].shape[0], len(lead_times)), np.nan)
    
    for i_cluster in range(len(res_fcst)):
        for i_real in range(res_fcst[i_cluster].shape[-1]):
            obs = res_obs[i_cluster]
            obs = obs.reshape(-1, obs.shape[-1])
            fcst = res_fcst[i_cluster][:, :, :, i_real]
            fcst = fcst.reshape(-1, fcst.shape[-1])
            tmp_crps = compute_ecdf_crps(obs, fcst)
            # shape of tmp_crps: (dates,lead_times)
            tmp_crps = tmp_crps.reshape(-1, len(lead_times))
            # shape of tmp_crps: (dates,lead_times)
            crps[i_cluster, i_real, :, :] = tmp_crps
    # #crps is of shape (n_clusters, n_realisations, n_ref_dates, n_lead_times)
    crps = np.transpose(crps, (1, 0, 2, 3))
    return crps

var_crps = var_icon_crps(icon_ds, obs_da, t_array, year, clusters, lead_times)

# crps_icon = np.nanmean(np.where(mask_21fold, np.nan, var_crps), axis = (1,2))
crps_icon = np.nanmean(var_crps[:,:,:143,:], axis = (1,2))

crpss_icon = (crps_icon - np.tile(crps_baseline[[1]], (21,1)))/(np.tile(crps_best_possible[[1]], (21,1)) - np.tile(crps_baseline[[1]], (21,1)))

# Create one DataArray per crpss
crpss_cnn_da = xr.DataArray(crpss_cnn,
                            coords = {'realisation': range(5),
                                      'lead_time': pd.to_timedelta(lead_times, unit = 'h')},
                            dims = ['realisation', 'lead_time'],
                            name = 'CRPSS')
crpss_vit_da = xr.DataArray(crpss_vit,
                            coords = {'realisation': range(5),
                                      'lead_time': pd.to_timedelta(lead_times, unit = 'h')},
                            dims = ['realisation', 'lead_time'],
                            name = 'CRPSS')
crpss_ann_da = xr.DataArray(crpss_ann,
                            coords = {'realisation': range(5),
                                      'lead_time': pd.to_timedelta(lead_times, unit = 'h')},
                            dims = ['realisation', 'lead_time'],
                            name = 'CRPSS')
crpss_vgam_da = xr.DataArray(crpss_vgam,
                            coords = {'realisation': range(5),
                                      'lead_time': pd.to_timedelta(lead_times, unit = 'h')},
                            dims = ['realisation', 'lead_time'],
                            name = 'CRPSS')
crpss_icon_da = xr.DataArray(crpss_icon,
                            coords = {'realisation': range(21),
                                      'lead_time': pd.to_timedelta(lead_times, unit = 'h')},
                            dims = ['realisation', 'lead_time'],
                            name = 'CRPSS')

# Create a giant pd.DataFrame, adding a supplementary column which is the name of each model
crpss_cnn_df = crpss_cnn_da.to_dataframe()
crpss_cnn_df['Model'] = 'CNN'
crpss_vit_df = crpss_vit_da.to_dataframe()
crpss_vit_df['Model'] = 'ViT'
crpss_ann_df = crpss_ann_da.to_dataframe()
crpss_ann_df['Model'] = 'ANN'
crpss_vgam_df = crpss_vgam_da.to_dataframe()
crpss_vgam_df['Model'] = 'VGAM'
crpss_icon_df = crpss_icon_da.to_dataframe()
crpss_icon_df['Model'] = 'ICON'

crpss_df = pd.concat([crpss_cnn_df, crpss_vit_df, crpss_ann_df, crpss_vgam_df, crpss_icon_df])
# Make the lead time a column instead of an index
crpss_df.reset_index(inplace = True)
# Make the lead time a float instead of a timedelta
crpss_df['lead_time'] = crpss_df['lead_time'].dt.total_seconds() / 3600
# Plot the CRPSS, with lead time in hours as x-axis
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style = 'whitegrid')
plt.figure(figsize = (10, 5))
sns.lineplot(data = crpss_df, x = 'lead_time', y = 'CRPSS', hue = 'Model')
plt.xlabel('Lead time (hours)')
plt.ylim(0.,1.5)
plt.ylabel('CRPSS')
plt.title('CRPSS of the different models')
plt.savefig('/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/delete_me/test4.png')
plt.close()
