import xarray as xr
import numpy as np
import pandas as pd


def compute_ecdf_crps(obs, fcst):
    """
    Compute the CRPS of the forecast using the empirical CDF of the observations.
    
    Parameters
    ----------
    obs: np.array
        Observations at each station of shape (batch, n_station).
    fcst: np.array
        Forecast at each station of shape (batch, n_station).
        
    Returns
    -------
    crps: np.array
        CRPS of the forecast. Shape (batch,).
    """
    assert obs.shape[0] == fcst.shape[0],\
        "Batch size of obs and fcst should be the same."
    batch_size = obs.shape[0]
    n_station = obs.shape[1]
    crps = np.zeros(batch_size)
    count = 0
    # Formula of CRPS for ensemble, adapted to the case of an empirical distribution
    # forecast.
    for ist in range(n_station):
        tmp = np.abs(fcst - obs[:, [ist]]).mean(axis=1) - 0.5 * np.abs(
            [
                fcst[:, i] - fcst[:, j]
                for i in range(n_station)
                for j in range(n_station)
            ]
        ).mean(axis=0)
        crps += tmp
        count += 1
    crps /= count
    return crps


def crps_arr(obs, fcst, nfold):
    """
    The function outputs the CRPS of the baseline. obs and fcst should be iterable
    of np.array of shape (n_time, n_station), each element of the list
    corresponding to a different cluster.
    
    Parameters
    ----------
    obs: list of np.array
        Observations at each station of shape (n_time, n_station). Each element of
        the list corresponds to a different cluster.
    fcst: list of np.array
        Forecast at each station of shape (n_time, n_station). Each element of
        the list corresponds to a different cluster.
    nfold: int
        Number of folds used for the cross-validation.
    
    Returns
    -------
    crps: np.array
        CRPS of the forecast. Shape (nfold, len(obs), n_cluster).
    """
    assert len(obs) == len(fcst)
    crps = np.zeros((len(obs), obs[0].shape[0]))
    for i_cluster in range(len(obs)):
        crps[i_cluster] = compute_ecdf_crps(obs[i_cluster], fcst[i_cluster])
    return np.repeat(np.expand_dims(crps, axis=0), nfold, axis=0)


def compute_ecdf_crps_climatology(obs, climatology):
    """
    Compute the CRPS of the climatology forecast using the empirical CDF of the observations.
    
    Parameters
    ----------
    obs: np.array
        Observations at each station of shape (batch, n_station).
    climatology: np.array
        Climatology forecast at each station of shape (n_values, 2) with first column
        being the values and second the repeats (formatted as so to gain computational
        efficiency).
        
    Returns
    -------
    crps: np.array
        CRPS of the forecast. Shape (batch,).
    """
    batch_size = obs.shape[0]
    n_station = obs.shape[1]
    crps = np.zeros(batch_size)
    count = 0
    # Formula of CRPS for ensemble, adapted to the case of an empirical distribution
    # forecast.
    adjustment = 0.5 * np.average(
        np.abs(
            [
                climatology[i, 0] - climatology[j, 0]
                for i in range(len(climatology))
                for j in range(len(climatology))
            ]
        ),
        weights=[
            climatology[i, 1] * climatology[j, 1]
            for i in range(len(climatology))
            for j in range(len(climatology))
        ],
    )
    clim = np.repeat(np.expand_dims(climatology[:, 0], axis=0), batch_size, axis=0)
    for ist in range(n_station):
        tmp = np.average(
            np.abs(clim - obs[:, [ist]]), weights=climatology[:, 1], axis=1
        )
        crps += tmp
        count += 1
    crps /= count
    return crps - adjustment


def crps_arr_climatology(obs, climatology, n_fold):
    """
    The function outputs the CRPS of the baseline. obs should be iterable of np.array
    of shape (n_time, n_station), each element of the list corresponding to a
    different cluster. climatology should be iterable of np.array of shape
    (n_values, 2), each element of the list corresponding to a different cluster.
    
    Parameters
    ----------
    obs: list of np.array
        Observations at each station of shape (n_time, n_station). Each element of
        the list corresponds to a different cluster.
    climatology: list of np.array
        Climatology forecast at each station of shape (n_values, 2) with first column
        being the values and second the repeats (formatted as so to gain computational
        efficiency). Each element of the list corresponds to a different cluster.
        
    Returns
    -------
    crps: np.array
        CRPS of the forecast. Shape (nfold, len(obs), n_cluster).
    """
    crps = np.zeros((len(obs), obs[0].shape[0]))
    for i_cluster in range(len(obs)):
        print("Cluster: ", i_cluster)
        crps[i_cluster] = compute_ecdf_crps_climatology(obs[i_cluster], climatology[i_cluster])
    return np.repeat(np.expand_dims(crps, axis=0), n_fold, axis=0)


def pointwise_baseline(obs_da, fcst_da, t_array, year, clusters):
    ### To be looked at again
    """
    Baseline that predicts the last observed value. It means that it will have
    best performance for lead time 0h, and the performance will decrease as the
    lead time increases. obs_da and fcst_da are xr.DataArray, tarray is the output
    time array from Experiment or RExperiment. clusters is a list of list of stations.
    
    Parameters
    ----------
    obs_da: xr.DataArray
        Observations at each station.
    fcst_da: xr.DataArray
        Forecast at each station. To create a proper pointwise baseline, the forecast
        should be the same at each lead time.
    tarray: np.array
        Time array output from Experiment or RExperiment.
    year: int
        Year of the forecast.
    clusters: list of list
        List of list of stations.
        
    Returns
    -------
    res_obs: list of np.array
        Observations at each station of shape (n_time, n_station). Each element of
        the list corresponds to a different cluster. n_time is the number of dates
        in tarray (including lead times).
    res_fcst: list of np.array
        Forecast at each station of shape (n_time, n_station). Each element of
        the list corresponds to a different cluster. n_time is the number of dates
        in tarray (including lead times).
    """
    stations = obs_da.station.values
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
    np_obs = obs_da.sel(time=dates).values  # shape dates, stations
    np_fcst = fcst_da.sel(time=dates).values  # shape lead_times, dates, stations
    np_fcst = np_fcst.reshape(-1, np_fcst.shape[-1]) # shape dates*lead_times, stations
    np_obs = np.tile(np_obs, (len(np.unique(t_array[:, -1])), 1))
    res_obs = []
    res_fcst = []
    for cluster in i_clusters:
        res_obs.append(np_obs[:, cluster])
        res_fcst.append(np_fcst[:, cluster])
    return res_obs, res_fcst

def icon_crps(icon_ds, obs_da, t_array, year, clusters, lead_times):
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
    lead_times_6h = [i for i in lead_times if i%6 == 0]
    stations = obs_da.station.values
    icon_da = icon_ds.sel(station = stations,
                          lead_time = pd.to_timedelta(lead_times_6h, unit = 'h')).wind_speed_of_gust
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
                          len(lead_times_6h),
                          len(stations)),
                         np.nan)
    for i_lt, lead_time in enumerate(lead_times_6h):
        for i_rd, ref_date in enumerate(icon_ds.forecast_reference_time.values):
            date = ref_date + pd.to_timedelta(lead_time, unit = 'h')
            if date in obs_da.time.values:
                obs_values[i_rd, i_lt, :] = obs_da.sel(
                    time = date).values # problem here, no lt take to find date from large array!
    new_obs_da = xr.DataArray(obs_values,
                          coords = {'forecast_reference_time': icon_ds.forecast_reference_time,
                                    'lead_time': pd.to_timedelta(lead_times_6h, unit = 'h'),
                                    'station': stations},
                          dims = ['forecast_reference_time', 'lead_time', 'station'])
    np_fcst = icon_da.values
    np_obs = new_obs_da.values
    mask_obs = np.logical_not(np.any(np.isnan(np_obs), axis = (1,2)))
    mask_fcst = np.logical_not(np.any(np.isnan(np_fcst), axis = (1,2,3)))
    mask = np.logical_and(mask_obs, mask_fcst)
    np_fcst = np_fcst[mask]
    np_obs = np_obs[mask]
    res_fcst = []
    res_obs = []
    for cluster in i_clusters:
        res_fcst.append(np_fcst[:, :, cluster])
        res_obs.append(np_obs[:, :, cluster])
    # res_fcst is a list of array of shape (ref_dates, lead_times, stations, realisation)
    crps = np.zeros((len(clusters), len(lead_times_6h), res_fcst[0].shape[-1]))
    
    for i_cluster in range(len(res_fcst)):
        for i_real in range(res_fcst[i_cluster].shape[-1]):
            obs = res_obs[i_cluster]
            obs = obs.reshape(-1, obs.shape[-1])
            fcst = res_fcst[i_cluster][:, :, :, i_real]
            fcst = fcst.reshape(-1, fcst.shape[-1])
            tmp_crps = compute_ecdf_crps(obs, fcst)
            # shape of tmp_crps: dates*lead_times
            tmp_crps = tmp_crps.reshape(-1, len(lead_times_6h))
            tmp_crps = np.nanmean(tmp_crps, axis = 0)
            crps[i_cluster, :, i_real] = tmp_crps
    #crps is of shape (n_clusters, n_lead_times, n_realisations)
    crps = np.transpose(crps, (2, 0, 1))
    # crps is of shape (n_realisations, n_clusters, n_lead_times)
    return crps

    # xr.DataArray(crps,
    #              coords = {'realisation': np.arange(crps.shape[0]),
    #                         'cluster': np.arange(crps.shape[1]),
    #                         'lead_time': lead_times_6h},
    #               dims = ['realisation', 'cluster', 'lead_time'],
    #               name = 'crps')