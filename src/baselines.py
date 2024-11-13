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
