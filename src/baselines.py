import xarray as xr
import numpy as np
import pandas as pd

def computeECDFcrps(obs, fcst):
    """
    Compute the CRPS of the forecast using the empirical CDF of the observations.
    """
    # Shape of obs: (batch, n_station)
    # Shape of frcst: (batch, n_station)
    assert obs.shape[0] == fcst.shape[0]
    batch_size = obs.shape[0]
    n_station = obs.shape[1]
    crps = np.zeros(batch_size)
    count = 0
    # Sort obs on axis 1
    for ist in range(n_station):
        tmp = np.abs(fcst - obs[:, [ist]]).mean(axis=1) -\
            0.5*np.abs([fcst[:,i] - fcst[:,j] for i in range(n_station) for j in range(n_station)]).mean(axis=0)
        crps += tmp
        count += 1
    crps /= count
    return crps

def CRPSarr(obs, fcst, nfold):
    """
    The function outputs the CRPS of the baseline.
    obs and fcst should be iterable of np.array of shape (n_time, n_station)
    """
    assert len(obs) == len(fcst)
    CRPS = np.zeros((len(obs), obs[0].shape[0]))
    for i in range(len(obs)):
        CRPS[i] = computeECDFcrps(obs[i], fcst[i])
    return np.repeat(np.expand_dims(CRPS, axis = 0), nfold, axis = 0)

def computeECDFcrpsClimatology(obs, climatology):
    # Shape of obs: (batch, n_station)
    # Shape of climatology: (n_values, 2) with first column being the values and second the repeats
    batch_size = obs.shape[0]
    n_station = obs.shape[1]
    crps = np.zeros(batch_size)
    count = 0
    adjustment = 0.5*np.average(np.abs([climatology[i,0] - climatology[j,0] for i in range(len(climatology)) for j in range(len(climatology))]),
                                weights = [climatology[i,1]*climatology[j,1] for i in range(len(climatology)) for j in range(len(climatology))])
    clim = np.repeat(np.expand_dims(climatology[:,0], axis = 0), batch_size, axis = 0)
    for ist in range(n_station):
        tmp = np.average(np.abs(clim - obs[:, [ist]]), weights = climatology[:,1], axis = 1)
        crps += tmp
        count += 1
    crps /= count
    return crps - adjustment
    
def CRPSarrClimatology(obs, climatology, nfold):
    """
    The function outputs the CRPS of the baseline.
    obs should be iterable of np.array of shape (n_time, n_station)
    """
    CRPS = np.zeros((len(obs), obs[0].shape[0]))
    for i in range(len(obs)):
        print("Cluster: ", i)
        CRPS[i] = computeECDFcrpsClimatology(obs[i], climatology[i])
    return np.repeat(np.expand_dims(CRPS, axis = 0), nfold, axis = 0)

def pointwise_baseline(obs_da, fcst_da, tarray, year, clusters):
    """
    Baseline that predicts the last observed value.
    obs_da and fcst_da are xr.DataArray, tarray is the time array as output Experiment or RExperiment.
    clusters is a list of list of stations.
    """
    stations = obs_da.station.values
    iclusters = [
        [np.argwhere(stations == s)[0][0] for s in cluster] for cluster in clusters
    ]
    days = tarray[:,0]*107+242
    hour = np.angle([complex(tarray[i,1],
                            tarray[i,2]) for i in range(len(tarray))]
                    )*12/np.pi
    hour += 24*(hour < 0)
    dates = pd.DatetimeIndex(
        [pd.Timestamp(f'{year}-01-01')+\
            pd.Timedelta(days=round(days[i]) -1, hours=round(hour[i])) for i in range(len(tarray))]
    ).unique()
    npobs = obs_da.sel(time=dates).values # shape dates, stations
    npfcst = fcst_da.sel(time=dates).values # shape lead_times, dates, stations
    npfcst = npfcst.reshape(-1, npfcst.shape[-1])
    npobs = np.tile(npobs, (len(np.unique(tarray[:,-1])), 1))
    resobs = []
    resfcst = []
    for cluster in iclusters:
        resobs.append(npobs[:, cluster])
        resfcst.append(npfcst[:, cluster])
    return resobs, resfcst
