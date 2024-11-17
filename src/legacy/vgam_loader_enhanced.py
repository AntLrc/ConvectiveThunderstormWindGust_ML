import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

import os

from src.vgam_loader import RExperiment

class EnhancedRPlotter:
    def __init__(self, experiment):
        """
        Class to plot the results of the experiment.
        
        Parameters
        ----------
        experiment : Experiment
            The experiment to plot.
        """
        self.experiment = experiment
        

    def case_study(
        self,
        date,
        lead_time,
        weather_map_file=None,
        all_lts=False,
        cluster_nb=None,
        precomputed=False,
    ):
        strdate = date.strftime("%Y%m%d%H%M")
        stations = (
            xr.open_dataset(self.experiment.files["test"]["labels"][0])
            .isel(time=0)
            .to_dataframe()[["latitude", "longitude"]]
        )
        stations = gpd.GeoDataFrame(
            stations,
            geometry=gpd.points_from_xy(stations.longitude, stations.latitude),
            crs="EPSG:4326",
        )
        file = None
        for f in self.experiment.files["test"]["inputs"]:
            if date in xr.open_dataset(f).time.values:
                file = f
                break
        if file is None:
            raise ValueError(f"No test file found for date {date}")
        l_file = None
        for f in self.experiment.files["test"]["labels"]:
            if date in xr.open_dataset(f).time.values:
                l_file = f
                break
        if l_file is None:
            raise ValueError(f"No test label file found for date {date}")
        # Create inputs
        ds0 = xr.open_dataset(file, engine="netcdf4").sel(time=date)
        ds2d = xr.open_dataset(weather_map_file, engine="netcdf4").sel(
            time=date, lead_time=lead_time
        )
        temperature = ds2d.t2m
        pressure = ds2d.msl
        wind = ds2d[["u10", "v10"]]
        np_inputs_s = [None] * self.experiment.clusters["n"]
        np_inputs_l = [None] * self.experiment.clusters["n"]
        if (
            self.experiment.Data["mean"] is None
            or self.experiment.Data["std"] is None
        ):
            self.experiment.load_mean_std()
        if not precomputed:
            if not all_lts:
                ds = ds0.sel(lead_time=lead_time)
                for i_var in range(len(self.experiment.features)):
                    # First obtaining corresponding data array
                    var = self.experiment.features[i_var]
                    if len(var.split("_")) == 2:
                        var, var_level = var.split("_")
                        var_level = int(var_level[:-3])
                        if var == "wind":
                            tmp = np.sqrt(
                                ds.sel(isobaricInhPa=var_level)["u10"] ** 2
                                + ds.sel(isobaricInhPa=var_level)["v10"] ** 2
                            )
                        else:
                            tmp = ds.sel(isobaricInhPa=var_level)[var]
                    else:
                        if var == "wind":
                            tmp = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
                        else:
                            tmp = ds[var]
                    # Selecting for each cluster, for each leadtime and converting to numpy array
                    for i_cluster in range(self.experiment.clusters["n"]):
                        tmp_cluster = tmp.sel(
                            station=self.experiment.clusters["groups"][i_cluster]
                        ).values
                        tmp_cluster = (
                            tmp_cluster - self.experiment.Data["mean"][i_var]
                        ) / self.experiment.Data["std"][i_var]
                        tmp_cluster = np.expand_dims(
                            tmp_cluster, axis=len(tmp_cluster.shape)
                        )
                        np_inputs_s[i_cluster] = (
                            tmp_cluster
                            if np_inputs_s[i_cluster] is None
                            else np.concatenate(
                                [np_inputs_s[i_cluster], tmp_cluster], axis=-1
                            )
                        )
                for i_cluster in range(self.experiment.clusters["n"]):
                    if self.experiment.model_kwargs["data"] == "normal":
                        np_inputs_s[i_cluster] = (
                            np_inputs_s[i_cluster]
                            .astype(np.float32)
                            .reshape(-1, np_inputs_s[i_cluster].shape[-1])
                        )
                        df = pd.DataFrame(
                            np_inputs_s[i_cluster], columns=self.experiment.features
                        )
                    elif self.experiment.model_kwargs["data"] == "mean":
                        # For each cluster, for each feature, mean of the feature over the stations
                        np_inputs_s[i_cluster] = (
                            np_inputs_s[i_cluster]
                            .astype(np.float32)
                            .mean(axis=0, keepdims=True)
                        )
                        np_inputs_s[i_cluster] = np.repeat(
                            np_inputs_s[i_cluster],
                            len(self.experiment.clusters["groups"][i_cluster]),
                            axis=0,
                        )
                        df = pd.DataFrame(
                            np_inputs_s[i_cluster], columns=self.experiment.features
                        )
                    df.to_csv(
                        os.path.join(
                            self.experiment.folders["scratch"]["folder"],
                            f"CASESTUDY_{i_cluster}_{strdate}_{lead_time}.csv",
                        ),
                        index=False,
                    )
                for fold in range(self.experiment.model_kwargs["n_folds"]):
                    for i_cluster in range(self.experiment.clusters["n"]):
                        os.system(
                            f"Rscript {self.experiment.files['R']['predict']} --test-predictors {os.path.join(self.experiment.folders['scratch']['folder'], f'CASESTUDY_{i_cluster}_{strdate}_{lead_time}.csv')} --output {os.path.join(self.experiment.folders['scratch']['folder'], f'CASESTUDY_{i_cluster}_{fold}_{strdate}_{lead_time}_preds.csv')} --model-file {os.path.join(self.experiment.folders['plot']['model'], f'{i_cluster}_{lead_time}_{fold}.rds')} --source {self.experiment.files['R']['source']}"
                        )
            else:
                for lt in self.experiment.filter["lead_times"]:
                    ds = ds0.sel(lead_time=lt)
                    np_inputs_s = [None] * self.experiment.clusters["n"]
                    np_inputs_l = [None] * self.experiment.clusters["n"]
                    for i_var in range(len(self.experiment.features)):
                        # First obtaining corresponding data array
                        var = self.experiment.features[i_var]
                        if len(var.split("_")) == 2:
                            var, var_level = var.split("_")
                            var_level = int(var_level[:-3])
                            if var == "wind":
                                tmp = np.sqrt(
                                    ds.sel(isobaricInhPa=var_level)["u10"] ** 2
                                    + ds.sel(isobaricInhPa=var_level)["v10"] ** 2
                                )
                            else:
                                tmp = ds.sel(isobaricInhPa=var_level)[var]
                        else:
                            if var == "wind":
                                tmp = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
                            else:
                                tmp = ds[var]
                        # Selecting for each cluster, for each leadtime and converting to numpy array
                        for i_cluster in range(self.experiment.clusters["n"]):
                            tmp_cluster = tmp.sel(
                                station=self.experiment.clusters["groups"][i_cluster]
                            ).values
                            tmp_cluster = (
                                tmp_cluster - self.experiment.Data["mean"][i_var]
                            ) / self.experiment.Data["std"][i_var]
                            tmp_cluster = np.expand_dims(
                                tmp_cluster, axis=len(tmp_cluster.shape)
                            )
                            np_inputs_s[i_cluster] = (
                                tmp_cluster
                                if np_inputs_s[i_cluster] is None
                                else np.concatenate(
                                    [np_inputs_s[i_cluster], tmp_cluster], axis=-1
                                )
                            )
                    for i_cluster in range(self.experiment.clusters["n"]):
                        if self.experiment.model_kwargs["data"] == "normal":
                            np_inputs_s[i_cluster] = (
                                np_inputs_s[i_cluster]
                                .astype(np.float32)
                                .reshape(-1, np_inputs_s[i_cluster].shape[-1])
                            )
                            df = pd.DataFrame(
                                np_inputs_s[i_cluster],
                                columns=self.experiment.features,
                            )
                        elif self.experiment.model_kwargs["data"] == "mean":
                            # For each cluster, for each feature, mean of the feature over the stations
                            np_inputs_s[i_cluster] = (
                                np_inputs_s[i_cluster]
                                .astype(np.float32)
                                .mean(axis=0, keepdims=True)
                            )
                            np_inputs_s[i_cluster] = np.repeat(
                                np_inputs_s[i_cluster],
                                len(self.experiment.clusters["groups"][i_cluster]),
                                axis=0,
                            )
                            df = pd.DataFrame(
                                np_inputs_s[i_cluster],
                                columns=self.experiment.features,
                            )
                        df.to_csv(
                            os.path.join(
                                self.experiment.folders["scratch"]["folder"],
                                f"CASESTUDY_{i_cluster}_{strdate}_{lt}.csv",
                            ),
                            index=False,
                        )
                    for fold in range(self.experiment.model_kwargs["n_folds"]):
                        for i_cluster in range(self.experiment.clusters["n"]):
                            os.system(
                                f"Rscript {self.experiment.files['R']['predict']} --test-predictors {os.path.join(self.experiment.folders['scratch']['folder'], f'CASESTUDY_{i_cluster}_{strdate}_{lt}.csv')} --output {os.path.join(self.experiment.folders['scratch']['folder'], f'CASESTUDY_{i_cluster}_{fold}_{strdate}_{lt}_preds.csv')} --model-file {os.path.join(self.experiment.folders['plot']['model'], f'{i_cluster}_{lt}_{fold}.rds')} --source {self.experiment.files['R']['source']}"
                            )
        # Labels
        labels = xr.open_dataset(l_file, engine="netcdf4")
        labels = labels.sel(time=date)[self.experiment.label]
        for i_cluster in range(self.experiment.clusters["n"]):
            lab_tmp = labels.sel(
                station=self.experiment.clusters["groups"][i_cluster]
            ).values
            np_inputs_l[i_cluster] = (
                lab_tmp
                if np_inputs_l[i_cluster] is None
                else np.concatenate([np_inputs_l[i_cluster], lab_tmp], axis=0)
            )
        for i_cluster in range(self.experiment.clusters["n"]):
            np_inputs_l[i_cluster] = (
                np_inputs_l[i_cluster].astype(np.float32).reshape(-1)
            )

        # Build gev_params and mus, sigmas, xis
        mus = [
            [
                pd.read_csv(
                    os.path.join(
                        self.experiment.folders["scratch"]["folder"],
                        f"CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv",
                    )
                ).values[:, 0]
                for icluster in range(self.experiment.clusters["n"])
            ]
            for fold in range(self.experiment.model_kwargs["n_folds"])
        ]
        sigmas = [
            [
                pd.read_csv(
                    os.path.join(
                        self.experiment.folders["scratch"]["folder"],
                        f"CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv",
                    )
                ).values[:, 1]
                for icluster in range(self.experiment.clusters["n"])
            ]
            for fold in range(self.experiment.model_kwargs["n_folds"])
        ]
        xis = [
            [
                pd.read_csv(
                    os.path.join(
                        self.experiment.folders["scratch"]["folder"],
                        f"CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv",
                    )
                ).values[:, 2]
                for icluster in range(self.experiment.clusters["n"])
            ]
            for fold in range(self.experiment.model_kwargs["n_folds"])
        ]

        gev_params = [
            [
                pd.read_csv(
                    os.path.join(
                        self.experiment.folders["scratch"]["folder"],
                        f"CASESTUDY_{icluster}_{fold}_{strdate}_{lead_time}_preds.csv",
                    )
                ).values
                for icluster in range(self.experiment.clusters["n"])
            ]
            for fold in range(self.experiment.model_kwargs["n_folds"])
        ]

        # Compute CRPS for each cluster
        crps_s = np.array(
            [
                [
                    gev_crps(
                        mus[ifold][icluster],
                        sigmas[ifold][icluster],
                        xis[ifold][icluster],
                        np_inputs_l[icluster],
                    ).mean()
                    for icluster in range(self.experiment.clusters["n"])
                ]
                for ifold in range(self.experiment.model_kwargs["n_folds"])
            ]
        )

        if all_lts:
            all_mus = [
                [
                    [
                        pd.read_csv(
                            os.path.join(
                                self.experiment.folders["scratch"]["folder"],
                                f"CASESTUDY_{icluster}_{fold}_{strdate}_{lt}_preds.csv",
                            )
                        ).values[:, 0]
                        for icluster in range(self.experiment.clusters["n"])
                    ]
                    for fold in range(self.experiment.model_kwargs["n_folds"])
                ]
                for lt in self.experiment.filter["lead_times"]
            ]
            all_sigmas = [
                [
                    [
                        pd.read_csv(
                            os.path.join(
                                self.experiment.folders["scratch"]["folder"],
                                f"CASESTUDY_{icluster}_{fold}_{strdate}_{lt}_preds.csv",
                            )
                        ).values[:, 1]
                        for icluster in range(self.experiment.clusters["n"])
                    ]
                    for fold in range(self.experiment.model_kwargs["n_folds"])
                ]
                for lt in self.experiment.filter["lead_times"]
            ]
            all_xis = [
                [
                    [
                        pd.read_csv(
                            os.path.join(
                                self.experiment.folders["scratch"]["folder"],
                                f"CASESTUDY_{icluster}_{fold}_{strdate}_{lt}_preds.csv",
                            )
                        ).values[:, 2]
                        for icluster in range(self.experiment.clusters["n"])
                    ]
                    for fold in range(self.experiment.model_kwargs["n_folds"])
                ]
                for lt in self.experiment.filter["lead_times"]
            ]
            all_crps_s = {
                lt: [
                    [
                        gev_crps(
                            all_mus[ilt][ifold][icluster],
                            all_sigmas[ilt][ifold][icluster],
                            all_xis[ilt][ifold][icluster],
                            np_inputs_l[icluster],
                        ).mean()
                        for icluster in range(self.experiment.clusters["n"])
                    ]
                    for ifold in range(self.experiment.model_kwargs["n_folds"])
                ]
                for ilt, lt in enumerate(self.experiment.filter["lead_times"])
            }
        storm_plot(
            stations,
            self.experiment.filter["storms"],
            gev_params,
            np_inputs_l,
            temperature,
            pressure,
            wind,
            self.experiment.clusters["groups"],
            crps_s,
            self.experiment.CRPS["mean"],
            date,
            os.path.join(
                self.experiment.folders["plot"]["folder"],
                f"Stormplot_{date}_{lead_time}.png",
            ),
            all_crps_s=all_crps_s,
            cluster_nb=cluster_nb,
        )
        return all_crps_s
