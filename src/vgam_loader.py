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

from utils import storm_plot, metrics_evolution


def double_exp(mu, sigma, y):
    """
    Custom definition of exp(-exp((y-mu)/sigma)) to avoid overflow. ONLY VALID FOR SIGMA AND MU
    "SLOWLY" VARYING (otehrwise, numerical instability may arise).
    
    Parameters
    ----------
    mu : float or array-like
        Location parameter.
    sigma : float or array-like
        Scale parameter.
    y : float or array-like
        Value at which to evaluate the function.
        
    Returns
    -------
    float or array-like
        Value of the double exponential function.
    """
    return np.exp(-np.exp(-(y - mu) / sigma))


def l_gamma(x):
    return gammaln(x)


def gev(mu, sigma, xi, y):
    """
    Computes the Generalized Extreme Value CDF.
    
    Parameters
    ----------
    mu : float or array-like
        Location parameter.
    sigma : float or array-like
        Scale parameter.
    xi : float or array-like
        Shape parameter.
    y : float or array-like
        Value at which to evaluate the CDF.
        
    Returns
    -------
    float or array-like
        Value of the Generalized Extreme Value CDF.
    """

    y_red = (y - mu) / sigma
    xi_null_mask = xi == 0
    xi_val = np.where(xi_null_mask, 0.5, xi)

    y0 = np.logical_and(xi > 0, y_red <= -1 / xi_val)
    y1 = np.logical_and(xi < 0, y_red >= -1 / xi_val)

    y_in_boundary = np.logical_or(
        np.logical_and(xi < 0, y_red < -1 / xi_val),
        np.logical_and(xi > 0, y_red > -1 / xi_val),
    )

    y_red_val = np.where(np.logical_or(y0, y1), (np.log(2) ** (-xi_val) - 1) / xi_val, y_red)

    return np.where(
        y_in_boundary,
        np.exp(-((1 + xi_val * y_red_val) ** (-1 / xi_val))),
        np.where(xi_null_mask, double_exp(mu, sigma, y), np.where(y1, 1.0, 0.0)),
    )


def gev_pdf(mu, sigma, xi, y):
    """
    Computes the Generalized Extreme Value PDF.
    
    Parameters
    ----------
    mu : float or array-like
        Location parameter.
    sigma : float or array-like
        Scale parameter.
    xi : float or array-like
        Shape parameter.
    y : float or array-like
        Value at which to evaluate the PDF.
        
    Returns
    -------
    float or array-like
        Value of the Generalized Extreme Value PDF.
    """
    y_red = (y - mu) / sigma
    y0 = np.logical_and(xi > 0, y_red <= -1 / xi)
    y1 = np.logical_and(xi < 0, y_red >= -1 / xi)

    xi_val = np.where(xi == 0, 0.5, xi)
    y_red_val = np.where(np.logical_or(y0, y1), (np.log(2) ** (-xi_val) - 1) / xi_val, y_red)

    y_in_boundary = np.logical_or(
        np.logical_and(xi < 0, y_red < -1 / xi_val),
        np.logical_and(xi > 0, y_red > -1 / xi_val),
    )

    y_in_boundary = np.logical_or(y_in_boundary, xi == 0)

    ty = np.where(xi == 0, np.exp(-y_red_val), (1 + xi_val * y_red_val) ** (-1 / xi))

    return np.where(y_in_boundary, (1 / sigma) * ty ** (xi + 1) * np.exp(-ty), 0.0)


def gev_crps(mu, sigma, xi, y):
    """
    Compute the closed form of the Continuous Ranked Probability Score (CRPS)
    for the Generalized Extreme Value distribution.
    Based on Friedrichs and Thorarinsdottir (2012).
    
    Parameters
    ----------
    mu : float or array-like
        Location parameter.
    sigma : float or array-like
        Scale parameter.
    xi : float or array-like
        Shape parameter.
    y : float or array-like
        Value at which to evaluate the CRPS.
        
    Returns
    -------
    float or array-like
        Value of the CRPS.
    """

    y_red = (y - mu) / sigma
    xi_null_mask = xi == 0
    xi_val = np.where(xi_null_mask, 0.5, xi)

    gev_val = gev(mu, sigma, xi, y)

    y0 = np.logical_and(xi > 0, y_red <= -1 / xi_val)
    y1 = np.logical_and(xi < 0, y_red >= -1 / xi_val)

    y_in_boundary = np.logical_and(
        np.logical_not(xi_null_mask), np.logical_not(np.logical_or(y1, y0))
    )

    y_red_val = np.where(np.logical_or(y0, y1), (np.log(2) ** (-xi_val) - 1) / xi_val, y_red)

    exp_y_red_null = -np.exp(np.where(xi_null_mask, -y_red, 0.0))

    return np.where(
        y_in_boundary,
        sigma * (-y_red_val - 1 / xi_val) * (1 - 2 * gev_val)
        - sigma
        / xi_val
        * np.exp(l_gamma(1 - xi_val))
        * (2**xi_val - 2 * gammainc(1 - xi_val, (1 + xi_val * y_red_val) ** (-1 / xi_val))),
        np.where(
            xi_null_mask,
            mu
            - y
            + sigma * (np.euler_gamma - np.log(2))
            - 2 * sigma * expi(exp_y_red_null),
            np.where(
                y1,
                sigma * (-y_red - 1 / xi_val) * (1 - 2 * gev_val)
                - sigma / xi_val * np.exp(l_gamma(1 - xi_val)) * 2**xi_val,
                sigma * (-y_red - 1 / xi_val) * (1 - 2 * gev_val)
                - sigma / xi_val * np.exp(l_gamma(1 - xi_val)) * (2**xi_val - 2),
            ),
        ),
    )


def visualise_gev(mu, sigma, xi, ys, save_path):
    """
    Visualise the Generalized Extreme Value distribution.
    
    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    xi : float
        Shape parameter.
    ys : array-like
        Values at which to evaluate the distribution.
    save_path : str
        Path to save the plot.
    """
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 9.6))
    sns.set_theme()
    x = np.linspace(-10, 50, 300)
    y_pdf = gev_pdf(np.repeat(mu, 300), np.repeat(sigma, 300), np.repeat(xi, 300), x)
    y_cdf = gev(np.repeat(mu, 300), np.repeat(sigma, 300), np.repeat(xi, 300), x)

    y_max = y_pdf.max()

    axs[0].plot(x, y_pdf)
    # Add ticks corresponding to the true values
    for i in range(len(ys)):
        # Find i such that x[i] is the closest to ys[i]
        i_ref = np.argmin(np.abs(x - ys[i]))
        axs[0].vlines(
            x=x[i_ref], ymin=-y_max / 50, ymax=y_pdf[i_ref], color="black", linewidths=0.5
        )

    # Plot empirical CDF
    axs[1].plot(x, y_cdf)
    sns.ecdfplot(ys, ax=axs[1], stat="proportion", color="black")

    plt.savefig(save_path)
    plt.close()


class RExperiment:

    def __init__(self, experiment_file):
        """
        Class to load the data and preprocess it for the VGAM / VGLM model.
        
        Parameters
        ----------
        experiment_file : str
            Path to the experiment file.
        """
        with open(experiment_file, "rb") as f:
            experiment = pickle.load(f)
        self.exp_number = experiment_file.split("_")[-1].split(".")[0]
        self.files = experiment["files"]
        default_files = {
            "storms": None,
            "clusters": None,
            "experiment": experiment_file,
            "train": {"inputs": [], "labels": None},
            "test": {"inputs": [], "labels": None},
            "R": {"script": None, "source": None, "predict": None},
        }
        for k, v in default_files.items():
            if not k in self.files.keys():
                self.files[k] = v
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if not subk in self.files[k].keys():
                        self.files[k][subk] = subv
        self.folders = experiment["folders"]
        default_folders = {
            "scratch": {"folder": None, "dir": None},
            "plot": {"folder": None, "dir": None, "model": None},
        }
        for k, v in default_folders.items():
            if not k in self.folders.keys():
                self.folders[k] = v
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if not subk in self.folders[k].keys():
                        self.folders[k][subk] = subv
        self.features = experiment["features"]
        self.label = experiment["label"]
        self.vgam_kwargs = experiment.get("vgam_kwargs", {})
        default_vgam_kwargs = {"model": "vglm", "spline_df": 3}
        for k, v in default_vgam_kwargs.items():
            if not k in self.vgam_kwargs.keys():
                self.vgam_kwargs[k] = v
        self.model_kwargs = experiment.get("model_kwargs", {})
        default_model_kwargs = {
            "data": "normal",
            "target": "GEV",
            "time_encoding": "sinusoidal",
            "n_folds": len(self.files["train"]["inputs"]),
            "seqfeatsel": False,
        }
        for k, v in default_model_kwargs.items():
            if not k in self.model_kwargs.keys():
                self.model_kwargs[k] = v
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if not subk in self.model_kwargs[k].keys():
                        self.model_kwargs[k][subk] = subv
        self.target = 0 if self.model_kwargs["target"] == "GEV" else 1  # encoding
        self.filter = experiment.get("filter", {})
        default_filter = {
            "lead_times": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                27,
                30,
                33,
                36,
                42,
                48,
                60,
                72,
            ],
            "storm_part": {"train": None, "test": None},
        }
        for k, v in default_filter.items():
            if not k in self.filter.keys():
                self.filter[k] = v
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if not subk in self.filter[k].keys():
                        self.filter[k][subk] = subv
        with open(self.files["storms"], "rb") as f:
            self.filter["storms"] = pickle.load(f)
        self.filter["dates"] = pd.DatetimeIndex(
            self.filter["storms"].dates.keys()
        ).unique()
        self.clusters = {
            "stations": experiment.get(
                "stations",
                xr.open_dataset(self.files["train"]["labels"][0]).station.values,
            ),
            "groups": pd.read_csv(self.files["clusters"], header=None),
        }
        self.clusters["n"] = len(self.clusters["groups"])
        self.clusters["groups"] = [
            list(
                np.intersect1d(
                    self.clusters["groups"].iloc[i, :].dropna(),
                    self.clusters["stations"],
                )
            )
            for i in range(self.clusters["n"])
        ]
        self.crps = experiment.get("CRPS", {})
        default_crps = {"mean": None, "std": None, "values": None}
        for k, v in default_crps.items():
            if not k in self.crps.keys():
                self.crps[k] = v
        self.loglik = experiment.get("LogLik", {})
        default_loglik = {"mean": None, "std": None, "values": None}
        for k, v in default_loglik.items():
            if not k in self.loglik.keys():
                self.loglik[k] = v
        self.data = experiment.get("Data", {})
        default_data = {"mean": None, "std": None}
        for k, v in default_data.items():
            if not k in self.data.keys():
                self.data[k] = v
        os.makedirs(self.folders["scratch"]["folder"], exist_ok=True)
        os.makedirs(self.folders["plot"]["folder"], exist_ok=True)
        if self.folders["plot"]["model"] is None:
            self.folders["plot"]["model"] = os.path.join(
                self.folders["plot"]["folder"], "models"
            )
        os.makedirs(self.folders["plot"]["model"], exist_ok=True)
        self.plot = self.Plotter(self)
        self.save = self.Saver(self)
        self.diag = self.Diagnostics(self)

    def load_mean_std(self):
        """
        Preparation to feature normalisation through review of training set.
        """
        print("Computing mean and std of features...", end="\n", flush=True)
        mean_ = np.zeros((len(self.features)))
        mean_sq = np.zeros((len(self.features)))
        count = np.zeros((len(self.features)))

        for file in self.files["train"]["inputs"]:
            print(file)
            inputs = xr.open_dataset(file, engine="netcdf4").sel(
                lead_time=self.filter["lead_times"]
            )
            loc_dates = pd.DatetimeIndex(inputs.time)
            if self.filter["storm_part"]["train"] is not None:
                seed, ratio = self.filter["storm_part"]["train"]
                rng = np.random.default_rng(seed)
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter["dates"])
                nostorm_dates = input_dates.difference(self.filter["dates"])
                if len(storm_dates) >= len(input_dates) * ratio:
                    storm_dates = storm_dates[
                        rng.permutation(len(storm_dates))[
                            : int(len(input_dates) * ratio)
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[
                        rng.permutation(len(nostorm_dates))[
                            : int(len(input_dates) * (1 - ratio))
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
            for i_var in range(len(self.features)):
                var = self.features[i_var]
                if len(var.split("_")) == 2:
                    var, var_level = var.split("_")
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(
                            inputs.sel(isobaricInhPa=var_level)["u"] ** 2
                            + inputs.sel(isobaricInhPa=var_level)["v"] ** 2
                        )
                    else:
                        tmp = inputs.sel(isobaricInhPa=var_level)[var]
                elif var == "wind":
                    tmp = np.sqrt(inputs["u10"] ** 2 + inputs["v10"] ** 2)
                elif var == "date":
                    tmp = inputs.time.dt.dayofyear.expand_dims(
                        {"station": inputs.station, "lead_time": inputs.lead_time},
                        axis=(-1, 0),
                    )
                elif var == "hour":
                    tmp = inputs.time.dt.hour.expand_dims(
                        {"station": inputs.station, "lead_time": inputs.lead_time},
                        axis=(-1, 0),
                    )
                else:
                    tmp = inputs[var]
                tmp = tmp.sel(time=loc_dates)
                tmp = tmp.values
                count[i_var] += len(tmp)
                mean_[i_var] += tmp.mean() * len(tmp)
                mean_sq[i_var] += (tmp**2).mean() * len(tmp)
            inputs.close()
        self.data["mean"] = mean_ / count
        self.data["std"] = np.sqrt(mean_sq / count - self.data["mean"] ** 2)
        self.save.experiment_file()
        print("Done.", flush=True)

    def create_inputs(self):
        """
        Creating  the inputs for the VGAM / VGLM model from netcdf files.
        """
        print("Creating inputs...", flush=True)
        if not self.model_kwargs["data"] in ["normal", "mean", "collective"]:
            raise ValueError("Mode must be 'normal', 'mean' or 'collective'.")
        # Creating the folds
        for file, labels, fold in zip(
            self.files["train"]["inputs"],
            self.files["train"]["labels"],
            range(len(self.files["train"]["inputs"])),
        ):
            # Each file will be use as a fold, ass it corresponds to a year
            inputs = xr.open_dataset(file, engine="netcdf4").sel(
                lead_time=self.filter["lead_times"]
            )
            loc_dates = pd.DatetimeIndex(inputs.time)
            if self.filter["storm_part"]["train"] is not None:
                seed, ratio = self.filter["storm_part"]["train"]
                rng = np.random.default_rng(seed)
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter["dates"])
                nostorm_dates = input_dates.difference(self.filter["dates"])
                if (ratio != 1) and len(storm_dates) >= len(nostorm_dates) * ratio / (
                    1 - ratio
                ):
                    storm_dates = storm_dates[
                        rng.permutation(len(storm_dates))[
                            : int(len(nostorm_dates) * ratio / (1 - ratio))
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[
                        rng.permutation(len(nostorm_dates))[
                            : int(len(storm_dates) * (1 - ratio) / ratio)
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
                storm_presence = np.zeros(len(loc_dates))
                storm_presence[loc_dates.isin(storm_dates)] = 1
                np_storms = [storm_presence] * self.clusters["n"]
            # Inputs
            np_inputs_s = [
                dict(
                    zip(
                        self.filter["lead_times"],
                        [None] * len(self.filter["lead_times"]),
                    )
                )
                for i in range(self.clusters["n"])
            ]
            for i_var in range(len(self.features)):
                # First obtaining corresponding data array
                var = self.features[i_var]
                if len(var.split("_")) == 2:
                    var, var_level = var.split("_")
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(
                            inputs.sel(isobaricInhPa=var_level)["u"] ** 2
                            + inputs.sel(isobaricInhPa=var_level)["v"] ** 2
                        )
                    else:
                        tmp = inputs.sel(isobaricInhPa=var_level)[var]
                else:
                    if var == "wind":
                        tmp = np.sqrt(inputs["u10"] ** 2 + inputs["v10"] ** 2)
                    elif var == "date":
                        tmp = inputs.time.dt.dayofyear.expand_dims(
                            {"station": inputs.station, "lead_time": inputs.lead_time},
                            axis=(-1, 0),
                        )
                    elif var == "hour":
                        tmp = inputs.time.dt.hour.expand_dims(
                            {"station": inputs.station, "lead_time": inputs.lead_time},
                            axis=(-1, 0),
                        )
                    else:
                        tmp = inputs[var]
                tmp = tmp.sel(time=loc_dates)
                # Selecting for each cluster, for each leadtime and converting to numpy array
                for i_cluster in range(self.clusters["n"]):
                    for lt in self.filter["lead_times"]:
                        tmp_cluster = tmp.sel(
                            station=self.clusters["groups"][i_cluster], lead_time=lt
                        ).values
                        tmp_cluster = (
                            tmp_cluster - self.data["mean"][i_var]
                        ) / self.data["std"][i_var]
                        tmp_cluster = np.expand_dims(tmp_cluster, axis=-1)
                        np_inputs_s[i_cluster][lt] = (
                            tmp_cluster
                            if np_inputs_s[i_cluster][lt] is None
                            else np.concatenate(
                                [np_inputs_s[i_cluster][lt], tmp_cluster], axis=-1
                            )
                        )
            # Labels
            labels = xr.open_dataset(labels, engine="netcdf4")
            labels = labels.sel(time=loc_dates)[self.label]
            np_labels = [None] * self.clusters["n"]
            for i_cluster in range(self.clusters["n"]):
                np_labels[i_cluster] = labels.sel(
                    station=self.clusters["groups"][i_cluster]
                ).values
                np_labels[i_cluster] = (
                    np_labels[i_cluster]
                    .astype(np.float32)
                    .reshape(np_labels[i_cluster].shape[0] * np_labels[i_cluster].shape[1])
                )
                np_storms[i_cluster] = np.repeat(
                    np.expand_dims(np_storms[i_cluster], axis=1),
                    len(self.clusters["groups"][i_cluster]),
                    axis=1,
                ).reshape(-1)
            for i_cluster in range(self.clusters["n"]):
                df_labels = pd.DataFrame(np_labels[i_cluster], columns=[self.label])
                df_labels.to_csv(
                    os.path.join(
                        self.folders["scratch"]["folder"],
                        f"{i_cluster}_fold_{fold}_labels.csv",
                    ),
                    index=False,
                )
                df_storms = pd.DataFrame(np_storms[i_cluster], columns=["storm"])
                df_storms.to_csv(
                    os.path.join(
                        self.folders["scratch"]["folder"],
                        f"{i_cluster}_fold_{fold}_storm.csv",
                    ),
                    index=False,
                )
                for lt in self.filter["lead_times"]:
                    if self.model_kwargs["data"] == "normal":
                        np_inputs_s[i_cluster][lt] = (
                            np_inputs_s[i_cluster][lt]
                            .astype(np.float32)
                            .reshape(-1, np_inputs_s[i_cluster][lt].shape[2])
                        )
                        df_inputs = pd.DataFrame(
                            np_inputs_s[i_cluster][lt], columns=self.features
                        )
                    elif self.model_kwargs["data"] == "mean":
                        # For each cluster, for each feature, mean of the feature over the stations
                        np_inputs_s[i_cluster][lt] = (
                            np_inputs_s[i_cluster][lt]
                            .astype(np.float32)
                            .mean(axis=1, keepdims=True)
                        )
                        np_inputs_s[i_cluster][lt] = np.repeat(
                            np_inputs_s[i_cluster][lt],
                            len(self.clusters["groups"][i_cluster]),
                            axis=1,
                        ).reshape(-1, np_inputs_s[i_cluster][lt].shape[2])
                        df_inputs = pd.DataFrame(
                            np_inputs_s[i_cluster][lt], columns=self.features
                        )
                    elif self.model_kwargs["data"] == "collective":
                        # For each time step, for each feature, concatenate the features of the stations
                        np_inputs_s[i_cluster][lt] = (
                            np_inputs_s[i_cluster][lt]
                            .astype(np.float32)
                            .transpose(1, 2, 0)
                            .reshape(
                                len(self.features)
                                * len(self.clusters["groups"][i_cluster]),
                                -1,
                            )
                            .transpose(1, 0)
                        )
                        np_inputs_s[i_cluster][lt] = np.repeat(
                            np_inputs_s[i_cluster][lt],
                            len(self.clusters["groups"][i_cluster]),
                            axis=0,
                        )
                        columns = [
                            f"{feature}_{station}"
                            for feature in self.features
                            for station in self.clusters["groups"][i_cluster]
                        ]
                        df_inputs = pd.DataFrame(
                            np_inputs_s[i_cluster][lt], columns=columns
                        )
                    df_inputs.to_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"{i_cluster}_{lt}_fold_{fold}_features.csv",
                        ),
                        index=False,
                    )
        # Creating the test files
        np_inputs_s = [
            dict(
                zip(self.filter["lead_times"], [None] * len(self.filter["lead_times"]))
            )
            for i in range(self.clusters["n"])
        ]
        np_labels = [None] * self.clusters["n"]
        for file, labels in zip(
            self.files["test"]["inputs"], self.files["test"]["labels"]
        ):
            inputs = xr.open_dataset(file, engine="netcdf4").sel(
                lead_time=self.filter["lead_times"]
            )
            loc_dates = pd.DatetimeIndex(inputs.time)
            if self.filter["storm_part"]["test"] is not None:
                seed, ratio = self.filter["storm_part"]["test"]
                rng = np.random.default_rng(seed)
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter["dates"])
                nostorm_dates = input_dates.difference(self.filter["dates"])
                if (ratio != 1) and len(storm_dates) >= len(nostorm_dates) * ratio / (
                    1 - ratio
                ):
                    storm_dates = storm_dates[
                        rng.permutation(len(storm_dates))[
                            : int(len(nostorm_dates) * ratio / (1 - ratio))
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[
                        rng.permutation(len(nostorm_dates))[
                            : int(len(storm_dates) * (1 - ratio) / ratio)
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
            # Inputs
            tmp_var = [
                dict(
                    zip(
                        self.filter["lead_times"],
                        [None] * len(self.filter["lead_times"]),
                    )
                )
                for i in range(self.clusters["n"])
            ]
            for i_var, var in enumerate(self.features):
                if len(var.split("_")) == 2:
                    var, var_level = var.split("_")
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(
                            inputs.sel(isobaricInhPa=var_level)["u"] ** 2
                            + inputs.sel(isobaricInhPa=var_level)["v"] ** 2
                        )
                    else:
                        tmp = inputs.sel(isobaricInhPa=var_level)[var]
                elif var == "wind":
                    tmp = np.sqrt(inputs["u10"] ** 2 + inputs["v10"] ** 2)
                elif var == "date":
                    tmp = inputs.time.dt.dayofyear.expand_dims(
                        {"station": inputs.station, "lead_time": inputs.lead_time},
                        axis=(-1, 0),
                    )
                elif var == "hour":
                    tmp = inputs.time.dt.hour.expand_dims(
                        {"station": inputs.station, "lead_time": inputs.lead_time},
                        axis=(-1, 0),
                    )
                else:
                    tmp = inputs[var]
                tmp = tmp.sel(time=loc_dates)
                for i_cluster in range(self.clusters["n"]):
                    for lt in self.filter["lead_times"]:
                        tmp_cluster = tmp.sel(
                            station=self.clusters["groups"][i_cluster], lead_time=lt
                        ).values
                        tmp_cluster = (
                            tmp_cluster - self.data["mean"][i_var]
                        ) / self.data["std"][i_var]
                        tmp_cluster = np.expand_dims(tmp_cluster, axis=-1)
                        tmp_var[i_cluster][lt] = (
                            tmp_cluster
                            if tmp_var[i_cluster][lt] is None
                            else np.concatenate(
                                [tmp_var[i_cluster][lt], tmp_cluster], axis=-1
                            )
                        )
            for i_cluster in range(self.clusters["n"]):
                for lt in self.filter["lead_times"]:
                    np_inputs_s[i_cluster][lt] = (
                        tmp_var[i_cluster][lt]
                        if np_inputs_s[i_cluster][lt] is None
                        else np.concatenate(
                            [np_inputs_s[i_cluster][lt], tmp_var[i_cluster][lt]], axis=0
                        )
                    )
            # Labels
            labels = xr.open_dataset(labels, engine="netcdf4")
            labels = labels.sel(time=loc_dates)[self.label]
            for i_cluster in range(self.clusters["n"]):
                lab_tmp = labels.sel(station=self.clusters["groups"][i_cluster]).values
                np_labels[i_cluster] = (
                    lab_tmp
                    if np_labels[i_cluster] is None
                    else np.concatenate([np_labels[i_cluster], lab_tmp], axis=0)
                )
        for i_cluster in range(self.clusters["n"]):
            np_labels[i_cluster] = (
                np_labels[i_cluster]
                .astype(np.float32)
                .reshape(np_labels[i_cluster].shape[0] * np_labels[i_cluster].shape[1])
            )
            df = pd.DataFrame(np_labels[i_cluster], columns=[self.label])
            df.to_csv(
                os.path.join(
                    self.folders["scratch"]["folder"], f"{i_cluster}_test_labels.csv"
                ),
                index=False,
            )
            for lt in self.filter["lead_times"]:
                if self.model_kwargs["data"] == "normal":
                    np_inputs_s[i_cluster][lt] = (
                        np_inputs_s[i_cluster][lt]
                        .astype(np.float32)
                        .reshape(-1, np_inputs_s[i_cluster][lt].shape[2])
                    )
                    df = pd.DataFrame(np_inputs_s[i_cluster][lt], columns=self.features)
                elif self.model_kwargs["data"] == "mean":
                    # For each cluster, for each feature, mean of the feature over the stations
                    np_inputs_s[i_cluster][lt] = (
                        np_inputs_s[i_cluster][lt]
                        .astype(np.float32)
                        .mean(axis=1, keepdims=True)
                    )
                    np_inputs_s[i_cluster][lt] = np.repeat(
                        np_inputs_s[i_cluster][lt],
                        len(self.clusters["groups"][i_cluster]),
                        axis=1,
                    ).reshape(-1, np_inputs_s[i_cluster][lt].shape[2])
                    df = pd.DataFrame(np_inputs_s[i_cluster][lt], columns=self.features)
                elif self.model_kwargs["data"] == "collective":
                    # For each time step, for each feature, concatenate the features of the stations
                    np_inputs_s[i_cluster][lt] = (
                        np_inputs_s[i_cluster][lt]
                        .astype(np.float32)
                        .transpose(1, 2, 0)
                        .reshape(
                            len(self.features) * len(self.clusters["groups"][i_cluster]),
                            -1,
                        )
                        .transpose(1, 0)
                    )
                    np_inputs_s[i_cluster][lt] = np.repeat(
                        np_inputs_s[i_cluster][lt],
                        len(self.clusters["groups"][i_cluster]),
                        axis=0,
                    )
                    columns = [
                        f"{feature}_{station}"
                        for feature in self.features
                        for station in self.clusters["groups"][i_cluster]
                    ]
                    df = pd.DataFrame(np_inputs_s[i_cluster][lt], columns=columns)
                df.to_csv(
                    os.path.join(
                        self.folders["scratch"]["folder"],
                        f"{i_cluster}_{lt}_test_features.csv",
                    ),
                    index=False,
                )
        print("Done.", flush=True)

    def run(self):
        """
        Run the VGAM / VGLM model.
        """
        # Create the inputs if needed
        fold_features_created = all(
            [
                os.path.exists(
                    os.path.join(
                        self.folders["scratch"]["folder"],
                        f"{icluster}_{lt}_fold_{fold}_features.csv",
                    )
                )
                for fold in range(self.model_kwargs["n_folds"])
                for lt in self.filter["lead_times"]
                for icluster in range(self.clusters["n"])
            ]
        )
        fold_labels_created = all(
            [
                os.path.exists(
                    os.path.join(
                        self.folders["scratch"]["folder"],
                        f"{icluster}_fold_{fold}_labels.csv",
                    )
                )
                for fold in range(self.model_kwargs["n_folds"])
                for icluster in range(self.clusters["n"])
            ]
        )
        test_features_created = all(
            [
                os.path.exists(
                    os.path.join(
                        self.folders["scratch"]["folder"],
                        f"{icluster}_{lt}_test_features.csv",
                    )
                )
                for lt in self.filter["lead_times"]
                for icluster in range(self.clusters["n"])
            ]
        )
        test_labels_created = all(
            [
                os.path.exists(
                    os.path.join(
                        self.folders["scratch"]["folder"], f"{icluster}_test_labels.csv"
                    )
                )
                for icluster in range(self.clusters["n"])
            ]
        )
        if not all(
            [
                fold_features_created,
                fold_labels_created,
                test_features_created,
                test_labels_created,
            ]
        ):
            self.load_mean_std()
            self.create_inputs()
        # Create a list of tuple containing the cluster and lead time
        args = [
            (cluster, lead_time, fold, self.model_kwargs["seqfeatsel"])
            for cluster in range(self.clusters["n"])
            for lead_time in self.filter["lead_times"]
            for fold in range(self.model_kwargs["n_folds"])
        ]
        with Pool() as p:
            p.map(self.run_single, args)
        # Computing CRPS
        preds_files = np.array(
            [
                [
                    [
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"{cluster}_{lead_time}_{fold}_test_preds.csv",
                        )
                        for fold in range(self.model_kwargs["n_folds"])
                    ]
                    for lead_time in self.filter["lead_times"]
                ]
                for cluster in range(self.clusters["n"])
            ]
        )
        if self.model_kwargs["seqfeatsel"]:
            labels_files = np.array(
                [
                    [
                        [
                            os.path.join(
                                self.folders["scratch"]["folder"],
                                f"{cluster}_{lead_time}_{fold}_test_labels.csv",
                            )
                            for fold in range(self.model_kwargs["n_folds"])
                        ]
                        for lead_time in self.filter["lead_times"]
                    ]
                    for cluster in range(self.clusters["n"])
                ]
            )
        else:
            labels_files = np.array(
                [
                    [
                        [
                            os.path.join(
                                self.folders["scratch"]["folder"],
                                f"{cluster}_test_labels.csv",
                            )
                            for fold in range(self.model_kwargs["n_folds"])
                        ]
                        for lead_time in self.filter["lead_times"]
                    ]
                    for cluster in range(self.clusters["n"])
                ]
            )

        def compute_crps(pred_file, label_file):
            preds = pd.read_csv(pred_file)
            labels = pd.read_csv(label_file)
            return gev_crps(
                preds.values[:, 0],
                preds.values[:, 1],
                preds.values[:, 2],
                labels.values[:, 0],
            ).mean()

        compute_crps = np.vectorize(compute_crps)
        crps = compute_crps(preds_files, labels_files)

        def compute_loglik(pred_file, label_file):
            preds = pd.read_csv(pred_file)
            labels = pd.read_csv(label_file)
            return -np.log(
                gev_pdf(
                    preds.values[:, 0],
                    preds.values[:, 1],
                    preds.values[:, 2],
                    labels.values[:, 0],
                )
            ).mean()

        compute_loglik = np.vectorize(compute_loglik)
        loglik = compute_loglik(preds_files, labels_files)
        with open(os.path.join(self.folders["plot"]["folder"], "CRPS.pkl"), "wb") as f:
            pickle.dump(crps, f)
        with open(
            os.path.join(self.folders["plot"]["folder"], "LogLik.pkl"), "wb"
        ) as f:
            pickle.dump(loglik, f)
        crps = crps.mean(axis=(0, 1))  # mean over clusters and lead times
        loglik = loglik.mean(axis=(0, 1))  # mean over clusters and lead times
        self.crps["mean"] = crps.mean()
        self.crps["std"] = crps.std()
        self.crps["values"] = crps
        self.loglik["mean"] = loglik.mean()
        self.loglik["std"] = loglik.std()
        self.loglik["values"] = loglik
        self.save.information()
        self.save.summary()
        self.save.experiment_file()

    def run_single(self, arg):
        """
        Run the VGAM / VGLM model for a single cluster, lead time and fold.
        """
        cluster, lead_time, fold, seq_feat_sel = arg
        # Check if already trained
        if os.path.exists(
            os.path.join(
                self.folders["plot"]["model"], f"{cluster}_{lead_time}_{fold}.rds"
            )
        ):
            # Only compute the predictions
            res = os.system(
                f"Rscript {self.files['R']['predict']} \
                            --test-predictors {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_test_features.csv')} \
                                --output {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv')} \
                                    --model-file {os.path.join(self.folders['plot']['model'], f'{cluster}_{lead_time}_{fold}.rds')} \
                                        --source {self.files['R']['source']}"
            )
        else:
            if not seq_feat_sel:
                # Testing will be performed on test set
                if not os.path.exists(
                    os.path.join(
                        self.folders["scratch"]["folder"],
                        f"{cluster}_{lead_time}_{fold}_test_preds.csv",
                    )
                ):
                    train_s = pd.concat(
                        [
                            pd.read_csv(
                                os.path.join(
                                    self.folders["scratch"]["folder"],
                                    f"{cluster}_{lead_time}_fold_{f}_features.csv",
                                )
                            )
                            for f in range(self.model_kwargs["n_folds"])
                            if f != fold
                        ]
                    )
                    train_l = pd.concat(
                        [
                            pd.read_csv(
                                os.path.join(
                                    self.folders["scratch"]["folder"],
                                    f"{cluster}_fold_{f}_labels.csv",
                                )
                            )
                            for f in range(self.model_kwargs["n_folds"])
                            if f != fold
                        ]
                    )
                    train_s.to_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"TS{cluster}_{lead_time}_{fold}.csv",
                        ),
                        index=False,
                    )
                    train_l.to_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"TL{cluster}_{lead_time}_{fold}.csv",
                        ),
                        index=False,
                    )
                    storms = pd.read_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"{cluster}_fold_{fold}_storm.csv",
                        )
                    )
                    res = os.system(
                        f"Rscript {self.files['R']['script']} \
                        --predictors {os.path.join(self.folders['scratch']['folder'], f'TS{cluster}_{lead_time}_{fold}.csv')} \
                            --response {os.path.join(self.folders['scratch']['folder'], f'TL{cluster}_{lead_time}_{fold}.csv')} \
                                --test-predictors {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_test_features.csv')} \
                                    --output {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv')} \
                                        --model-file {os.path.join(self.folders['plot']['model'], f'{cluster}_{lead_time}_{fold}.rds')} \
                                            --model {self.vgam_kwargs['model']} --source {self.files['R']['source']}"
                    )
                    os.remove(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"TS{cluster}_{lead_time}_{fold}.csv",
                        )
                    )
                    os.remove(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"TL{cluster}_{lead_time}_{fold}.csv",
                        )
                    )
                    return res
                else:
                    return 0
            else:
                # Testing will be performed on remaining fold
                if not os.path.exists(
                    os.path.join(
                        self.folders["scratch"]["folder"],
                        f"{cluster}_{lead_time}_{fold}_test_preds.csv",
                    )
                ):
                    train_s = pd.concat(
                        [
                            pd.read_csv(
                                os.path.join(
                                    self.folders["scratch"]["folder"],
                                    f"{cluster}_{lead_time}_fold_{f}_features.csv",
                                )
                            )
                            for f in range(self.model_kwargs["n_folds"])
                            if f != fold
                        ]
                    )
                    train_l = pd.concat(
                        [
                            pd.read_csv(
                                os.path.join(
                                    self.folders["scratch"]["folder"],
                                    f"{cluster}_fold_{f}_labels.csv",
                                )
                            )
                            for f in range(self.model_kwargs["n_folds"])
                            if f != fold
                        ]
                    )
                    train_s.to_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"TS{cluster}_{lead_time}_{fold}.csv",
                        ),
                        index=False,
                    )
                    train_l.to_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"TL{cluster}_{lead_time}_{fold}.csv",
                        ),
                        index=False,
                    )
                    storms = pd.read_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"{cluster}_fold_{fold}_storm.csv",
                        )
                    )
                    test_s = pd.read_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"{cluster}_{lead_time}_fold_{fold}_features.csv",
                        )
                    )
                    test_s = test_s[storms.values == 1]
                    test_s.to_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"{cluster}_{lead_time}_{fold}_test_features.csv",
                        ),
                        index=False,
                    )
                    test_l = pd.read_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"{cluster}_fold_{fold}_labels.csv",
                        )
                    )
                    test_l = test_l[storms.values == 1]
                    test_l.to_csv(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"{cluster}_{lead_time}_{fold}_test_labels.csv",
                        ),
                        index=False,
                    )
                    res = os.system(
                        f"Rscript {self.files['R']['script']} \
                        --predictors {os.path.join(self.folders['scratch']['folder'], f'TS{cluster}_{lead_time}_{fold}.csv')} \
                            --response {os.path.join(self.folders['scratch']['folder'], f'TL{cluster}_{lead_time}_{fold}.csv')} \
                                --test-predictors {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_features.csv')} \
                                    --output {os.path.join(self.folders['scratch']['folder'], f'{cluster}_{lead_time}_{fold}_test_preds.csv')} \
                                        --model-file {os.path.join(self.folders['plot']['model'], f'{cluster}_{lead_time}_{fold}.rds')} \
                                            --model {self.vgam_kwargs['model']} --source {self.files['R']['source']}"
                    )
                    os.remove(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"TS{cluster}_{lead_time}_{fold}.csv",
                        )
                    )
                    os.remove(
                        os.path.join(
                            self.folders["scratch"]["folder"],
                            f"TL{cluster}_{lead_time}_{fold}.csv",
                        )
                    )
                    return res
                else:
                    return 0

    def copy(self, **kwargs):
        """
        Copy the experiment with new parameters. Useful to create a new experiment
        based on an existing one with slight modifications.
        
        Parameters
        ----------
        kwargs : dict
            New parameters for the experiment. Keys are the attributes to modify
            and values are the new values. If target is a dictionary, the key
            is the attribute to modify and the value is a dictionary with the
            new values. It is possible to use as value a dictionary targetting
            only one subparameter: in this case, the other subparameters are unchanged.
            Example:
            exp.copy(model_kwargs={"epochs": 100, "batch_size": 32})
            # In this case, the learning rate is unchanged.
        """
        new_exp = RExperiment(self.files["experiment"])
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
        save_path = os.path.dirname(new_exp.files["experiment"])
        new_exp.exp_number = len(os.listdir(save_path))
        new_exp.folders["scratch"]["folder"] = os.path.join(
            new_exp.folders["scratch"]["dir"], f"Experiment_{new_exp.exp_number}"
        )
        new_exp.folders["plot"]["folder"] = os.path.join(
            new_exp.folders["plot"]["dir"], f"Experiment_{new_exp.exp_number}"
        )
        new_exp.folders["plot"]["model"] = os.path.join(
            new_exp.folders["plot"]["folder"], "models"
        )
        new_exp.files["experiment"] = (
            "_".join(self.files["experiment"].split("_")[:-1])
            + f"_{new_exp.exp_number}.pkl"
        )
        return new_exp

    def __str__(self):
        result = f"R Experiment n°{self.exp_number}\n\n"
        result += "Training files:\n"
        for file in self.files["train"]["inputs"]:
            result += f"{file}\n"

        result += "\nTest files:\n"
        for file in self.files["test"]["inputs"]:
            result += f"{file}\n"

        result += "\nTraining labels:\n"
        for file in self.files["train"]["labels"]:
            result += f"{file}\n"

        result += "\nTest labels:\n"
        for file in self.files["test"]["labels"]:
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

        result += f"\n\n--- CRPS: {self.crps['mean']} +/- {self.crps['std']} m/s ---"

        result += f"\n\n--- LogLik: {self.loglik['mean']} +/- {self.loglik['std']} ---"

        return result

    class Diagnostics:
        def __init__(self, experiment):
            """
            Class to compute diagnostics on the trained model.
            
            Parameters
            ----------
            experiment : Experiment
                The experiment to diagnose.
            """
            self.experiment = experiment

        def predict(
            self,
            inputs,
            label,
            features_filename,
            preds_filename,
            label_filename,
            lead_times=None,
            clusters=None,
            folds=None,
        ):
            """
            Uses the trained model to predict the labels of the inputs.
            
            Parameters
            ----------
            inputs : xr.Dataset
                The inputs to predict.
            label : str
                The label to predict.
            features_filename : str
                The filename where to save the features.
            preds_filename : str
                The filename where to save the predictions.
            label_filename : str
                The filename where to save the labels.
            lead_times : list of int or int, optional
                The lead times to consider. If None, all lead times are considered.
            clusters : list of int or int, optional
                The clusters to consider. If None, all clusters are considered.
            folds : list of int or int, optional
                The folds to consider. If None, all folds are considered.
            """
            lead_times = (
                self.experiment.filter["lead_times"][0]
                if lead_times is None
                else lead_times
            )
            if not isinstance(lead_times, list):
                lead_times = [lead_times]
            clusters = (
                list(range(self.experiment.clusters["n"]))
                if clusters is None
                else clusters
            )
            if not isinstance(clusters, list):
                clusters = [clusters]
            folds = (
                list(range(self.experiment.model_kwargs["n_folds"]))
                if folds is None
                else folds
            )
            if not isinstance(folds, list):
                folds = [folds]
            # Creating the input file
            np_inputs_s = [
                dict(
                    zip(
                        self.filter["lead_times"],
                        [None] * len(self.filter["lead_times"]),
                    )
                )
                for i in range(self.clusters["n"])
            ]
            np_labels = [None] * self.clusters["n"]
            for i_var, var in enumerate(self.features):
                if len(var.split("_")) == 2:
                    var, var_level = var.split("_")
                    var_level = int(var_level[:-3])
                    if var == "wind":
                        tmp = np.sqrt(
                            inputs.sel(isobaricInhPa=var_level)["u"] ** 2
                            + inputs.sel(isobaricInhPa=var_level)["v"] ** 2
                        )
                    else:
                        tmp = inputs.sel(isobaricInhPa=var_level)[var]
                elif var == "wind":
                    tmp = np.sqrt(inputs["u10"] ** 2 + inputs["v10"] ** 2)
                elif var == "date":
                    tmp = inputs.time.dt.dayofyear.expand_dims(
                        {"station": inputs.station, "lead_time": inputs.lead_time},
                        axis=(-1, 0),
                    )
                elif var == "hour":
                    tmp = inputs.time.dt.hour.expand_dims(
                        {"station": inputs.station, "lead_time": inputs.lead_time},
                        axis=(-1, 0),
                    )
                else:
                    tmp = inputs[var]
                for i_cluster in range(self.clusters["n"]):
                    for lt in self.filter["lead_times"]:
                        tmp_cluster = tmp.sel(
                            station=self.clusters["groups"][i_cluster], lead_time=lt
                        ).values
                        tmp_cluster = (
                            tmp_cluster - self.Data["mean"][i_var]
                        ) / self.Data["std"][i_var]
                        tmp_cluster = np.expand_dims(tmp_cluster, axis=-1)
                        np_inputs_s[i_cluster][lt] = (
                            tmp_cluster
                            if np_inputs_s[i_cluster][lt] is None
                            else np.concatenate(
                                [np_inputs_s[i_cluster][lt], tmp_cluster], axis=-1
                            )
                        )
            # Labels
            labels = xr.open_dataset(labels, engine="netcdf4")
            labels = labels[self.label]
            for i_cluster in range(self.clusters["n"]):
                lab_tmp = labels.sel(station=self.clusters["groups"][i_cluster]).values
                np_labels[i_cluster] = lab_tmp
            for i_cluster in range(self.clusters["n"]):
                np_labels[i_cluster] = (
                    np_labels[i_cluster]
                    .astype(np.float32)
                    .reshape(np_labels[i_cluster].shape[0] * np_labels[i_cluster].shape[1])
                )
                df = pd.DataFrame(np_labels[i_cluster], columns=[self.label])
                df.to_csv(label_filename + f"{i_cluster}.csv", index=False)
                for lt in self.filter["lead_times"]:
                    if self.model_kwargs["data"] == "normal":
                        np_inputs_s[i_cluster][lt] = (
                            np_inputs_s[i_cluster][lt]
                            .astype(np.float32)
                            .reshape(-1, np_inputs_s[i_cluster][lt].shape[2])
                        )
                        df = pd.DataFrame(
                            np_inputs_s[i_cluster][lt], columns=self.features
                        )
                    elif self.model_kwargs["data"] == "mean":
                        # For each cluster, for each feature, mean of the feature over the stations
                        np_inputs_s[i_cluster][lt] = (
                            np_inputs_s[i_cluster][lt]
                            .astype(np.float32)
                            .mean(axis=1, keepdims=True)
                        )
                        np_inputs_s[i_cluster][lt] = np.repeat(
                            np_inputs_s[i_cluster][lt],
                            len(self.clusters["groups"][i_cluster]),
                            axis=1,
                        ).reshape(-1, np_inputs_s[i_cluster][lt].shape[2])
                        df = pd.DataFrame(
                            np_inputs_s[i_cluster][lt], columns=self.features
                        )
                    elif self.model_kwargs["data"] == "collective":
                        # For each time step, for each feature, concatenate the features of the stations
                        np_inputs_s[i_cluster][lt] = (
                            np_inputs_s[i_cluster][lt]
                            .astype(np.float32)
                            .transpose(1, 2, 0)
                            .reshape(
                                len(self.features)
                                * len(self.clusters["groups"][i_cluster]),
                                -1,
                            )
                            .transpose(1, 0)
                        )
                        np_inputs_s[i_cluster][lt] = np.repeat(
                            np_inputs_s[i_cluster][lt],
                            len(self.clusters["groups"][i_cluster]),
                            axis=0,
                        )
                        columns = [
                            f"{feature}_{station}"
                            for feature in self.features
                            for station in self.clusters["groups"][i_cluster]
                        ]
                        df = pd.DataFrame(np_inputs_s[i_cluster][lt], columns=columns)
                    df.to_csv(features_filename + f"{i_cluster}_{lt}.csv", index=False)
            # Computing the predictions
            for i_cluster in clusters:
                for lt in lead_times:
                    for fold in folds:
                        os.system(
                            f"Rscript {self.files['R']['predict']} \
                            --test-predictors {features_filename + f'{i_cluster}_{lt}.csv'} \
                                --output {preds_filename + f'{i_cluster}_{lt}_{fold}.csv'} \
                                    --model-file {os.path.join(self.folders['plot']['model'], f'{i_cluster}_{lt}_{fold}.rds')} \
                                        --source {self.files['R']['source']}"
                        )

        def compute_crps(
            self,
            preds_filename,
            label_filename,
            lead_times=None,
            clusters=None,
            folds=None,
        ):
            """
            Uses the predictions and the labels to compute the CRPS.
            
            Parameters
            ----------
            preds_filename : str
                The filename where the predictions are stored (only the prefix).
            label_filename : str
                The filename where the labels are stored (only the prefix).
            lead_times : list of int or int, optional
                The lead times to consider. If None, all lead times are considered.
            clusters : list of int or int, optional
                The clusters to consider. If None, all clusters are considered.
            folds : list of int or int, optional
                The folds to consider. If None, all folds are considered.
            """
            lead_times = (
                self.experiment.filter["lead_times"][0]
                if lead_times is None
                else lead_times
            )
            if not isinstance(lead_times, list):
                lead_times = [lead_times]
            clusters = (
                list(range(self.experiment.clusters["n"]))
                if clusters is None
                else clusters
            )
            if not isinstance(clusters, list):
                clusters = [clusters]
            folds = (
                list(range(self.experiment.model_kwargs["n_folds"]))
                if folds is None
                else folds
            )
            if not isinstance(folds, list):
                folds = [folds]
            preds_files = np.array(
                [
                    [
                        [
                            preds_filename + f"{cluster}_{lead_time}_{fold}.csv"
                            for fold in folds
                        ]
                        for lead_time in lead_times
                    ]
                    for cluster in clusters
                ]
            )
            labels_files = np.array(
                [
                    [label_filename + f"{cluster}.csv" for cluster in clusters]
                    for lead_time in lead_times
                ]
            )

            def compute_crps(pred_file, label_file):
                preds = pd.read_csv(pred_file)
                labels = pd.read_csv(label_file)
                return gev_crps(
                    preds.values[:, 0],
                    preds.values[:, 1],
                    preds.values[:, 2],
                    labels.values[:, 0],
                ).mean()

            compute_crps = np.vectorize(compute_crps)
            crps = compute_crps(preds_files, labels_files)
            return crps

        def all_crps(self, save=False):
            """
            Compute the CRPS for all clusters, lead times and folds.
            First recreate the inputs (which won't be used to compute CRPS as
            predicitons were already made when the model was run). Same calculations
            as nn_loader.Experiment.create_inputs.
            
            Parameters
            ----------
            save : bool, optional
                If True, save the CRPS in a file.
            """
            # Creating the test files
            if not os.path.exists(
                os.path.join(
                    self.experiment.folders["scratch"]["folder"], "test_set.pkl"
                )
            ):
                np_inputs_t = None
                np_inputs_s = None
                np_labels = [None] * self.experiment.clusters["n"]
                for file, labels in zip(
                    self.experiment.files["test"]["inputs"],
                    self.experiment.files["test"]["labels"],
                ):
                    inputs = xr.open_dataset(file, engine="netcdf4").sel(
                        lead_time=self.experiment.filter["lead_times"]
                    )
                    loc_dates = pd.DatetimeIndex(inputs.time)
                    if self.experiment.filter["storm_part"]["test"] is not None:
                        seed, ratio = self.experiment.filter["storm_part"]["test"]
                        rng = np.random.default_rng(seed)
                        input_dates = pd.DatetimeIndex(inputs.time)
                        storm_dates = input_dates.intersection(
                            self.experiment.filter["dates"]
                        )
                        nostorm_dates = input_dates.difference(
                            self.experiment.filter["dates"]
                        )
                        if (ratio != 1) and len(storm_dates) >= len(
                            nostorm_dates
                        ) * ratio / (1 - ratio):
                            storm_dates = storm_dates[
                                rng.permutation(len(storm_dates))[
                                    : int(len(nostorm_dates) * ratio / (1 - ratio))
                                ]
                            ]
                            loc_dates = storm_dates.union(nostorm_dates)
                        else:
                            nostorm_dates = nostorm_dates[
                                rng.permutation(len(nostorm_dates))[
                                    : int(len(storm_dates) * (1 - ratio) / ratio)
                                ]
                            ]
                            loc_dates = storm_dates.union(nostorm_dates)
                        loc_dates = loc_dates.sort_values()
                    # Encoding of time
                    if self.experiment.model_kwargs["time_encoding"] == "sinusoidal":
                        tmp_t = np.array(
                            [
                                [
                                    [
                                        (time.day_of_year - 242)
                                        / 107,  # Encoding of day of year between -1 and +1
                                        np.cos(time.hour * np.pi / 12),
                                        np.sin(time.hour * np.pi / 12),
                                        lt / 72,
                                    ]
                                    for time in loc_dates
                                ]
                                for lt in inputs.lead_time
                            ],
                            dtype=np.float32,
                        ).reshape(-1, 4)
                    elif self.experiment.model_kwargs["time_encoding"] == "rbf":
                        pass
                    np_inputs_t = (
                        tmp_t
                        if np_inputs_t is None
                        else np.concatenate([np_inputs_t, tmp_t], axis=0)
                    )
                    # Inputs
                    tmp_var = None
                    for ivar in range(len(self.experiment.features)):
                        # First obtaining corresponding data array
                        var = self.experiment.features[ivar]
                        if len(var.split("_")) == 2:
                            var, var_level = var.split("_")
                            var_level = int(var_level[:-3])
                            if var == "wind":
                                tmp = np.sqrt(
                                    inputs.sel(isobaricInhPa=var_level)["u"] ** 2
                                    + inputs.sel(isobaricInhPa=var_level)["v"] ** 2
                                )
                            else:
                                tmp = inputs.sel(isobaricInhPa=var_level)[var]
                        elif var == "wind":
                            tmp = np.sqrt(inputs["u10"] ** 2 + inputs["v10"] ** 2)
                        elif var == "CAPE":
                            tmp = inputs["CAPE"].fillna(0.0)
                        else:
                            tmp = inputs[var]
                        tmp = tmp.sel(time=loc_dates)
                        tmp = tmp.values
                        sh = tmp.shape
                        tmp = tmp.reshape((sh[0] * sh[1], sh[2]))
                        tmp = (
                            tmp - self.experiment.Data["mean"][ivar]
                        ) / self.experiment.Data["std"][ivar]
                        tmp = np.expand_dims(tmp, axis=-1)
                        tmp_var = (
                            tmp
                            if tmp_var is None
                            else np.concatenate([tmp_var, tmp], axis=-1)
                        )
                    np_inputs_s = (
                        tmp_var
                        if np_inputs_s is None
                        else np.concatenate([np_inputs_s, tmp_var], axis=0)
                    )
                    # Labels
                    labels = xr.open_dataset(labels, engine="netcdf4")
                    labels = labels.sel(time=loc_dates)[self.experiment.label]
                    for i in range(self.experiment.clusters["n"]):
                        lab_tmp = labels.sel(
                            station=self.experiment.clusters["groups"][i]
                        ).values
                        lab_tmp = np.repeat(
                            np.expand_dims(lab_tmp, axis=0),
                            len(inputs.lead_time),
                            axis=0,
                        )
                        lab_tmp = lab_tmp.reshape(
                            lab_tmp.shape[0] * lab_tmp.shape[1], lab_tmp.shape[2]
                        )
                        np_labels[i] = (
                            lab_tmp
                            if np_labels[i] is None
                            else np.concatenate([np_labels[i], lab_tmp], axis=0)
                        )
                with open(
                    os.path.join(
                        self.experiment.folders["scratch"]["folder"], "test_set.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump((np_inputs_s, np_inputs_t, np_labels), f)
            # Get the test set
            with open(
                os.path.join(
                    self.experiment.folders["scratch"]["folder"], f"test_set.pkl"
                ),
                "rb",
            ) as f:
                s, t, l = pickle.load(f)
            crps = np.zeros(
                (
                    self.experiment.model_kwargs["n_folds"],
                    self.experiment.clusters["n"],
                    t.shape[0],
                )
            )
            for i_fold, fold in enumerate(
                range(self.experiment.model_kwargs["n_folds"])
            ):
                for i_cluster, cluster in enumerate(
                    range(self.experiment.clusters["n"])
                ):
                    for i_lt, lt in enumerate(self.experiment.filter["lead_times"]):
                        preds = pd.read_csv(
                            os.path.join(
                                self.experiment.folders["scratch"]["folder"],
                                f"{cluster}_{lt}_{fold}_test_preds.csv",
                            )
                        )
                        labels = pd.read_csv(
                            os.path.join(
                                self.experiment.folders["scratch"]["folder"],
                                f"{cluster}_test_labels.csv",
                            )
                        )
                        crps = gev_crps(
                            preds.values[:, 0],
                            preds.values[:, 1],
                            preds.values[:, 2],
                            labels.values[:, 0],
                        )
                        # Reshaping crps to account for the different stations
                        crps = crps.reshape(
                            -1, len(self.experiment.clusters["groups"][cluster])
                        ).mean(axis=1)
                        crps[i_fold, i_cluster, t[:, 3] == lt / 72] = crps
            if save:
                with open(
                    os.path.join(
                        self.experiment.folders["plot"]["folder"], "allCRPSs.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(crps, f)
            return s, t, l, crps

    # Plotting
    class Plotter:
        def __init__(self, experiment):
            """
            Class to plot the results of the experiment.
            
            Parameters
            ----------
            experiment : Experiment
                The experiment to plot.
            """
            self.experiment = experiment

        def lt_crps(self, keep=False):
            """
            Plot the CRPS for each lead time.
            
            Parameters
            ----------
            keep : bool, optional
                If True, keep the plot in the plotting directory.
            """
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "CRPS.pkl"),
                "rb",
            ) as f:
                crps = pickle.load(f)
            crps = crps.mean(axis=0).T
            metrics_evolution(
                {"CRPS": crps},
                self.experiment.filter["lead_times"],
                os.path.join(
                    self.experiment.folders["plot"]["folder"], "CRPS_leadtime.png"
                ),
            )
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "ltCRPS.pkl"),
                "wb",
            ) as f:
                pickle.dump(crps, f)

        def cluster_crps(self, cluster, date):
            """
            Plot the CRPS for a specific cluster and date.
            
            Parameters
            ----------
            cluster : int
                The cluster to consider.
            date : datetime
                The date to consider.
            """
            strdate = date.strftime("%Y%m%d%H%M")
            # Labels
            labels = xr.open_dataset(
                self.experiment.files["test"]["labels"][0], engine="netcdf4"
            )
            labels = labels.sel(time=date)[self.experiment.label]
            np_inputs_l = [None] * self.experiment.clusters["n"]
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
            mus = [
                [
                    pd.read_csv(
                        os.path.join(
                            self.experiment.folders["scratch"]["folder"],
                            f"CASESTUDY_{cluster}_{fold}_{strdate}_{lt}_preds.csv",
                        )
                    ).values[:, 0]
                    for fold in range(self.experiment.model_kwargs["n_folds"])
                ]
                for lt in self.experiment.filter["lead_times"]
            ]
            sigmas = [
                [
                    pd.read_csv(
                        os.path.join(
                            self.experiment.folders["scratch"]["folder"],
                            f"CASESTUDY_{cluster}_{fold}_{strdate}_{lt}_preds.csv",
                        )
                    ).values[:, 1]
                    for fold in range(self.experiment.model_kwargs["n_folds"])
                ]
                for lt in self.experiment.filter["lead_times"]
            ]
            xis = [
                [
                    pd.read_csv(
                        os.path.join(
                            self.experiment.folders["scratch"]["folder"],
                            f"CASESTUDY_{cluster}_{fold}_{strdate}_{lt}_preds.csv",
                        )
                    ).values[:, 2]
                    for fold in range(self.experiment.model_kwargs["n_folds"])
                ]
                for lt in self.experiment.filter["lead_times"]
            ]
            all_crps_s = np.array(
                [
                    [
                        gev_crps(
                            mus[ilt][ifold][cluster],
                            sigmas[ilt][ifold][cluster],
                            xis[ilt][ifold][cluster],
                            np_inputs_l[cluster],
                        ).mean()
                        for ifold in range(self.experiment.model_kwargs["n_folds"])
                    ]
                    for ilt, lt in enumerate(self.experiment.filter["lead_times"])
                ]
            ).T
            print(all_crps_s)
            metrics_evolution(
                {"CRPS": all_crps_s},
                self.experiment.filter["lead_times"],
                os.path.join(
                    self.experiment.folders["plot"]["folder"],
                    f"CRPS_{cluster}_{date}.png",
                ),
            )
            return all_crps_s

    class Saver:
        def __init__(self, experiment):
            """
            Class to save the results of the experiment.
            
            Parameters
            ----------
            experiment : Experiment
                The experiment to save.
            """
            self.experiment = experiment

        def information(self):
            """
            Save the information of the experiment in a file.
            """
            with open(
                os.path.join(
                    self.experiment.folders["plot"]["folder"], "Information.txt"
                ),
                "w",
            ) as f:
                f.write(str(self.experiment))

        def summary(self):
            pass

        def experiment_file(self):
            """
            Save the experiment in a file.
            """
            experiment_dict = {
                "files": self.experiment.files,
                "folders": self.experiment.folders,
                "features": self.experiment.features,
                "label": self.experiment.label,
                "vgam_kwargs": self.experiment.vgam_kwargs,
                "model_kwargs": self.experiment.model_kwargs,
                "filter": {
                    k: self.experiment.filter[k] for k in ["lead_times", "storm_part"]
                },
                "CRPS": self.experiment.CRPS,
                "LogLik": self.experiment.LogLik,
                "Data": self.experiment.Data,
            }
            with open(self.experiment.files["experiment"], "wb") as f:
                pickle.dump(experiment_dict, f)
