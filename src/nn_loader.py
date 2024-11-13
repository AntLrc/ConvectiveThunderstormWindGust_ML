import xarray as xr
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
from flax import linen as nn

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pickle

import os

from nn_block import *
from nn_train import train_loop, create_train_state, create_batches, loss_and_crps, mae
from nn_losses import gev_crps_loss, gev_crps, return_level, return_level_loss
from utils import pit_histogram, storm_plot, metrics_evolution, param_distribution


class Experiment:

    def __init__(self, experiment_file):
        """
        Class to load the data and preprocess it for the NN models.
        
        Parameters
        ----------
        experiment_file : str
            Path to the experiment file.
        """
        with open(experiment_file, "rb") as f:
            experiment = pickle.load(f)
        self.expnumber = experiment_file.split("_")[-1].split(".")[0]
        self.files = experiment["files"]
        default_files = {
            "storms": None,
            "clusters": None,
            "experiment": experiment_file,
            "train": {"inputs": [], "labels": None},
            "test": {"inputs": [], "labels": None},
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
            "plot": {"folder": None, "dir": None},
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
        self.nn_kwargs = experiment.get("nn_kwargs", {})
        self.model_kwargs = experiment.get("model_kwargs", {})
        default_model_kwargs = {
            "batch_size": 64,
            "rng": {"init": jax.random.key(0), "shuffle": jax.random.key(1)},
            "epochs": 50,
            "learning_rate": 1e-4,
            "regularisation": None,
            "alpha": 0.0,
            "n_best_states": 3,
            "target": "GEV",
            "time_encoding": "sinusoidal",
            "n_folds": len(self.files["train"]["inputs"]),
            "early_stopping": 6,
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
            "2d": True,
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
        self.data = experiment.get("Data", {})
        default_data = {"mean": None, "std": None, "ndates": None}
        for k, v in default_data.items():
            if not k in self.data.keys():
                self.data[k] = v
        os.makedirs(self.folders["scratch"]["folder"], exist_ok=True)
        os.makedirs(self.folders["plot"]["folder"], exist_ok=True)
        self.nn = self.nn_()
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
        ndates = np.zeros((len(self.files["train"]["inputs"])))

        for i, file in enumerate(self.files["train"]["inputs"]):
            print(file)
            inputs = xr.open_dataset(file, engine="netcdf4").sel(
                lead_time=self.filter["lead_times"]
            )
            loc_dates = pd.DatetimeIndex(inputs.time)
            if self.filter["storm_part"]["train"] is not None:
                rng, ratio = self.filter["storm_part"]["train"]
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter["dates"])
                nostorm_dates = input_dates.difference(self.filter["dates"])
                if (ratio == 1) or len(storm_dates) >= len(nostorm_dates) * ratio / (
                    1 - ratio
                ):
                    storm_dates = storm_dates[
                        jax.random.permutation(rng, len(storm_dates))[
                            : int(
                                len(nostorm_dates) * ratio / (1 - ratio)
                                if ratio != 1.0
                                else -1
                            )
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[
                        jax.random.permutation(rng, len(nostorm_dates))[
                            : int(
                                len(storm_dates) * (1 - ratio) / ratio
                                if ratio != 0.0
                                else -1
                            )
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
                ndates[i] = len(loc_dates)
            for ivar in range(len(self.features)):
                var = self.features[ivar]
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
                count[ivar] += len(tmp)
                mean_[ivar] += tmp.mean() * len(tmp)
                mean_sq[ivar] += (tmp**2).mean() * len(tmp)
            inputs.close()
        self.data["mean"] = mean_ / count
        self.data["std"] = np.sqrt(mean_sq / count - self.data["mean"] ** 2)
        self.data["ndates"] = ndates
        self.save.experimentfile()
        print("Done.", flush=True)

    def create_inputs(self):
        """
        Create the input arrays for the CNN model, following the k-fold cross validation scheme.
        self.files['train']['inputs'] and self.files['train']['labels'] should be lists of string,
        each string being the file of one year (which will be used as a single fold).
        self.files['val'] should no longer exist (or at least, be useless).
        """
        print("Creating inputs...", flush=True)

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
                rng, ratio = self.filter["storm_part"]["train"]
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter["dates"])
                nostorm_dates = input_dates.difference(self.filter["dates"])
                if (ratio != 1) and len(storm_dates) >= len(nostorm_dates) * ratio / (
                    1 - ratio
                ):
                    storm_dates = storm_dates[
                        jax.random.permutation(rng, len(storm_dates))[
                            : int(
                                len(nostorm_dates) * ratio / (1 - ratio)
                                if ratio != 1.0
                                else -1
                            )
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[
                        jax.random.permutation(rng, len(nostorm_dates))[
                            : int(
                                len(storm_dates) * (1 - ratio) / ratio
                                if ratio != 0.0
                                else -1
                            )
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
                storm_presence = np.zeros(len(loc_dates))
                storm_presence[loc_dates.isin(storm_dates)] = 1
            # Encoding of time
            if self.model_kwargs["time_encoding"] == "sinusoidal":
                npinputs_t = np.array(
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
                npstorms = np.repeat(
                    np.expand_dims(storm_presence, axis=0),
                    len(inputs.lead_time),
                    axis=0,
                ).reshape(-1)
            elif self.model_kwargs["time_encoding"] == "rbf":
                pass
            # Inputs
            npinputs_s = None
            for ivar in range(len(self.features)):
                # First obtaining corresponding data array
                var = self.features[ivar]
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
                tmp = tmp.values  # Converting to numpy array
                sh = tmp.shape
                if self.filter["2d"]:
                    tmp = tmp.reshape(
                        (sh[0] * sh[1], sh[2], sh[3])
                    )  # Shape of (lead_time*time, lat, lon)
                else:
                    tmp = tmp.reshape(
                        (sh[0] * sh[1], sh[2])
                    )  # Shape of (lead_time*time, station)
                tmp = (tmp - self.data["mean"][ivar]) / self.data["std"][
                    ivar
                ]  # Normalizing
                tmp = np.expand_dims(tmp, axis=-1)
                npinputs_s = (
                    tmp
                    if npinputs_s is None
                    else np.concatenate([npinputs_s, tmp], axis=-1)
                )
            # Labels
            labels = xr.open_dataset(labels, engine="netcdf4")
            labels = labels.sel(time=loc_dates)[self.label]
            nplabels = [None] * self.clusters["n"]
            for i in range(self.clusters["n"]):
                labtmp = labels.sel(station=self.clusters["groups"][i]).values
                labtmp = np.repeat(
                    np.expand_dims(labtmp, axis=0), len(inputs.lead_time), axis=0
                )
                labtmp = labtmp.reshape(
                    labtmp.shape[0] * labtmp.shape[1], labtmp.shape[2]
                )
                nplabels[i] = labtmp
            # Saving
            with open(
                os.path.join(self.folders["scratch"]["folder"], f"fold_{fold}.pkl"),
                "wb",
            ) as f:
                pickle.dump((npinputs_s, npinputs_t, nplabels), f)
            with open(
                os.path.join(self.folders["scratch"]["folder"], f"storms_{fold}.pkl"),
                "wb",
            ) as f:
                pickle.dump(npstorms, f)
        # Creating the test files
        npinputs_t = None
        npinputs_s = None
        nplabels = [None] * self.clusters["n"]
        for file, labels in zip(
            self.files["test"]["inputs"], self.files["test"]["labels"]
        ):
            inputs = xr.open_dataset(file, engine="netcdf4").sel(
                lead_time=self.filter["lead_times"]
            )
            loc_dates = pd.DatetimeIndex(inputs.time)
            if self.filter["storm_part"]["test"] is not None:
                rng, ratio = self.filter["storm_part"]["test"]
                input_dates = pd.DatetimeIndex(inputs.time)
                storm_dates = input_dates.intersection(self.filter["dates"])
                nostorm_dates = input_dates.difference(self.filter["dates"])
                if (ratio != 1) and len(storm_dates) >= len(nostorm_dates) * ratio / (
                    1 - ratio
                ):
                    storm_dates = storm_dates[
                        jax.random.permutation(rng, len(storm_dates))[
                            : int(len(nostorm_dates) * ratio / (1 - ratio))
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                else:
                    nostorm_dates = nostorm_dates[
                        jax.random.permutation(rng, len(nostorm_dates))[
                            : int(len(storm_dates) * (1 - ratio) / ratio)
                        ]
                    ]
                    loc_dates = storm_dates.union(nostorm_dates)
                loc_dates = loc_dates.sort_values()
            # Encoding of time
            if self.model_kwargs["time_encoding"] == "sinusoidal":
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
            elif self.model_kwargs["time_encoding"] == "rbf":
                pass
            npinputs_t = (
                tmp_t
                if npinputs_t is None
                else np.concatenate([npinputs_t, tmp_t], axis=0)
            )
            # Inputs
            tmp_var = None
            for ivar in range(len(self.features)):
                # First obtaining corresponding data array
                var = self.features[ivar]
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
                if self.filter["2d"]:
                    tmp = tmp.reshape((sh[0] * sh[1], sh[2], sh[3]))
                else:
                    tmp = tmp.reshape((sh[0] * sh[1], sh[2]))
                tmp = (tmp - self.data["mean"][ivar]) / self.data["std"][ivar]
                tmp = np.expand_dims(tmp, axis=-1)
                tmp_var = (
                    tmp if tmp_var is None else np.concatenate([tmp_var, tmp], axis=-1)
                )
            npinputs_s = (
                tmp_var
                if npinputs_s is None
                else np.concatenate([npinputs_s, tmp_var], axis=0)
            )
            # Labels
            labels = xr.open_dataset(labels, engine="netcdf4")
            labels = labels.sel(time=loc_dates)[self.label]
            for i in range(self.clusters["n"]):
                labtmp = labels.sel(station=self.clusters["groups"][i]).values
                labtmp = np.repeat(
                    np.expand_dims(labtmp, axis=0), len(inputs.lead_time), axis=0
                )
                labtmp = labtmp.reshape(
                    labtmp.shape[0] * labtmp.shape[1], labtmp.shape[2]
                )
                nplabels[i] = (
                    labtmp
                    if nplabels[i] is None
                    else np.concatenate([nplabels[i], labtmp], axis=0)
                )
        print(
            f"Test set:\n - shape of spatial_inputs: {npinputs_s.shape}\n - shape of temporal_inputs: {npinputs_t.shape}\n - shape of labels: {nplabels[0].shape}"
        )
        with open(
            os.path.join(self.folders["scratch"]["folder"], "test_set.pkl"), "wb"
        ) as f:
            pickle.dump((npinputs_s, npinputs_t, nplabels), f)
        print("Done.", flush=True)

    def load_data(self):
        print(f"Loading data from {self.model_kwargs['n_folds']} folds...", flush=True)
        self.train_s = []
        self.train_t = []
        self.train_l = []
        self.train_storms = []
        for ifold in range(self.model_kwargs["n_folds"]):
            with open(
                os.path.join(self.folders["scratch"]["folder"], f"fold_{ifold}.pkl"),
                "rb",
            ) as f:
                s, t, l = pickle.load(f)
            with open(
                os.path.join(self.folders["scratch"]["folder"], f"storms_{ifold}.pkl"),
                "rb",
            ) as f:
                storms = pickle.load(f)
            self.train_s.append(s)
            self.train_t.append(t)
            self.train_l.append(l)
            self.train_storms.append(storms)
        with open(
            os.path.join(self.folders["scratch"]["folder"], "test_set.pkl"), "rb"
        ) as f:
            self.test_s, self.test_t, self.test_l = pickle.load(f)

    def nn_(self):
        nns = {
            "Dense_NN": DenseNN,
            "DeepFCN": DeepFCN,
            "Identity_NN": IdentityNN,
            "Killed_NN": KilledNN,
            "Conv_NN": ConvNN,
            "ConvNeXt_Block": ConvNeXtBlock,
            "ConvNeXt_Blocks_NN": ConvNeXtBlocksNN,
            "ConvNeXt_NN": ConvNeXtNN,
            "AttentionBlock": AttentionBlock,
            "VisionTransformer": VisionTransformer,
            "AddTrainableXi": AddTrainableXi,
            "SimpleBaseline": SimpleBaseline,
            "DDNN_GEV": GevDDNN,
            "Discrete_NN": DiscreteNN,
        }
        if not self.nn_kwargs["spatial"] is None:
            spatial_nn = nns[self.nn_kwargs["spatial"]["name"]](
                **self.nn_kwargs["spatial"]["kwargs"]
            )
        if not self.nn_kwargs["temporal"] is None:
            temporal_nn = nns[self.nn_kwargs["temporal"]["name"]](
                **self.nn_kwargs["temporal"]["kwargs"]
            )
        if not self.nn_kwargs["distributional"] is None:
            dd_nn = nns[self.nn_kwargs["distributional"]["name"]](
                **self.nn_kwargs["distributional"]["kwargs"]
            )
        return AlpThNN(
            n_clusters=self.clusters["n"],
            spatial_nn=spatial_nn,
            temporal_nn=temporal_nn,
            dd_nn=dd_nn,
        )

    def run(self, pre_computed=False, fold=None):
        # Create data if needed
        if not (
            all(
                [
                    os.path.exists(
                        os.path.join(self.folders["scratch"]["folder"], f"fold_{i}.pkl")
                    )
                    for i in range(self.model_kwargs["n_folds"])
                ]
            )
            and os.path.exists(
                os.path.join(self.folders["scratch"]["folder"], "test_set.pkl")
            )
        ):
            print("Inputs do not exist. Creating them...", flush=True)
            self.load_mean_std()
            self.create_inputs()
        # Load data
        self.load_data()
        # Visualise data
        if not all(
            [
                os.path.exists(
                    os.path.join(
                        self.folders["plot"]["folder"], file + "Distribution.png"
                    )
                )
                for file in ["Labels", "Features"]
            ]
        ):
            print("Visualising data...", flush=True)
            pass
        # Train self.model_kwargs['n_folds'] models
        if not pre_computed:
            if not os.path.exists(
                os.path.join(self.folders["plot"]["folder"], "params.pkl")
            ):
                params = [None] * self.model_kwargs["n_folds"]
            else:
                with open(
                    os.path.join(self.folders["plot"]["folder"], "params.pkl"), "rb"
                ) as f:
                    params = pickle.load(f)
            if not os.path.exists(
                os.path.join(self.folders["plot"]["folder"], "losses.pkl")
            ):
                losses = [None] * self.model_kwargs["n_folds"]
            else:
                with open(
                    os.path.join(self.folders["plot"]["folder"], "losses.pkl"), "rb"
                ) as f:
                    losses = pickle.load(f)
            if not os.path.exists(
                os.path.join(self.folders["plot"]["folder"], "metrics.pkl")
            ):
                metrics = [None] * self.model_kwargs["n_folds"]
            else:
                with open(
                    os.path.join(self.folders["plot"]["folder"], "metrics.pkl"), "rb"
                ) as f:
                    metrics = pickle.load(f)
            for ifold in range(self.model_kwargs["n_folds"]):
                if fold is not None and ifold != fold:
                    continue
                print(
                    f"Training model with fold {ifold} as validation set...", flush=True
                )
                model_state = create_train_state(
                    self.nn,
                    self.model_kwargs["rng"]["init"],
                    self.model_kwargs["learning_rate"],
                    self.model_kwargs["batch_size"],
                    len(self.features),
                    stationwise=not self.filter["2d"],
                    n_stations=len(self.clusters["stations"]),
                )
                # Get data
                train_s = jnp.array(
                    np.concatenate(
                        [
                            self.train_s[i]
                            for i in range(self.model_kwargs["n_folds"])
                            if i != ifold
                        ],
                        axis=0,
                    )
                )
                train_t = jnp.array(
                    np.concatenate(
                        [
                            self.train_t[i]
                            for i in range(self.model_kwargs["n_folds"])
                            if i != ifold
                        ],
                        axis=0,
                    )
                )
                train_l = tuple(
                    [
                        jnp.array(
                            np.concatenate(
                                [
                                    self.train_l[i][j]
                                    for i in range(self.model_kwargs["n_folds"])
                                    if i != ifold
                                ],
                                axis=0,
                            )
                        )
                        for j in range(self.clusters["n"])
                    ]
                )
                val_s = jnp.array(self.train_s[ifold][self.train_storms[ifold] == 1])
                val_t = jnp.array(self.train_t[ifold][self.train_storms[ifold] == 1])
                val_l = tuple(
                    [
                        jnp.array(self.train_l[ifold][j][self.train_storms[ifold] == 1])
                        for j in range(self.clusters["n"])
                    ]
                )
                # Print shapes
                print(
                    f"Train set shapes:\n - shape of spatial inputs: {train_s.shape}\n - shape of temporal inputs: {train_t.shape}\n - shape of labels: {train_l[0].shape}"
                )
                print(
                    f"Validation set shapes:\n - shape of spatial inputs: {val_s.shape}\n - shape of temporal inputs: {val_t.shape}\n - shape of labels: {val_l[0].shape}"
                )
                # Train model
                best_params_with_scores, train_loss, val_loss, train_crps, val_crps = (
                    train_loop(
                        model_state,
                        train_s,
                        train_t,
                        train_l,
                        val_s,
                        val_t,
                        val_l,
                        self.model_kwargs["batch_size"],
                        self.model_kwargs["epochs"],
                        len(self.clusters["stations"]),
                        self.clusters["n"],
                        self.model_kwargs["rng"]["shuffle"],
                        regularisation=self.model_kwargs["regularisation"],
                        alpha=self.model_kwargs["alpha"],
                        n_best_states=self.model_kwargs["n_best_states"],
                        target=self.target,
                        early_stopping=self.model_kwargs["early_stopping"],
                    )
                )
                output_params = jax.tree.map(
                    lambda *x: jnp.stack(x).mean(axis=0),
                    *map(lambda x: x[0], best_params_with_scores),
                )
                params[ifold] = output_params
                losses[ifold] = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_CRPS": train_crps,
                    "val_CRPS": val_crps,
                }
                output_state = model_state.replace(params=output_params)
                # Evaluate model
                output_crps = 0
                count = 0
                metrics_fn = (
                    loss_and_crps if self.model_kwargs["target"] == "GEV" else mae
                )
                for x_s, x_t, y_true in create_batches(
                    val_s,
                    val_t,
                    val_l,
                    self.model_kwargs["batch_size"],
                    self.model_kwargs["rng"]["shuffle"],
                ):
                    tmp_crps, _ = metrics_fn(
                        output_state,
                        output_state.params,
                        (x_s, x_t, y_true),
                        self.model_kwargs["batch_size"],
                        len(self.clusters["stations"]),
                        self.clusters["n"],
                        self.model_kwargs["regularisation"],
                        self.model_kwargs["alpha"],
                    )
                    output_crps += tmp_crps
                    count += 1
                metrics[ifold] = output_crps / count
                # Save results at each fold
                with open(
                    os.path.join(self.folders["plot"]["folder"], "params.pkl"), "wb"
                ) as f:
                    pickle.dump(params, f)
                with open(
                    os.path.join(self.folders["plot"]["folder"], "losses.pkl"), "wb"
                ) as f:
                    pickle.dump(losses, f)
                with open(
                    os.path.join(self.folders["plot"]["folder"], "metrics.pkl"), "wb"
                ) as f:
                    pickle.dump(metrics, f)
        else:
            with open(
                os.path.join(self.folders["plot"]["folder"], "params.pkl"), "rb"
            ) as f:
                params = pickle.load(f)
            with open(
                os.path.join(self.folders["plot"]["folder"], "losses.pkl"), "rb"
            ) as f:
                losses = pickle.load(f)
            with open(
                os.path.join(self.folders["plot"]["folder"], "metrics.pkl"), "rb"
            ) as f:
                metrics = pickle.load(f)
        # Save results
        if all([metrics[i] is not None for i in range(self.model_kwargs["n_folds"])]):
            self.crps["mean"] = np.mean(metrics)
            self.crps["std"] = np.std(metrics)
            self.crps["values"] = np.array(metrics)
        self.save.information()
        self.save.experimentfile()
        self.save.summary()
        # Plot results
        self.plot.losses(losses)

    def copy(self, **kwargs):
        """
        Copy the experiment with new parameters.
        """
        new_exp = Experiment(self.files["experiment"])
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
        new_exp.expnumber = len(os.listdir(save_path))
        new_exp.folders["scratch"]["folder"] = os.path.join(
            new_exp.folders["scratch"]["dir"], f"Experiment_{new_exp.expnumber}"
        )
        new_exp.folders["plot"]["folder"] = os.path.join(
            new_exp.folders["plot"]["dir"], f"Experiment_{new_exp.expnumber}"
        )
        new_exp.files["experiment"] = (
            "_".join(self.files["experiment"].split("_")[:-1])
            + f"_{new_exp.expnumber}.pkl"
        )
        new_exp.crps = {"mean": None, "std": None, "values": None}
        return new_exp

    def __str__(self):
        result = f"Experiment n°{self.expnumber}\n\n"
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

        result += f"\nLead times: {self.filter['lead_times']}\n"

        result += f"\nPart of storm in dataset: {self.filter['storm_part']}\n"

        result += f"\nNumber of epochs: {self.model_kwargs['epochs']}"
        result += f"\nBatch size: {self.model_kwargs['batch_size']}"
        result += f"\nLearning rate: {self.model_kwargs['learning_rate']}"
        result += f"\nRegularisation: {self.model_kwargs['regularisation']}"
        if self.model_kwargs["regularisation"] is not None:
            result += f", alpha = {self.model_kwargs['alpha']}"
        result += f"\nNumber of states kept to create output state: {self.model_kwargs['n_best_states']}"
        result += f"\nTarget: {self.model_kwargs['target']}"
        erls = (
            "After " + str(self.model_kwargs["early_stopping"]) + " epochs"
            if self.model_kwargs["early_stopping"] != 0
            else "No early stopping"
        )
        result += f"\nEarly stopping: {erls}"

        result += f"\nModel:\n{self.nn}\n"

        result += f"\n\n--- CRPS: {self.crps['mean']} +/- {self.crps['std']} m/s ---"

        return result

    # Subclass for diagnostics
    class Diagnostics:
        def __init__(self, experiment):
            self.experiment = experiment

        def all_crps(self, save=False):
            """
            Computes CRPS for each cluster, each time step, each fold, each lead time to allow more precise diagnostics.
            """
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
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl"),
                "rb",
            ) as f:
                paramlist = pickle.load(f)
            gevparams = None
            for i, params in enumerate(paramlist):
                # Compute outputs
                for ibatch in range(len(t) // 100 + 1):
                    sb = s[100 * ibatch : 100 * (ibatch + 1)]
                    tb = t[100 * ibatch : 100 * (ibatch + 1)]
                    lb = tuple(
                        [
                            l[j][100 * ibatch : 100 * (ibatch + 1)]
                            for j in range(self.experiment.clusters["n"])
                        ]
                    )
                    if len(sb) == 0:
                        continue
                    gevparams = self.experiment.NN.apply(params, sb, tb)
                    mus, sigmas, xis = jnp.split(gevparams, 3, axis=1)
                    for icluster in range(self.experiment.clusters["n"]):
                        crps = gev_crps(
                            jnp.repeat(
                                jnp.expand_dims(mus[:, icluster], axis=1),
                                lb[icluster].shape[1],
                                axis=1,
                            ),
                            jnp.repeat(
                                jnp.expand_dims(sigmas[:, icluster], axis=1),
                                lb[icluster].shape[1],
                                axis=1,
                            ),
                            jnp.repeat(
                                jnp.expand_dims(xis[:, icluster], axis=1),
                                lb[icluster].shape[1],
                                axis=1,
                            ),
                            lb[icluster],
                        ).mean(
                            axis=1
                        )  # Mean on stations
                        crps[i, icluster, 100 * ibatch : 100 * (ibatch + 1)] = crps
            if save:
                with open(
                    os.path.join(
                        self.experiment.folders["plot"]["folder"], "allCRPSs.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(crps, f)
            return s, t, l, crps

    # Subclass for saving
    class Saver:
        def __init__(self, experiment):
            self.experiment = experiment

        def information(self):
            """
            Save the information of the experiment by printing it to a text file in self.folders['plot']['folder'].
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
            # assert 'output' in self.experiment.CRPS.keys(), "Run the experiment first."
            # if os.path.exists(os.path.join(self.experiment.folders['plot']['dir'], 'Summary.csv')):
            #     smr = pd.read_csv(os.path.join(self.experiment.folders['plot']['dir'], 'Summary.csv'), index_col = 0)
            #     # Add line with experience number, output CRPS, and test CRPS
            #     if int(self.experiment.expnumber) in smr.index:
            #         smr.loc[int(self.experiment.expnumber)] = [self.experiment.CRPS['mean'], self.experiment.CRPS['std']]
            #     else:
            #         smr = pd.concat((smr, pd.DataFrame({'CRPS': self.experiment.CRPS['mean'], 'Std': self.experiment.CRPS['std']}, index = [int(self.experiment.expnumber)])))
            #     # Sort by experience number
            #     smr.sort_index(inplace = True)
            #     # Save to file
            #     smr.to_csv(os.path.join(self.experiment.folders['plot']['dir'], 'Summary.csv'), index_label = "Experiment")
            # else:
            #     smr = pd.DataFrame({'CRPS': [self.experiment.CRPS['mean']], 'Std': [self.experiment.CRPS['std']]}, index = [int(self.experiment.expnumber)])
            #     smr.to_csv(os.path.join(self.experiment.folders['plot']['dir'], 'Summary.csv'), index_label = "Experiment")

        def experimentfile(self):
            experiment_dict = {
                "files": self.experiment.files,
                "folders": self.experiment.folders,
                "features": self.experiment.features,
                "label": self.experiment.label,
                "nn_kwargs": self.experiment.nn_kwargs,
                "model_kwargs": self.experiment.model_kwargs,
                "filter": {
                    k: self.experiment.filter[k]
                    for k in ["lead_times", "storm_part", "2d"]
                },
                "CRPS": self.experiment.CRPS,
                "Data": self.experiment.Data,
            }
            with open(self.experiment.files["experiment"], "wb") as f:
                pickle.dump(experiment_dict, f)

    # Subclass for plotting
    class Plotter:
        def __init__(self, experiment):
            self.experiment = experiment

        def pithist(self):
            assert os.path.exists(
                os.path.join(
                    self.experiment.folders["plot"]["folder"], "best_states.pkl"
                )
            ), "Run the experiment first."
            if not hasattr(self.experiment, "test_s"):
                self.experiment.load_data()
            with open(
                os.path.join(
                    self.experiment.folders["plot"]["folder"], "best_states.pkl"
                ),
                "rb",
            ) as f:
                output_params, best_states = pickle.load(f)
            set_s = self.experiment.test_s
            set_t = self.experiment.test_t
            set_l = self.experiment.test_l
            ntimes = len(set_s) // 100
            print(len(set_l[0]))
            param_pred = jnp.concatenate(
                [
                    self.experiment.NN.apply(
                        output_params,
                        set_s[ntimes * i : ntimes * (i + 1)],
                        set_t[ntimes * i : ntimes * (i + 1)],
                    )
                    for i in range(100)
                ],
                axis=0,
            )
            print(len(param_pred))
            pit_histogram(
                tuple(map(lambda x: x[: ntimes * 100], set_l)),
                param_pred,
                os.path.join(
                    self.experiment.folders["plot"]["folder"], "PIT_histogram.png"
                ),
                title="PIT histogram",
                leadtime=None,
            )

            return gev_crps_loss(
                param_pred,
                tuple(map(lambda x: x[: len(set_s) // 100 * 100], set_l)),
                len(self.experiment.clusters["stations"]),
                len(set_s) // 100 * 100,
                self.experiment.clusters["n"],
            )

        def casestudy(self, date, lead_time):
            assert self.experiment.filter[
                "2d"
            ], "Case study only available for 2D data."
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
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl"),
                "rb",
            ) as f:
                param_set = pickle.load(f)
            model = self.experiment.NN_()
            # Create inputs
            ds = xr.open_dataset(file, engine="netcdf4").sel(time=date)
            temperature = ds.t2m.sel(lead_time=lead_time)
            pressure = ds.msl.sel(lead_time=lead_time)
            wind = ds[["u10", "v10"]].sel(lead_time=lead_time)
            ilt = ds.get_index("lead_time")
            ilt = {ilt[i]: i for i in range(len(ilt))}
            x_t = np.array(
                [
                    [
                        [
                            (date.day_of_year - 242)
                            / 107,  # Encoding of day of year between -1 and +1
                            np.cos(date.hour * np.pi / 12),
                            np.sin(date.hour * np.pi / 12),
                            lt / 72,
                        ]
                        for lt in ds.lead_time
                    ]
                ],
                dtype=np.float32,
            ).reshape(-1, 4)
            x_s = None
            if (
                self.experiment.Data["mean"] is None
                or self.experiment.Data["std"] is None
            ):
                self.experiment.load_mean_std()
            for ivar in range(len(self.experiment.features)):
                var = self.experiment.features[ivar]
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
                elif var == "wind":
                    tmp = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
                else:
                    tmp = ds[var]
                tmp = tmp.values
                tmp = (tmp - self.experiment.Data["mean"][ivar]) / self.experiment.Data[
                    "std"
                ][ivar]
                tmp = np.expand_dims(tmp, axis=-1)
                x_s = tmp if x_s is None else np.concatenate([x_s, tmp], axis=-1)
            x_s = jnp.array(x_s, dtype=jnp.float32)
            x_t = jnp.array(x_t, dtype=jnp.float32)
            # Create labels
            labels = xr.open_dataset(l_file).sel(time=date)
            labels = labels[self.experiment.label]
            nplabels = [None] * self.experiment.clusters["n"]
            for i in range(self.experiment.clusters["n"]):
                labtmp = labels.sel(
                    station=self.experiment.clusters["groups"][i]
                ).values
                labtmp = np.repeat(
                    np.expand_dims(labtmp, axis=0), len(ds.lead_time), axis=0
                )
                nplabels[i] = labtmp
            windgusts = tuple(
                [
                    jnp.array(nplabels[i], dtype=jnp.float32)
                    for i in range(self.experiment.clusters["n"])
                ]
            )

            gev_params = list(
                map(lambda param: model.apply(param, x_s, x_t), param_set)
            )
            gev_params = np.array(
                gev_params
            )  # Shape : (n_fold, n_lead_time, n_clusters*3)
            plot_params = gev_params[
                :, ilt[lead_time], :
            ]  # Shape : (n_fold, n_clusters*3)

            mus = plot_params[:, :5]
            sigmas = plot_params[:, 5:10]
            xis = plot_params[:, 10:]

            plot_params = plot_params.reshape(
                self.experiment.model_kwargs["n_folds"],
                3,
                self.experiment.clusters["n"],
            ).transpose(0, 2, 1)

            # Compute CRPS for each cluster
            CRPSs = np.array(
                [
                    [
                        gev_crps(
                            jnp.expand_dims(mus[[ifold], [icluster]], axis=1).repeat(
                                len(self.experiment.clusters["groups"][icluster]),
                                axis=-1,
                            ),
                            jnp.expand_dims(sigmas[[ifold], [icluster]], axis=1).repeat(
                                len(self.experiment.clusters["groups"][icluster]),
                                axis=-1,
                            ),
                            jnp.expand_dims(xis[[ifold], [icluster]], axis=1).repeat(
                                len(self.experiment.clusters["groups"][icluster]),
                                axis=-1,
                            ),
                            windgusts[icluster],
                        ).mean()
                        for icluster in range(self.experiment.clusters["n"])
                    ]
                    for ifold in range(self.experiment.model_kwargs["n_folds"])
                ]
            )
            lead_times = [k for k in ilt.keys()]

            storm_plot(
                stations,
                self.experiment.filter["storms"],
                plot_params,
                tuple(map(lambda x: x[ilt[lead_time]], windgusts)),
                temperature,
                pressure,
                wind,
                self.experiment.clusters["groups"],
                CRPSs,
                self.experiment.CRPS["mean"],
                date,
                os.path.join(
                    self.experiment.folders["plot"]["folder"],
                    f"Stormplot_{date}_{lead_time}.png",
                ),
            )

        def losses(self, exp_losses):
            fig, ax = plt.subplots(
                2,
                self.experiment.model_kwargs["n_folds"],
                figsize=(5 * self.experiment.model_kwargs["n_folds"], 10),
                sharex=True,
                sharey="row",
                subplot_kw={"xlabel": "Epochs"},
            )
            for ifold, loss in enumerate(exp_losses):
                if not loss is None:
                    train_loss = loss["train_loss"]
                    val_loss = loss["val_loss"]
                    train_crps = loss["train_CRPS"]
                    val_crps = loss["val_CRPS"]
                    epochs = np.arange(len(train_loss))
                    ax[0, ifold].plot(epochs, train_loss, label="train loss")
                    ax[0, ifold].plot(epochs, val_loss, label="val loss")
                    ax[0, ifold].set_title(f"Loss and CRPS for fold {ifold}")
                    ax[1, ifold].plot(epochs, train_crps, label="train CRPS")
                    ax[1, ifold].plot(epochs, val_crps, label="val CRPS")
                    ax[0, ifold].legend()
                    ax[1, ifold].legend()
                    ax[0, ifold].grid()
                    ax[1, ifold].grid()
                    if ifold == 0:
                        ax[0, ifold].set_ylabel("Loss")
                        ax[1, ifold].set_ylabel("CRPS")
            plt.savefig(
                os.path.join(self.experiment.folders["plot"]["folder"], "Losses.png")
            )

        def ltCRPS(self, keep=False):
            """
            Plots the variation of the CRPS with the lead time.
            """
            if not os.path.exists(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl")
            ):
                raise ValueError("Run the experiment first.")
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl"),
                "rb",
            ) as f:
                paramlist = pickle.load(f)
            crps = np.zeros(
                (
                    self.experiment.model_kwargs["n_folds"],
                    len(self.experiment.filter["lead_times"]),
                )
            )
            with open(
                os.path.join(
                    self.experiment.folders["scratch"]["folder"], f"test_set.pkl"
                ),
                "rb",
            ) as f:
                s, t, l = pickle.load(f)
            for i, params in enumerate(paramlist):
                for j, lt in enumerate(self.experiment.filter["lead_times"]):
                    print(lt)
                    sb = jnp.array(s[t[:, 3] == lt / 72])
                    tb = jnp.array(t[t[:, 3] == lt / 72])
                    lb = tuple(
                        [
                            jnp.array(l[i][t[:, 3] == lt / 72])
                            for i in range(self.experiment.clusters["n"])
                        ]
                    )
                    count = 0
                    for x_s, x_t, y_true in create_batches(
                        sb,
                        tb,
                        lb,
                        self.experiment.model_kwargs["batch_size"],
                        self.experiment.model_kwargs["rng"]["shuffle"],
                    ):
                        gevparams = self.experiment.NN.apply(params, x_s, x_t)
                        if self.experiment.model_kwargs["target"] == "GEV":
                            crps[i, j] += gev_crps_loss(
                                gevparams,
                                y_true,
                                len(self.experiment.clusters["stations"]),
                                self.experiment.model_kwargs["batch_size"],
                                self.experiment.clusters["n"],
                            ).mean()
                        else:
                            crps[i, j] += mae(
                                gevparams,
                                y_true,
                                self.experiment.model_kwargs["batch_size"],
                                len(self.experiment.clusters["stations"]),
                                self.experiment.clusters["n"],
                            )
                        count += 1
                    crps[i, j] /= count
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

        def clusterCRPS(self, cluster, date=None):
            if not os.path.exists(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl")
            ):
                raise ValueError("Run the experiment first.")
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl"),
                "rb",
            ) as f:
                paramlist = pickle.load(f)
            crps = np.zeros(
                (
                    self.experiment.model_kwargs["n_folds"],
                    len(self.experiment.filter["lead_times"]),
                )
            )
            ds = xr.open_dataset(
                self.experiment.files["test"]["inputs"][0], engine="netcdf4"
            )
            if date is not None:
                input_time = ds.sel(time=date).time.values
                if not isinstance(input_time, np.ndarray):
                    input_time = np.array([input_time])
                input_time = pd.DatetimeIndex(input_time)
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
                            for time in input_time
                        ]
                        for lt in self.experiment.filter["lead_times"]
                    ],
                    dtype=np.float32,
                ).reshape(-1, 4)
            with open(
                os.path.join(
                    self.experiment.folders["scratch"]["folder"], f"test_set.pkl"
                ),
                "rb",
            ) as f:
                s, t, l = pickle.load(f)
            index = np.array(
                [
                    any([np.all(np.where(ti == titmp, True, False)) for titmp in tmp_t])
                    for ti in t
                ]
            )
            s = s[index]
            l = tuple([l[i][index] for i in range(self.experiment.clusters["n"])])
            t = t[index]
            for i, params in enumerate(paramlist):
                for j, lt in enumerate(self.experiment.filter["lead_times"]):
                    sb = jnp.array(s[t[:, 3] == lt / 72])
                    tb = jnp.array(t[t[:, 3] == lt / 72])
                    lb = tuple(
                        [
                            jnp.array(l[i][t[:, 3] == lt / 72])
                            for i in range(self.experiment.clusters["n"])
                        ]
                    )
                    outputs = self.experiment.NN.apply(params, sb, tb)
                    mus, sigmas, xis = jnp.split(outputs, 3, axis=1)
                    crps[i, j] = gev_crps(
                        jnp.repeat(
                            jnp.expand_dims(mus[:, cluster], axis=1),
                            lb[cluster].shape[-1],
                            axis=1,
                        ),
                        jnp.repeat(
                            jnp.expand_dims(sigmas[:, cluster], axis=1),
                            lb[cluster].shape[-1],
                            axis=1,
                        ),
                        jnp.repeat(
                            jnp.expand_dims(xis[:, cluster], axis=1),
                            lb[cluster].shape[-1],
                            axis=1,
                        ),
                        lb[cluster],
                    ).mean()
            metrics_evolution(
                {"CRPS": crps},
                self.experiment.filter["lead_times"],
                os.path.join(
                    self.experiment.folders["plot"]["folder"],
                    f'CRPS_cluster_{cluster}{"_" + date if date is not None else None}.png',
                ),
            )
            return crps

        def returnlevel(self, returnlevels):
            """
            Compute the return periods for the given periods.
            """
            if not os.path.exists(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl")
            ):
                raise ValueError("Run the experiment first.")
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl"),
                "rb",
            ) as f:
                paramlist = pickle.load(f)
            rl_model = np.zeros(
                (
                    self.experiment.model_kwargs["n_folds"],
                    len(self.experiment.filter["lead_times"]),
                    len(returnlevels),
                )
            )
            for i, params in enumerate(paramlist):
                with open(
                    os.path.join(
                        self.experiment.folders["scratch"]["folder"], f"fold_{i}.pkl"
                    ),
                    "rb",
                ) as f:
                    s, t, l = pickle.load(f)
                with open(
                    os.path.join(
                        self.experiment.folders["scratch"]["folder"], f"storms_{i}.pkl"
                    ),
                    "rb",
                ) as f:
                    storms = pickle.load(f)
                s = s[storms == 1]
                t = t[storms == 1]
                l = tuple(map(lambda x: x[storms == 1], l))
                for j, lt in enumerate(self.experiment.filter["lead_times"]):
                    print(lt)
                    sb = jnp.array(s[t[:, 3] == lt / 72])
                    tb = jnp.array(t[t[:, 3] == lt / 72])
                    lb = tuple(
                        [
                            jnp.array(l[i][t[:, 3] == lt / 72])
                            for i in range(self.experiment.clusters["n"])
                        ]
                    )
                    count = 0
                    for x_s, x_t, y_true in create_batches(
                        sb,
                        tb,
                        lb,
                        self.experiment.model_kwargs["batch_size"],
                        self.experiment.model_kwargs["rng"]["shuffle"],
                    ):
                        gevparams = self.experiment.NN.apply(params, x_s, x_t)
                        for ip, p in enumerate(returnlevels):
                            rl_model[i, j, ip] += return_level_loss(
                                gevparams,
                                y_true,
                                len(self.experiment.clusters["stations"]),
                                self.experiment.model_kwargs["batch_size"],
                                self.experiment.clusters["n"],
                                p,
                            )
                        count += 1
                    rl_model[i, j, :] /= count
            with open(
                os.path.join(
                    self.experiment.folders["plot"]["folder"], "returnlevels.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(rl_model, f)
            for ip, p in enumerate(returnlevels):
                metrics_evolution(
                    {f"Return level {p}": rl_model[:, :, ip]},
                    self.experiment.filter["lead_times"],
                    os.path.join(
                        self.experiment.folders["plot"]["folder"],
                        f"ReturnLevel_{p}.png",
                    ),
                )

        def parameters(self):
            """
            Plots the variation of the CRPS with the lead time.
            """
            if not os.path.exists(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl")
            ):
                raise ValueError("Run the experiment first.")
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl"),
                "rb",
            ) as f:
                paramlist = pickle.load(f)
            gevparams = None
            with open(
                os.path.join(
                    self.experiment.folders["scratch"]["folder"], f"test_set.pkl"
                ),
                "rb",
            ) as f:
                s, t, l = pickle.load(f)
            for i, params in enumerate(paramlist):
                for j, lt in enumerate(self.experiment.filter["lead_times"]):
                    print(lt)
                    sb = jnp.array(s[t[:, 3] == lt / 72])
                    tb = jnp.array(t[t[:, 3] == lt / 72])
                    lb = tuple(
                        [
                            jnp.array(l[i][t[:, 3] == lt / 72])
                            for i in range(self.experiment.clusters["n"])
                        ]
                    )
                    count = 0
                    for x_s, x_t, y_true in create_batches(
                        sb,
                        tb,
                        lb,
                        self.experiment.model_kwargs["batch_size"],
                        self.experiment.model_kwargs["rng"]["shuffle"],
                    ):
                        tmpparams = self.experiment.NN.apply(params, x_s, x_t)
                        gevparams = (
                            tmpparams
                            if gevparams is None
                            else jnp.concatenate([gevparams, tmpparams], axis=0)
                        )
            param_distribution(
                gevparams,
                os.path.join(
                    self.experiment.folders["plot"]["folder"], "Parameters.png"
                ),
            )

        def shapvals(
            self,
            parameter=None,
            cluster=None,
            rng=jax.random.PRNGKey(0),
            samplesize=5,
            fold=0,
        ):
            # Not tested with unfixed parameter and cluster
            with open(
                os.path.join(
                    self.experiment.folders["scratch"]["folder"], f"test_set.pkl"
                ),
                "rb",
            ) as f:
                s, t, l = pickle.load(f)
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl"),
                "rb",
            ) as f:
                paramlist = pickle.load(f)
            sShape = s.shape
            tShape = t.shape
            s = s.reshape(sShape[0], -1)
            data = np.concatenate((t, s), axis=-1)
            iparam = 0 if parameter == "mu" else 1
            ioutputs = (
                list(range(15))
                if parameter is None and cluster is None
                else (
                    [5 * i + cluster for i in range(3)]
                    if parameter is None
                    else (
                        [iparam * 5 + cluster for cluster in range(5)]
                        if cluster is None
                        else [iparam * 5 + cluster]
                    )
                )
            )

            def predict(ds):
                t, s = np.split(ds, [tShape[1]], axis=-1)
                s = s.reshape(-1, sShape[1], sShape[2], sShape[3])
                return np.array(
                    self.experiment.NN.apply(paramlist[fold], s, t)[:, ioutputs]
                )

            explainer = shap.KernelExplainer(predict, data.mean(axis=0, keepdims=True))
            shap_values = explainer(
                data[jax.random.permutation(rng, len(data))[:samplesize]]
            )
            res = shap_values.values
            resmean = np.abs(res).mean(axis=0)
            temporal, spatial = np.split(resmean, [tShape[1]])
            spatial = spatial.reshape(sShape[1], sShape[2], sShape[3])
            # Save
            with open(
                os.path.join(
                    self.experiment.folders["plot"]["folder"],
                    f"shap_values_fold{fold}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(shap_values, f)
            # Map
            ds = xr.open_dataset(
                self.experiment.files["test"]["inputs"][0], engine="netcdf4"
            ).isel(time=0, lead_time=0)
            for i in range(spatial.shape[-1]):
                feature = self.experiment.features[i]
                ds["shap"] = (["lat", "lon"], spatial[:, :, i])
                # Figure
                fig = plt.figure(figsize=(4.8, 3.0))
                proj = ccrs.PlateCarree()
                extents = [5.0, 11.5, 45.0, 48.5]
                ax = fig.add_subplot(1, 1, 1, projection=proj)
                ax.add_feature(cfeature.BORDERS)
                ax.set_extent(extents)
                vext = max(abs(ds.shap.min()), abs(ds.shap.max()))
                cbar = ds.shap.plot(
                    cmap="RdBu_r",
                    add_colorbar=False,
                    transform=ccrs.PlateCarree(),
                    ax=ax,
                    vmin=-vext,
                    vmax=vext,
                )
                plt.title(f"Shap values for {feature}")
                plt.colorbar(
                    cbar, ax=ax, orientation="vertical", label="Shap values", shrink=0.8
                )
                plt.savefig(
                    os.path.join(
                        self.experiment.folders["plot"]["folder"],
                        f"Shap_{feature}_{fold}.png",
                    )
                )
                plt.close()
            # Feature importance
            fig = plt.figure(figsize=(4.8, 3.0))
            shapspatial = spatial.mean(axis=(0, 1))
            plt.barh(range(len(shapspatial)), shapspatial)
            plt.yticks(range(len(shapspatial)), self.experiment.features)
            plt.title("Feature importance")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.experiment.folders["plot"]["folder"],
                    f"Feature_importance_{fold}.png",
                )
            )
            plt.close()
            return resmean

        def shap(
            self,
            parameter="mu",
            cluster=0,
            date="2021-06-20 17:00",
            lead_time=12,
            rng=jax.random.PRNGKey(0),
            output="map",
            explainer="partition",
        ):
            assert parameter in [
                "mu",
                "sigma",
            ], "Parameter must be either 'mu' or 'sigma'."
            iparam = 0 if parameter == "mu" else 1
            with open(
                os.path.join(
                    self.experiment.folders["scratch"]["folder"], f"test_set.pkl"
                ),
                "rb",
            ) as f:
                s, t, l = pickle.load(f)
            with open(
                os.path.join(self.experiment.folders["plot"]["folder"], "params.pkl"),
                "rb",
            ) as f:
                paramlist = pickle.load(f)
            if date is not None:
                date = pd.to_datetime(date)
                t_line = np.array(
                    [
                        [
                            [
                                (date.day_of_year - 242)
                                / 107,  # Encoding of day of year between -1 and +1
                                np.cos(date.hour * np.pi / 12),
                                np.sin(date.hour * np.pi / 12),
                                lead_time / 72,
                            ]
                        ]
                    ],
                    dtype=np.float32,
                ).reshape(-1, 4)
                index = np.array(
                    [np.all(np.where(ti == t_line, True, False)) for ti in t]
                )
            ltindex = (
                t[:, 3] == lead_time / 72
                if lead_time is not None
                else np.ones(t.shape[0], dtype=bool)
            )
            s_shape = s.shape
            t_shape = t.shape
            print(s_shape, t_shape)
            if explainer == "partition":
                t = np.tile(
                    np.expand_dims(t, axis=(1, 2)), [1, s_shape[1], s_shape[2], 1]
                )
                data = np.concatenate((t, s), axis=-1)

                def predict(ds):
                    t, s = jnp.split(ds, [t_shape[1]], axis=-1)
                    t = t[:, 0, 0, :]
                    return np.array(
                        self.experiment.NN.apply(paramlist[0], s, t)[
                            :, iparam * 5 + cluster
                        ]
                    )

                masker = shap.maskers.Image("blur(3,3)", data.shape[1:])
                explainer = shap.PartitionExplainer(predict, masker)
                if output == "map":
                    if date is not None:
                        res = explainer(data[index]).values
                        # Open test file
                        ds = xr.open_dataset(
                            self.experiment.files["test"]["inputs"][0], engine="netcdf4"
                        ).sel(time=date, lead_time=lead_time)
                        stations = (
                            xr.open_dataset(self.experiment.files["test"]["labels"][0])
                            .isel(time=0)
                            .sel(station=self.experiment.clusters["groups"][cluster])
                        )
                        lons, lats = stations.longitude.values, stations.latitude.values
                        ds["shap"] = (["lat", "lon"], res[0, :, :, 0])
                        # Figure
                        fig = plt.figure(figsize=(4.8, 5.0))
                        # proj = ccrs.Orthographic(central_longitude = 8.3, central_latitude = 46.8)
                        proj = ccrs.PlateCarree()
                        extents = [5.0, 11.5, 45.0, 48.5]
                        ax = [None] * 2
                        ax[0] = fig.add_subplot(2, 1, 1, projection=proj)
                        ax[0].add_feature(cfeature.BORDERS)
                        ax[0].set_extent(extents)
                        np.sqrt(ds.u10**2 + ds.v10**2).plot(
                            cmap="viridis",
                            add_colorbar=False,
                            transform=ccrs.PlateCarree(),
                            ax=ax[0],
                        )
                        plt.scatter(
                            lons,
                            lats,
                            transform=ccrs.PlateCarree(),
                            c="tab:orange",
                            marker=".",
                            s=1,
                        )
                        plt.title("10m wind speed")
                        ax[1] = fig.add_subplot(2, 1, 2, projection=proj)
                        ax[1].add_feature(cfeature.BORDERS)
                        ax[1].set_extent(extents)
                        vext = max(abs(ds.shap.min()), abs(ds.shap.max()))
                        np.sqrt(ds.u10**2 + ds.v10**2).plot(
                            cmap="gray",
                            add_colorbar=False,
                            transform=ccrs.PlateCarree(),
                            ax=ax[1],
                        )
                        sh = ds.shap.plot(
                            alpha=0.6,
                            transform=ccrs.PlateCarree(),
                            ax=ax[1],
                            add_colorbar=False,
                            cmap="RdBu_r",
                            vmin=-vext,
                            vmax=vext,
                        )
                        plt.title("Shap values")
                        cbar = plt.colorbar(
                            sh,
                            ax=ax,
                            orientation="vertical",
                            label="Shap values",
                            shrink=0.8,
                        )
                        plt.suptitle(
                            f"Shap values for {parameter}, cluster {cluster} \nat {date} and lead time {lead_time}h"
                        )
                        plt.savefig(
                            os.path.join(
                                self.experiment.folders["plot"]["folder"],
                                f"Shap_{parameter}_{cluster}_{date}_{lead_time}.png",
                            )
                        )
                        plt.close()
                        with open(
                            os.path.join(
                                self.experiment.folders["plot"]["folder"],
                                f"Shap_{parameter}_{cluster}_{date}_{lead_time}.pkl",
                            ),
                            "wb",
                        ) as f:
                            pickle.dump(res, f)
                    else:
                        res = explainer(data[ltindex]).values
                        res = res[:, :, :, 0].mean(axis=0)
                        ds = xr.open_dataset(
                            self.experiment.files["test"]["inputs"][0], engine="netcdf4"
                        ).isel(time=0, lead_time=0)
                        stations = (
                            xr.open_dataset(self.experiment.files["test"]["labels"][0])
                            .isel(time=0)
                            .sel(station=self.experiment.clusters["groups"][cluster])
                        )
                        lons, lats = stations.longitude.values, stations.latitude.values
                        ds["shap"] = (["lat", "lon"], res)
                        # Figure
                        fig = plt.figure(figsize=(4.8, 3.0))
                        proj = ccrs.PlateCarree()
                        extents = [5.0, 11.5, 45.0, 48.5]
                        ax = fig.add_subplot(1, 1, 1, projection=proj)
                        ax.add_feature(cfeature.BORDERS)
                        ax.set_extent(extents)
                        vext = max(abs(ds.shap.min()), abs(ds.shap.max()))
                        cbar = ds.shap.plot(
                            cmap="RdBu_r",
                            add_colorbar=False,
                            transform=ccrs.PlateCarree(),
                            ax=ax,
                            vmin=-vext,
                            vmax=vext,
                        )
                        plt.scatter(
                            lons, lats, transform=ccrs.PlateCarree(), c="tab:green", s=5
                        )
                        plt.title(
                            f"Mean shap values for {parameter}, cluster {cluster}, lead time {lead_time}h"
                        )
                        plt.colorbar(
                            cbar,
                            ax=ax,
                            orientation="vertical",
                            label="Shap values",
                            shrink=0.8,
                        )
                        plt.savefig(
                            os.path.join(
                                self.experiment.folders["plot"]["folder"],
                                f"Shap_{parameter}_{cluster}_{lead_time}.png",
                            )
                        )
                        plt.close()
                        with open(
                            os.path.join(
                                self.experiment.folders["plot"]["folder"],
                                f"Shap_{parameter}_{cluster}_{lead_time}.pkl",
                            ),
                            "wb",
                        ) as f:
                            pickle.dump(res, f)
                elif output == "feature importance":
                    res = explainer(data[ltindex]).values
                    return res
            else:
                s = s.reshape(s_shape[0], -1)
                data = np.concatenate((t, s), axis=-1)
                print(data.shape)

                def predict(ds):
                    t, s = np.split(ds, [t_shape[1]], axis=-1)
                    s = s.reshape(-1, s_shape[1], s_shape[2], s_shape[3])
                    return np.array(
                        self.experiment.NN.apply(paramlist[0], s, t)[
                            :, iparam * 5 + cluster
                        ]
                    )

                print(data.mean(axis=0, keepdims=True).shape)
                explainer = shap.KernelExplainer(
                    predict, data.mean(axis=0, keepdims=True)
                )
                return explainer, data, ltindex

    @staticmethod
    def createExperimentFile(**kwargs):

        save_path = kwargs.pop("save_path", None)
        if save_path:
            if os.path.isdir(save_path):
                save_path = os.path.join(
                    save_path, f"experiment_{len(os.listdir(save_path))}.pkl"
                )
            else:
                if os.path.exists(save_path):
                    if not kwargs.pop("overwrite", False):
                        raise ValueError(
                            f"{save_path} already exists. Set overwrite to True to overwrite it."
                        )
        else:
            save_path = os.getcwd()
            experiments = os.listdir(save_path)
            save_path = os.path.join(save_path, f"experiment_{len(experiments)}.pkl")
        experiment = {}
        for key, value in kwargs.items():
            experiment[key] = value

        with open(save_path, "wb") as f:
            pickle.dump(experiment, f)

        print(f"Experiment file saved at {save_path}")

        return
