import xarray as xr
import numpy as np
import geopandas as gpd
import os
import pickle
import jax.numpy as jnp
from src.utils import pit_histogram, gev_crps_loss, storm_plot, metrics_evolution
from src.nn_train import create_batches
from src.nn_losses import return_level_loss
from src.nn_loader import Experiment


class EnhancedPlotter:
    def __init__(self, experiment):
        self.experiment = experiment

    def pithist(self):
        """
        Plot the PIT histogram of the model.
        """
        assert os.path.exists(
            os.path.join(self.experiment.folders["plot"]["folder"], "best_states.pkl")
        ), "Run the experiment first."
        if not hasattr(self.experiment, "test_s"):
            self.experiment.load_data()
        with open(
            os.path.join(self.experiment.folders["plot"]["folder"], "best_states.pkl"),
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

    def returnlevel(self, returnlevels):
        """
        Compute the return periods for the given periods.
        
        Parameters
        ----------
        returnlevels : list
            List of return periods to consider.
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
