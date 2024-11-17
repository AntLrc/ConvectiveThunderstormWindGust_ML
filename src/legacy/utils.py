import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd

from src.legacy.loss import gev_crps_loss
from src.legacy.distributions import gev

def pit_histogram(y_true, param_pred, save_path, title=None, leadtime=None):
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
    mu, sigma, xi = jnp.split(param_pred, 3, axis=1)

    clusters_len = jnp.asarray(jax.tree_map(lambda x: x.shape[1], y_true))

    n_clusters = len(clusters_len)

    mu = jnp.repeat(mu, clusters_len, axis=1)
    sigma = jnp.repeat(sigma, clusters_len, axis=1)
    xi = jnp.repeat(xi, clusters_len, axis=1)

    y_true_concat = jnp.concatenate(y_true, axis=1)
    total_len = y_true_concat.shape[1]

    sns.set_theme()
    pit = gev(mu, sigma, xi, y_true_concat)

    if leadtime is None:
        data = pit.flatten()
        sns.histplot(data, stat="density", bins=50)
        plt.plot([0, 1], [1, 1], color="black")
        plt.vlines(x=[0.0, 1.0], ymin=0, ymax=1, color="black")
        if title:
            plt.title(title)
    else:
        correspondance_lt = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
            13: 13,
            14: 14,
            15: 15,
            16: 16,
            17: 17,
            18: 18,
            19: 19,
            20: 20,
            21: 21,
            22: 22,
            23: 23,
            24: 24,
            27: 25,
            30: 26,
            33: 27,
            36: 28,
            42: 29,
            48: 30,
            60: 31,
            72: 32,
        }
        if not isinstance(leadtime, list):
            leadtime = [leadtime]
        n_cols = (len(leadtime) + 1) // 2
        fig, axs = plt.subplots(
            2, n_cols, figsize=(5 * n_cols, 10), sharex=True, sharey=True
        )
        plt.tight_layout()
        ntimes = len(pit) // 33
        i = 0
        for lt in leadtime:
            data = pit[
                correspondance_lt[lt] * ntimes : (correspondance_lt[lt] + 1) * ntimes
            ].flatten()
            dataCRPS = gev_crps_loss(
                param_pred[
                    correspondance_lt[lt]
                    * ntimes : (correspondance_lt[lt] + 1)
                    * ntimes
                ],
                tuple(
                    map(
                        lambda x: x[
                            correspondance_lt[lt]
                            * ntimes : (correspondance_lt[lt] + 1)
                            * ntimes
                        ],
                        y_true,
                    )
                ),
                total_len=total_len,
                batch_size=ntimes,
                n_clusters=n_clusters,
            )
            sns.histplot(
                data,
                stat="density",
                bins=50,
                ax=axs[i // n_cols, i % n_cols],
                label=f"CRPS: {dataCRPS:.3f}",
            )
            axs[i // n_cols, i % n_cols].plot([0, 1], [1, 1], color="black")
            axs[i // n_cols, i % n_cols].vlines(
                x=[0.0, 1.0], ymin=0, ymax=1, color="black"
            )
            axs[i // n_cols, i % n_cols].set_title(
                f"Rank histogram for lead time {lt} hours"
            )
            axs[i // n_cols, i % n_cols].legend()
            i += 1
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_temporal_vars(data, var, filtr, save_path, stations=None):
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
            x = (
                ds.sel(time=pd.DatetimeIndex(np.intersect1d(ds.time, filtr)))
                .to_dataframe()
                .reset_index()
            )
            x["month"] = x.time.dt.month
            x["hour"] = x.time.dt.hour
            x["Filtered"] = "Storm"
            x = x.drop(labels=["time", "latitude", "longitude"], axis=1)
            if not stations is None:
                x = x[x.station.isin(stations)]
            else:
                x = x.drop(labels="station", axis=1)
            df = x if df is None else pd.concat([df, x], ignore_index=True)

            y = ds.to_dataframe().reset_index()
            y["month"] = y.time.dt.month
            y["hour"] = y.time.dt.hour
            y["Filtered"] = "Unfiltered"
            y = y.drop(labels=["time", "latitude", "longitude"], axis=1)
            if not stations is None:
                y = y[y.station.isin(stations)]
            else:
                y = y.drop(labels="station", axis=1)
            df = pd.concat([df, y], ignore_index=True)
    else:
        # filtr is structured as a dict with time steps for keys and a set of stations for values
        df = None
        for file in data:
            ds = xr.open_dataset(file)[var]
            x = (
                ds.sel(
                    time=pd.DatetimeIndex(
                        np.intersect1d(ds.time, pd.DatetimeIndex(filtr.keys()))
                    )
                )
                .to_dataframe()
                .reset_index()
            )
            x["month"] = x.time.dt.month
            x["hour"] = x.time.dt.hour
            x["Filtered"] = "0"
            for t in filtr.keys():
                # Set the Filtered value to "Storm" for the rows for which time is t and station is in filtr[t]
                x.loc[(x.time == t) & (x.station.isin(filtr[t])), "Filtered"] = "Storm"
            x = x.drop(labels=["time", "latitude", "longitude"], axis=1)
            x = x[x.Filtered != "0"]
            df = x if df is None else pd.concat([df, x], ignore_index=True)

            y = ds.to_dataframe().reset_index()
            y["month"] = y.time.dt.month
            y["hour"] = y.time.dt.hour
            y["Filtered"] = "Unfiltered"
            y = y.drop(labels=["time", "latitude", "longitude"], axis=1)
            df = pd.concat([df, y], ignore_index=True)

    # Plot the data
    # Relplot
    sns.set_theme()
    sns.relplot(
        data=df,
        x="hour",
        y="wind_speed_of_gust",
        hue="Filtered",
        col="month",
        row="station" if not stations is None else None,
        kind="line",
        errorbar="sd",
    )
    plt.savefig(
        os.path.join(
            save_path,
            "_".join(stations if not stations is None else []) + "RelPlotWindSpeed.png",
        )
    )
    plt.close()

    # Catplot
    sns.set_theme()
    sns.catplot(
        data=df,
        kind="violin",
        x="hour",
        y="wind_speed_of_gust",
        row="station" if not stations is None else None,
        hue="Filtered",
        split=True,
        height=5,
        aspect=4,
    )
    plt.savefig(
        os.path.join(
            save_path,
            "_".join(stations if not stations is None else []) + "CatPlotWindSpeed.png",
        )
    )
    plt.close()
    
def param_distribution(param, save_path):
    """
    Plot the distribution of the parameters.

    param: dict with the model name as key and the parameters as values.
    """
    n_clusters = param.shape[1] // 3
    df = pd.DataFrame(
        param,
        columns=[f"mu_{i}" for i in range(n_clusters)]
        + [f"sigma_{i}" for i in range(n_clusters)]
        + [f"xi_{i}" for i in range(n_clusters)],
    )
    df = df.melt(var_name="Cluster", value_name="Value")
    df["Parameter"] = df.Cluster.apply(lambda x: x.split("_")[0])
    df["Cluster"] = df.Cluster.apply(lambda x: x.split("_")[1])
    sns.displot(
        data=df, x="Value", col="Cluster", row="Parameter", kind="hist", stat="density"
    )
    plt.savefig(save_path)
    plt.close()