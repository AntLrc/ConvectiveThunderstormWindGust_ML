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

from nn_losses import gev, gev_crps_loss, gev_pdf

from storm import Storm, Storms


def visualise_labels(train_labels, val_labels, test_labels, save_path, label_name):
    """
    Visualise labels distribution.
    
    Parameters
    ----------
    train_labels: jnp.array
        Labels of the training set.
    val_labels: jnp.array
        Labels of the validation set.
    test_labels: jnp.array
        Labels of the test set.
    save_path: str
        Path where to save the plot.
    label_name: str
        Name of the label.
    """
    sns.set_theme(style="darkgrid")
    train_labels = pd.DataFrame(
        jnp.concat(train_labels, axis=None), columns=[label_name]
    )
    train_labels["dataset"] = "train"
    val_labels = pd.DataFrame(jnp.concat(val_labels, axis=None), columns=[label_name])
    val_labels["dataset"] = "validation"
    test_labels = pd.DataFrame(jnp.concat(test_labels, axis=None), columns=[label_name])
    test_labels["dataset"] = "test"
    dataset = pd.concat([train_labels, val_labels, test_labels])
    sns.histplot(
        data=dataset,
        x=label_name,
        hue="dataset",
        stat="proportion",
        binwidth=0.5,
        common_norm=False,
    )
    plt.savefig(save_path)
    plt.close()


def visualise_features(
    train_features, val_features, test_features, save_path, features_names
):
    """
    Visualise features distribution.
    
    Parameters
    ----------
    train_features: jnp.array
        Features of the training set.
    val_features: jnp.array
        Features of the validation set.
    test_features: jnp.array
        Features of the test set.
    save_path: str
        Path where to save the plot.
    features_names: list
        Names of the features.
    """
    assert (
        len(features_names)
        == train_features.shape[-1]
        == val_features.shape[-1]
        == test_features.shape[-1]
    ), "Number of features names must be equal to the number of features in the dataset."

    sns.set_theme(style="darkgrid")
    # Create subplot with four rows (one for each set, plus one with all sets) and the number of features columns
    fig, axs = plt.subplots(4, len(features_names), figsize=(20, 20))
    train_features = train_features.reshape(-1, train_features.shape[-1])
    val_features = val_features.reshape(-1, val_features.shape[-1])
    test_features = test_features.reshape(-1, test_features.shape[-1])

    train_features = pd.DataFrame(train_features, columns=features_names)
    val_features = pd.DataFrame(val_features, columns=features_names)
    test_features = pd.DataFrame(test_features, columns=features_names)

    train_features["dataset"] = "train"
    val_features["dataset"] = "validation"
    test_features["dataset"] = "test"

    dataset = pd.concat([train_features, val_features, test_features])

    for i in range(len(features_names)):
        sns.histplot(
            data=dataset,
            x=features_names[i],
            hue="dataset",
            stat="proportion",
            common_norm=False,
            ax=axs[0, i],
        )
        sns.histplot(
            data=train_features,
            x=features_names[i],
            stat="proportion",
            common_norm=False,
            ax=axs[1, i],
        )
        sns.histplot(
            data=val_features,
            x=features_names[i],
            stat="proportion",
            common_norm=False,
            ax=axs[2, i],
        )
        sns.histplot(
            data=test_features,
            x=features_names[i],
            stat="proportion",
            common_norm=False,
            ax=axs[3, i],
        )

    plt.savefig(save_path)
    plt.close()


def storm_plot(
    stations,
    storms,
    gev_params,
    wind_gusts,
    temperature,
    pressure,
    wind,
    clusters,
    crps_s,
    exp_crps,
    date,
    save_path,
    all_crps_s=None,
    cluster_nb=None,
):
    """
    Plot the storms on a map.
    """

    n_clusters = len(clusters)
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(date, fontsize=20)

    # Plot the map
    gs = GridSpec(2, 2, top=0.95, bottom=0.55)
    crs_ortho = ccrs.Orthographic(central_longitude=8.3, central_latitude=46.8)
    ax = fig.add_subplot(gs[0, 0], projection=crs_ortho)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([5.8, 10.7, 45.7, 47.8])
    colors = [
        "tab:blue",
        "tab:brown",
        "tab:orange",
        "tab:pink",
        "tab:green",
        "tab:gray",
        "tab:olive",
        "tab:purple",
        "tab:cyan",
    ]
    if not stations.crs == crs_ortho:
        stations = stations.to_crs(crs_ortho)
    stations["cluster"] = "0."
    for i in range(len(clusters)):
        for station in clusters[i]:
            if station in stations.index:
                stations.loc[station, "cluster"] = f"Cluster {int(i + 1)}"
    for i in range(len(clusters)):
        stations.where(stations.cluster == f"Cluster {int(i + 1)}").plot(
            ax=ax,
            color=colors[i],
            legend=True,
            legend_kwds={"label": f"Cluster {int(i + 1)}"},
        )
    if date in storms.dates:
        geom = stations.loc[storms.dates[date]["stations"]].geometry
        if not geom.crs == crs_ortho:
            geom = geom.to_crs(crs_ortho)
        geom.plot(ax=ax, color="red", marker="x")
        storm_ids = storms.dates[date]["storms"]
        for storm_id in storm_ids:
            geom = (
                storms.storms[storm_id]
                .track.iloc[
                    (storms.storms[storm_id].track.index + pd.Timedelta("55min")).floor(
                        "h"
                    )
                    == date
                ]
                .geometry
            )
            if not geom.crs == crs_ortho:
                geom = geom.to_crs(crs_ortho)
            chx_storm, chy_storm = geom.x, geom.y
            ax.plot(chx_storm, chy_storm, color="red")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Stations and storms")
    colrs = colors[: len(clusters)]
    labels = [f"Cluster {int(i + 1)}" for i in range(len(clusters))]
    handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=10,
            label=label,
        )
        for color, label in zip(colrs, labels)
    ]
    ax.legend(handles=handles)

    ax = fig.add_subplot(gs[0, 1], projection=crs_ortho)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([5.8, 10.7, 45.7, 47.8])
    ts = temperature.plot.contourf(
        ax=ax,
        cmap="coolwarm",
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": "Temperature (K)", "shrink": 0.6},
    )
    ax.clabel(ts, inline=True, colors="k")
    ax.set_title("Temperature")
    if date in storms.dates:
        storm_ids = storms.dates[date]["storms"]
        for storm_id in storm_ids:
            geom = (
                storms.storms[storm_id]
                .track.iloc[
                    (storms.storms[storm_id].track.index + pd.Timedelta("55min")).floor(
                        "h"
                    )
                    == date
                ]
                .geometry
            )
            if not geom.crs == crs_ortho:
                geom = geom.to_crs(crs_ortho)
            chx_storm, chy_storm = geom.x, geom.y
            ax.plot(chx_storm, chy_storm, color="red")

    ax = fig.add_subplot(gs[1, 0], projection=crs_ortho)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([5.8, 10.7, 45.7, 47.8])
    ps = pressure.plot.contour(
        ax=ax,
        cmap="coolwarm",
        transform=ccrs.PlateCarree(),
        add_colorbar=True,
        cbar_kwargs={"label": "Pressure (Pa)", "shrink": 0.6},
    )
    ax.clabel(ps, inline=True)
    ax.set_title("Pressure")
    if date in storms.dates:
        storm_ids = storms.dates[date]["storms"]
        for storm_id in storm_ids:
            geom = (
                storms.storms[storm_id]
                .track.iloc[
                    (storms.storms[storm_id].track.index + pd.Timedelta("55min")).floor(
                        "h"
                    )
                    == date
                ]
                .geometry
            )
            if not geom.crs == crs_ortho:
                geom = geom.to_crs(crs_ortho)
            chx_storm, chy_storm = geom.x, geom.y
            ax.plot(chx_storm, chy_storm, color="red")

    ax = fig.add_subplot(gs[1, 1], projection=crs_ortho)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([5.8, 10.7, 45.7, 47.8])
    np.sqrt(wind.u10**2 + wind.v10**2).sel(
        lon=slice(5.5, 11), lat=slice(48, 45.5)
    ).plot(
        ax=ax,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": "Wind speed (m/s)", "shrink": 0.6},
    )
    wind.plot.quiver(
        x="lon", y="lat", u="u10", v="v10", ax=ax, transform=ccrs.PlateCarree()
    )
    ax.set_title("Wind")
    if date in storms.dates:
        storm_ids = storms.dates[date]["storms"]
        for storm_id in storm_ids:
            geom = (
                storms.storms[storm_id]
                .track.iloc[
                    (storms.storms[storm_id].track.index + pd.Timedelta("55min")).floor(
                        "h"
                    )
                    == date
                ]
                .geometry
            )
            if not geom.crs == crs_ortho:
                geom = geom.to_crs(crs_ortho)
            chx_storm, chy_storm = geom.x, geom.y
            ax.plot(chx_storm, chy_storm, color="red")

    gs = GridSpec(2, n_clusters, top=0.5, bottom=0.25)

    for i_cluster in range(n_clusters):
        ax_pdf = fig.add_subplot(
            gs[0, i_cluster],
            sharex=ax_pdf if i_cluster > 0 else None,
            sharey=ax_pdf if i_cluster > 0 else None,
        )
        pdf = []
        cdf = []
        for i_fold in range(len(gev_params)):
            if gev_params[i_fold][i_cluster].shape == (3,):
                mu, sigma, xi = tuple(gev_params[i_fold][i_cluster])
                x = np.linspace(0.0, 50.0, 300)
                pdf.append(
                    gev_pdf(
                        np.repeat(mu, 300), np.repeat(sigma, 300), np.repeat(xi, 300), x
                    )
                )
                cdf.append(
                    gev(
                        np.repeat(mu, 300), np.repeat(sigma, 300), np.repeat(xi, 300), x
                    )
                )
            else:
                for i_station in range(gev_params[i_fold][i_cluster].shape[0]):
                    mu, sigma, xi = tuple(gev_params[i_fold][i_cluster][i_station])
                    x = np.linspace(0.0, 50.0, 300)
                    pdf.append(
                        gev_pdf(
                            np.repeat(mu, 300),
                            np.repeat(sigma, 300),
                            np.repeat(xi, 300),
                            x,
                        )
                    )
                    cdf.append(
                        gev(
                            np.repeat(mu, 300),
                            np.repeat(sigma, 300),
                            np.repeat(xi, 300),
                            x,
                        )
                    )
        xs = np.repeat(np.expand_dims(x, 0), len(pdf), axis=0)
        pdf = pd.DataFrame(
            np.stack((xs, pdf)).reshape(2, -1).T,
            columns=["Wind gust", "Density"],
            dtype=np.float32,
        )
        cdf = pd.DataFrame(
            np.stack((xs, cdf)).reshape(2, -1).T, columns=["Wind gust", "Probability"]
        )
        sns.histplot(
            wind_gusts[i_cluster],
            stat="density",
            ax=ax_pdf,
            color="lightgrey",
            label="Empirical",
        )
        sns.lineplot(
            data=pdf,
            x="Wind gust",
            y="Density",
            ax=ax_pdf,
            label="Theoretical distribution",
            errorbar=lambda x: (x.min(), x.max()),
            err_kws={"alpha": 0.5},
        )
        ax_pdf.set_xlabel(None)
        ax_pdf.set_ylabel(None)
        ax_pdf.set_title(f"Cluster {i_cluster+1}")
        ax_pdf.legend(fontsize="x-small")
        ax_pdf.grid()
        ax_cdf = fig.add_subplot(
            gs[1, i_cluster],
            sharex=ax_cdf if i_cluster > 0 else ax_pdf,
            sharey=ax_cdf if i_cluster > 0 else None,
        )
        sns.ecdfplot(
            wind_gusts[i_cluster],
            ax=ax_cdf,
            stat="proportion",
            color="black",
            label="Empirical",
        )
        sns.lineplot(
            data=cdf,
            x="Wind gust",
            y="Probability",
            ax=ax_cdf,
            label="Theoretical CDF",
            errorbar=lambda x: (x.min(), x.max()),
            err_kws={"alpha": 0.5},
        )
        ax_cdf.set_xlabel(None)
        ax_cdf.set_ylabel(None)
        ax_cdf.legend(fontsize="x-small")
        ax_cdf.grid()

    gs = GridSpec(1, 1, top=0.2, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0])
    if all_crps_s is None or cluster_nb is None:
        crps_df = pd.DataFrame(
            crps_s, columns=[f"Cluster {i}" for i in range(1, n_clusters + 1)]
        )
        sns.boxplot(
            crps_df, ax=ax, palette=colors[:n_clusters], whis=(0, 100), fill=False
        )
        ax.scatter(
            [f"Cluster {i}" for i in range(1, n_clusters + 1)],
            [exp_crps] * n_clusters,
            color="black",
            label="Model CRPS",
        )
        ax.set_ylabel("CRPS [m/s]")
        ax.legend()
        ax.grid()
    else:
        all_crps_s_df = None
        for lt in all_crps_s.keys():
            crps_df = pd.DataFrame(
                all_crps_s[lt], columns=[f"Cluster {i}" for i in range(1, n_clusters + 1)]
            )
            crps_df["Lead time"] = lt
            all_crps_s_df = (
                crps_df
                if all_crps_s_df is None
                else pd.concat([all_crps_s_df, crps_df], ignore_index=True)
            )
        sns.lineplot(
            data=all_crps_s_df,
            x="Lead time",
            y=f"Cluster {cluster_nb}",
            ax=ax,
            label=f"Cluster {cluster_nb}",
            color=colors[cluster_nb - 1],
        )
        crps_df = pd.DataFrame(
            crps_s, columns=[f"Cluster {i}" for i in range(1, n_clusters + 1)]
        )

    fig.savefig(save_path)

    plt.close()


def metrics_evolution(
    model_crps,
    lead_times,
    save_path,
    ptype="lines",
    metrics_name="Metrics",
    style=None,
    style_name="Data",
):
    """
    Plot the evolution of the CRPS during the training for different models.
    
    Parameters
    ----------
    modelCRPS: dict
        Dictionary with the model name as key and the CRPSs as values.
    lead_times: list
        List of lead times.
    save_path: str
        Path where to save the plot.
    ptype: str
        Type of plot. Either "lines" or "rel".
    metrics_name: str
        Name of the metrics (CRPS, CRPSS, RMSE...)
    style: dict
        Dictionary with the model name as key and the style as values.
    style_name: str
        Name of the style if ptype is "rel".
    """
    df = None
    for model in model_crps.keys():
        x = pd.DataFrame(model_crps[model], columns=lead_times)
        x["Fold"] = [f"Fold {i}" for i in range(len(model_crps[model]))]
        x = x.melt(id_vars="Fold", var_name="Lead time", value_name=metrics_name)
        x["Model"] = model
        if style is not None:
            x[style_name] = style[model]
            x["Model"] = model.split(" ")[0]
        df = x if df is None else pd.concat([df, x], ignore_index=True)
    fig = plt.figure(figsize=(7, 4))
    if ptype == "lines":
        sns.lineplot(data=df, x="Lead time", y=metrics_name, hue="Model", style="Fold")
    else:
        sns.lineplot(
            data=df, x="Lead time", y=metrics_name, hue="Model", style=style_name
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def parse_nested_structure(lines, level=0, current_key=None):
    """Parses indented text lines into a nested dictionary."""
    result = {}
    
    while lines:
        line = lines.pop(0)
        stripped = line.strip()
        
        if not stripped or stripped.startswith("#"): 
            # Skip empty lines or comments
            continue
        
        indent_level = len(line) - len(stripped)
        
        if indent_level < level:
            # This line belongs to a higher level of nesting
            lines.insert(0, line)
            break
        
        if stripped.endswith(":"):
            key = stripped[:-1]
            result[key] = parse_nested_structure(lines,
                                                 level=indent_level + 1,
                                                 current_key=key)
        else:
            if current_key:
                if isinstance(result, dict) and not result:
                    # check if the dict is empty, meaning it is a leaf
                    result = parse_line(stripped)
                elif not isinstance(result, list):
                    result = [result, parse_line(stripped)]
                elif isinstance(result, list):
                    result = result + [parse_line(stripped)]
                else:
                    raise ValueError(f"Unexpected line with context: {line}")
            else:
                raise ValueError(f"Unexpected line without context: {line}")
    
    return result

# Load and parse the file
def load_as_nested_dict(filepath):
    """Reads a structured file and parses it into a nested dictionary."""
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return parse_nested_structure(lines)

def parse_line(line):
    try:
        # Try to convert to an integer
        return int(line)
    except ValueError:
        try:
            # Try to convert to a float
            return float(line)
        except ValueError:
            if line == 'None':
                return None
            elif line == 'True':
                return True
            elif line == 'False':
                return False
            else:
                return line.strip()

def write_nested_dict(data, file, level=0):
    """
    Writes a nested dictionary to a file with proper indentation.
    
    Args:
        data (dict): The nested dictionary to write.
        file (file object): Open file object for writing.
        level (int): The current indentation level.
    """
    indent = ' ' * (level * 4)  # Use 4 spaces per indentation level
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                file.write(f"{indent}{key}:\n")
                write_nested_dict(value, file, level + 1)
            else:
                if isinstance(value, jax._src.prng.PRNGKeyArray):
                    # temporary modification to find seed and save it
                    value = str(value).strip().split(' ')[-1].split(']')[0]
                file.write(f"{indent}{key}:\n{indent}    {value}\n")
    elif isinstance(data, (list, tuple, np.ndarray)):
        for item in data:
            if isinstance(item, (dict, list)):
                write_nested_dict(item, file, level)
            else:
                if isinstance(item, jax._src.prng.PRNGKeyArray):
                    # temporary modification to find seed and save it
                    item = str(item).strip().split(' ')[-1].split(']')[0]
                file.write(f"{indent}{item}\n")
    else:
        if isinstance(data, jax._src.prng.PRNGKeyArray):
            # temporary modification to find seed and save it
            data = str(data).strip().split(' ')[-1].split(']')[0]
        file.write(f"{indent}{data}\n")

def save_nested_dict(filepath, data):
    """Saves a nested dictionary to a structured text file."""
    with open(filepath, 'w') as file:
        write_nested_dict(data, file)

def create_crps_from_fcst_ref_dates(long_crps, t_array,
                                year, lead_times, fcst_ref_dates):
    """
    Create the CRPS array from the long CRPS array.
    
    Parameters
    ----------
    long_crps: np.array
        Array of CRPS of shape (n_fold, n_cluster, n_dates), as output by the model.
    t_array: np.array
        Array of shape (n_dates, 4) with the time array.
    year: int
        Year of the forecast.
    lead_times: np.ndarray
        Array of lead times.
    fcst_ref_dates: list
        List of forecast reference dates.
        
    Returns
    -------
    crps: np.array
        Array of CRPS of shape (n_cluster, n_lead_times, n_fold), as output by the model.
    """
    lead_times_6h = [i for i in lead_times if i%6 == 0]
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
    crps = np.full((long_crps.shape[1],
                     len(lead_times_6h),
                     long_crps.shape[0],
                     len(fcst_ref_dates)),
                   np.nan)
    # crps is of shape (n_clusters, n_lead_times, n_fold, n_dates)
    for i_cluster in range(crps.shape[0]):
        for i_lead_time, lead_time in enumerate(lead_times_6h):
            for i_fold in range(crps.shape[2]):
                for i_ref_date, ref_date in enumerate(fcst_ref_dates):
                    date = ref_date + pd.Timedelta(lead_time, unit = 'h')
                    if date in dates:
                        i_date = np.where(dates == date)[0][0] +\
                            np.where(lead_times == lead_time)[0][0]*\
                                (len(t_array)//len(lead_times))
                        crps[i_cluster, i_lead_time, i_fold, i_ref_date] =\
                            long_crps[i_fold, i_cluster, i_date]
    #crps = np.transpose(np.nanmean(crps, axis = 3), (2,0,1))
    return crps
                        