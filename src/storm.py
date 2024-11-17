import xarray as xr
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from collections import defaultdict


def plot_stations(
    stations, switzerland_file, save_path, distance=0.0, date=None, station_with_storm=None
):
    """
    Plot the stations on a map of Switzerland.
    
    Parameters
    ----------
    stations : gpd.GeoDataFrame
        The stations to plot.
    switzerland_file : str
        The path to the shapefile of Switzerland.
    save_path : str
        The path where to save the plot.
    distance : float, optional
        The distance around each station to plot a circle. The default is 0.0.
    date : str, optional
        The date of the storm to plot. The default is None.
    station_with_storm : dict, optional
        The stations that detected a storm on a given date. The default is None.
    """
    fig, ax = plt.subplots()

    geom = stations.geometry

    switzerland = gpd.read_file(switzerland_file)

    if not geom.crs == switzerland.crs:
        geom = geom.to_crs(switzerland.crs)

    switzerland.plot(ax=ax)
    geom.plot(ax=ax, color="green")

    # Add a circle around each point from geom, the radius of which is distance.
    for point in geom:
        ax.add_patch(
            plt.Circle(
                (point.x, point.y), distance, color="orange", fill=True, alpha=0.2
            )
        )

    if date:
        if station_with_storm is None:
            raise ValueError("stationWithStorm must be given if date is given")
        detecting_station = list(station_with_storm[date])
        detecting_geom = stations.loc[detecting_station].geometry

        if not detecting_geom.crs == switzerland.crs:
            detecting_geom = detecting_geom.to_crs(switzerland.crs)

        detecting_geom.plot(ax=ax, color="red")

    plt.title(
        f"Stations "
        + f"which detected a storm on {date} " * (date is not None)
        + f"with a radius of {distance/1000} km" * (distance != 0.0)
    )
    plt.xlabel("CH1903+ x (m)")
    plt.ylabel("CH1903+ y (m)")

    plt.savefig(save_path)


class Storm:

    def __init__(self, data, storm_id, stations, **kwargs):
        """
        Create a storm object, which contains the track of the storm and the stations that detected it.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The data from which to extract the storm's track, made from Monika's article.
            data should have ID as index.
        storm_id : str
            The ID of the storm.
        stations : array like with three columns: name, chx, chy
            The stations that detected the storm.
        **kwargs : dict
            Additional arguments.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")
        if not isinstance(storm_id, str):
            raise TypeError("ID must be a str")
        if not storm_id in data.index:
            raise ValueError("ID must be in data.index")

        self.storm_id = storm_id
        self.track = (
            data[["time", "chx", "chy"]].loc[[self.storm_id]].reset_index().drop("ID", axis=1)
        )
        self.track = gpd.GeoDataFrame(
            self.track,
            geometry=gpd.points_from_xy(self.track.chx, self.track.chy),
            crs="EPSG:21781",
        )
        self.track = self.track.set_index("time")

        self.define_stations(stations, **kwargs)

        self.empty_track = len(self.track) == 0

    def define_stations(self, stations, distance=25000, **kwargs):
        """
        Define the stations that are in the storm's track.
        
        Parameters
        ----------
        stations : gpd.GeoDataFrame
            The stations to consider. The names of the stations should be the index.
        distance : float, optional
            The distance around each point to consider a station. The default is 25000.
        """
        if not isinstance(stations, gpd.GeoDataFrame):
            raise TypeError("stations must be a gpd.GeoDataFrame")

        nums = [stations.distance(point).argmin() for point in self.track.geometry]
        stations = stations.iloc[nums].reset_index()
        distances = pd.Series(
            [
                station.distance(point)
                for station, point in zip(stations.geometry, self.track.geometry)
            ],
            index=self.track.index,
        )

        self.stations = gpd.GeoDataFrame(
            stations, geometry=stations.geometry, crs="EPSG:21781"
        )
        self.stations.index = self.track.index
        self.stations["distances"] = distances

        self.stations = self.stations.where(self.stations.distances < distance)

        self.track = self.track.where(self.stations.distances < distance).dropna()
        self.stations = self.stations.dropna()

    def __str__(self):
        return f"Storm {self.storm_id}"


class Storms:
    def __init__(self, data, stations, keep_empty=False, **kwargs):
        """
        Create a storms object, which contains all the storms and the stations that detected them.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The data from which to extract the storms, made from Monika's article.
            data should have ID as index.
        stations : array like with three columns: name, chx, chy
            The stations that detected the storms.
        keep_empty : bool, optional
            Whether to keep the storms with an empty track. The default is False.
        **kwargs : dict
            Additional arguments.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")

        self.storms = {
            storm_id: Storm(data, storm_id, stations, **kwargs) for storm_id in data.index.unique()
        }
        self.storms = {
            ID: self.storms[ID]
            for ID in self.storms.keys()
            if (not self.storms[ID].empty_track or keep_empty)
        }

        self.distance = kwargs.get("distance", 25000)

        self.keep_empty = keep_empty

        presence = defaultdict(set)
        storm_ids = defaultdict(set)
        for storm_id in self.storms:
            storm = self.storms[storm_id]
            for time in storm.stations.index:
                time_rounded = (time + pd.Timedelta("55min")).floor("h")
                presence[time_rounded].add(storm.stations.loc[time].station)
                storm_ids[time_rounded].add(storm_id)
        self.dates = {
            time: {"stations": list(presence[time]), "storms": list(storm_ids[time])}
            for time in presence
        }

    def __str__(self):
        return self.storms.__str__()
