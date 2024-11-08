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

def plotStations(stations, SwitzerlandFile, savepath,
                 distance = 0., date = None, stationWithStorm = None):
    """
    stations: geoDataFrame with the stations to plot.
    """
    fig, ax = plt.subplots()
    
    geom = stations.geometry
    
    switzerland = gpd.read_file(SwitzerlandFile)
    
    if not geom.crs == switzerland.crs:
        geom = geom.to_crs(switzerland.crs)
    
    switzerland.plot(ax = ax)
    geom.plot(ax = ax, color = "green")
    
    # Add a circle around each point from geom, the radius of which is distance.
    for point in geom:
        ax.add_patch(plt.Circle((point.x, point.y), distance, color = "orange", fill = True, alpha = 0.2))
    
    if date:
        if stationWithStorm is None:
            raise ValueError("stationWithStorm must be given if date is given")
        detectingStations = list(stationWithStorm[date])
        detectingGeom = stations.loc[detectingStations].geometry
        
        if not detectingGeom.crs == switzerland.crs:
            detectingGeom = detectingGeom.to_crs(switzerland.crs)
            
        detectingGeom.plot(ax = ax, color = "red")
    
    plt.title(f"Stations " + f"which detected a storm on {date} "*(date is not None) + f"with a radius of {distance/1000} km"*(distance != 0.))
    plt.xlabel("CH1903+ x (m)")
    plt.ylabel("CH1903+ y (m)")
    
    plt.savefig(savepath)
        
class Storm:
    
    def __init__(self, data, ID, stations, **kwargs):
        """
        data: pandas.DataFrame, made from csv dataset from Monika's article from which to extract information about the storms.
        data should have ID as index.
        id: str, id of the storm in the dataset.
        stations: array like with three columns: name, chx, chy.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")
        if not isinstance(ID, str):
            raise TypeError("ID must be a str")
        if not ID in data.index:
            raise ValueError("ID must be in data.index")
        
        self.ID = ID
        self.track = data[["time", "chx", "chy"]].loc[[self.ID]].reset_index().drop("ID", axis = 1)
        self.track = gpd.GeoDataFrame(
            self.track, geometry = gpd.points_from_xy(self.track.chx, self.track.chy), crs = "EPSG:21781"
            )
        self.track = self.track.set_index("time")
        
        self.defineStations(stations, **kwargs)
        
        self.emptyTrack = (len(self.track) == 0)
    
    def defineStations(self, stations, distance = 25000, **kwargs):
        """
        Define the stations that are in the storm's track.
        stations: gpd.GeoDataFrame with the names of the stations as index. Coordinates should
        """
        if not isinstance(stations, gpd.GeoDataFrame):
            raise TypeError("stations must be a gpd.GeoDataFrame")
        
        nums = [stations.distance(point).argmin() for point in self.track.geometry]
        stations = stations.iloc[nums].reset_index()
        distances = pd.Series([station.distance(point) for station, point in zip(stations.geometry, self.track.geometry)], index = self.track.index)  
        
        self.stations = gpd.GeoDataFrame(
            stations, geometry = stations.geometry, crs = "EPSG:21781"
            )
        self.stations.index = self.track.index
        self.stations["distances"] = distances
        
        self.stations = self.stations.where(self.stations.distances < distance)
        
        self.track = self.track.where(self.stations.distances < distance).dropna()
        self.stations = self.stations.dropna()
    
    def __str__(self):
        return f"Storm {self.ID}"
    
class Storms:
    def __init__(self, data, stations, keepEmpty = False, **kwargs):
        """
        data: pandas.DataFrame, made from csv dataset from Monika's article from which to extract information about the storms.
        data should have ID as index.
        stations: array like with three columns: name, chx, chy.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")
        
        self.storms = {ID:Storm(data, ID, stations, **kwargs) for ID in data.index.unique()}
        self.storms = {ID:self.storms[ID] for ID in self.storms.keys() if (not self.storms[ID].emptyTrack or keepEmpty)}
        
        self.distance = kwargs.get("distance", 25000)
        
        self.keepEmpty = keepEmpty
    
        presence = defaultdict(set)
        stormids = defaultdict(set)
        for ID in self.storms:
            storm = self.storms[ID]
            for time in storm.stations.index:
                time_rounded = (time + pd.Timedelta("55min")).floor("h")
                presence[time_rounded].add(storm.stations.loc[time].station)
                stormids[time_rounded].add(ID)
        self.dates = {time:{"stations":list(presence[time]), "storms":list(stormids[time])} for time in presence}
        
    
    def __str__(self):
        return self.storms.__str__()