import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from sklearn.cluster import SpectralClustering
from sklearn import metrics


class Clustering:
    def __init__(self, data, **kwargs):
        """
        Class to cluster stations based on meteorological data.
        
        Parameters
        ----------
        data: xarray.Dataset
            Dataset of meteorological data. Should have the following dimensions:
            - station: the station ID
            - time: the time of the observation
            - latitude: the latitude of the station
            - longitude: the longitude of the station
            - the meteorological variables to cluster
        correlations: str, optional
            Path to the file containing the correlations between stations. If not
            provided, the correlations are computed from the data as the correlation
            between stations for the meteorological variable provided in the `var`
            argument.
        var: str or list of str, optional
            The meteorological variable to use to compute the correlations. If a list is
            provided, the correlations are computed for each variable. Default is
            "wind_speed_of_gust".
        """
        self.data = data
        self.load_correlations(**kwargs)
        self.clusters = {}
        self.plot = self.Plotter(self)
        self.save = self.Saver(self)

    def load_correlations(self, **kwargs):
        """
        Load the correlations between stations from a file or compute them from the data.
        
        Parameters
        ----------
        correlations: str, optional
            Path to the file containing the correlations between stations. If not
            provided, the correlations are computed from the data as the correlation
            between stations for the meteorological variable provided in the `var`
            argument.
        var: str or list of str, optional
            The meteorological variable to use to compute the correlations. If a list is
            provided, the correlations are computed for each variable. Default is
            "wind_speed_of_gust".
        """
        correlation_file = kwargs.pop("correlations", None)
        if not (correlation_file is None):
            try:
                with open(correlation_file, "rb") as f:
                    corrs = pickle.load(f)
                self.correlations = corrs
            except:
                return self.load_correlations(**kwargs)
        else:
            var = kwargs.pop("var", "wind_speed_of_gust")
            nb_of_stations = len(self.data.station.values)
            self.correlations = {}
            if isinstance(var, list):
                for v in var:
                    self.correlations[v] = np.array(
                        [
                            [
                                xr.corr(
                                    self.data[v].isel(station=i),
                                    self.data[v].isel(station=j),
                                    dim="time",
                                ).values
                                for i in range(nb_of_stations)
                            ]
                            for j in range(nb_of_stations)
                        ]
                    )
            else:
                self.correlations[var] = np.array(
                    [
                        [
                            xr.corr(
                                self.data[var].isel(station=i),
                                self.data[var].isel(station=j),
                                dim="time",
                            ).values
                            for i in range(nb_of_stations)
                        ]
                        for j in range(nb_of_stations)
                    ]
                )

    def spectral_station_clustering(self, n_clusters, affinities):
        """
        Cluster data based on correlation between station, aggregated over time,
        with Spectral Clustering algorithm. Made for the filtered SwissMetNet dataset.
        
        Parameters
        ----------
        n_clusters: int
            Number of clusters to create.
        affinities: np.array
            Affinities between stations. Should be a square matrix of shape
            (n_stations, n_stations) containing positive values. Values could be
            absolute correlations but are not limited to that. Example used in the
            article is:
            np.abs(self.correlations["wind_speed_of_gust"]*self.correlations["precipitation"])
        
        Returns
        -------
        labels: xr.DataArray
            Labels of the stations for each cluster. Shape (n_stations,).
        score: float
            Silhouette score of the clustering.
        """
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            n_init=500,
            assign_labels="discretize",
            affinity="precomputed",
        ).fit(affinities)
        labels = xr.DataArray(
            clustering.labels_, coords={"station": self.data.station}, dims=["station"]
        )
        distances = np.clip(-np.log(affinities), a_min=0, a_max=1e8)
        np.fill_diagonal(distances, 0.0)
        distances = np.nan_to_num(distances, 0.0)
        self.clusters[n_clusters] = {
            "labels": labels,
            "score": metrics.silhouette_score(
                distances, clustering.labels_, metric="precomputed"
            ),
        }
        return self.clusters[n_clusters]["labels"], self.clusters[n_clusters]["score"]

    class Plotter:
        def __init__(self, clustering):
            """
            Class to plot the results of the clustering.
            
            Parameters
            ----------
            clustering: Clustering
                Clustering object.
            """
            self.clustering = clustering

        def clusters(self, n_clusters, savepath):
            """
            Plot the clusters of the stations on a map.
            
            Parameters
            ----------
            n_clusters: int
                Number of clusters to plot.
            savepath: str
                Path to save the figure
            """
            google_tiles = cimgt.GoogleTiles(style="satellite")
            fig, ax = plt.subplots(subplot_kw={"projection": google_tiles.crs})
            ax.add_image(google_tiles, 9, alpha=0.5)
            ax.gridlines(draw_labels=True, auto_inline=True)
            ax.add_feature(cfeature.BORDERS)
            ax.set_extent([5.8, 10.7, 45.7, 47.8], crs=ccrs.PlateCarree())
            ax.scatter(
                self.clustering.data.longitude.isel(time=0).values,
                self.clustering.data.latitude.isel(time=0).values,
                c=self.clustering.clusters[n_clusters]["labels"].values,
                cmap="inferno",
                transform=ccrs.PlateCarree(),
            )
            ax.set_title(f"Clustering for {n_clusters} clusters")
            fig.tight_layout()
            plt.savefig(savepath)
            plt.close()

        def silhouette_score(self, save_path):
            """
            Plot the silhouette score of the clustering for different number
            of clusters.
            
            Parameters
            ----------
            save_path: str
                Path to save the figure.
            """
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(
                [n for n in self.clustering.clusters.keys()],
                [
                    self.clustering.clusters[n]["score"]
                    for n in self.clustering.clusters.keys()
                ],
            )
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Silhouette score")
            ax.set_title("Spectral clustering")
            ax.grid()
            fig.tight_layout()
            plt.savefig(save_path)
            plt.close()

    class Saver:
        def __init__(self, clustering):
            """
            Class to save the results of the clustering.
            
            Parameters
            ----------
            clustering: Clustering
                Clustering object.
            """
            self.clustering = clustering

        def csv(self, n_clusters, savepath):
            """
            Save the clusters of the stations in a csv file suitable for the
            usage by postprocessing scripts.
            
            Parameters
            ----------
            n_clusters: int
                Number of clusters with which spectral clustering was performed.
            savepath: str
                Path to save the csv file.
            """
            clusterds = self.clustering.clusters[n_clusters]["labels"]
            clusterlist = [
                list(clusterds.where(clusterds == i).dropna("station").station.values)
                for i in range(n_clusters)
            ]
            # Complete the list with empty string to have the same length
            max_length = max([len(cluster) for cluster in clusterlist])
            for cluster in clusterlist:
                cluster += [""] * (max_length - len(cluster))
            df = pd.DataFrame(clusterlist)
            df.to_csv(savepath, header=False, index=False)
