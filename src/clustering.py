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
        self.data = data
        self.load_affinities(**kwargs)
        self.clusters = {}
        self.plot = self.Plotter(self)
        self.save = self.Saver(self)

    def load_affinities(self, **kwargs):
        aff_file = kwargs.pop("affinities", None)
        if not (aff_file is None):
            try:
                with open(aff_file, "rb") as f:
                    affs = pickle.load(f)
                self.affinities = affs
            except:
                return self.load_affinities(**kwargs)
        else:
            var = kwargs.pop("var", "wind_speed_of_gust")
            nb_of_stations = len(self.data.station.values)
            self.affinities = {}
            if isinstance(var, list):
                for v in var:
                    self.affinities[v] = np.array(
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
                self.affinities[var] = np.array(
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
        Cluster data based on correlation between station, aggregated over time, with Spectral Clustering algorithm.
        Made for the filtered SwissMetNet dataset.
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
            self.clustering = clustering

        def clusters(self, n_clusters, savepath):
            """
            Plot the clusters of the stations.
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
            self.clustering = clustering

        def csv(self, n_clusters, savepath):
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
