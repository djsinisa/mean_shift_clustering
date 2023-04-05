import utils as utl
import numpy as np


class MeanShift():

    def __init__(self):
        self.centroids = None
        self.labels = None

    def _classify(self, data):
        self.labels = {}
        centroid_points = np.array(list(self.centroids.values()), dtype=float)
        for i in range(centroid_points.shape[0]):
            self.labels[i] = []
        for point in data:
            # compare distaces to either centroid
            # utl.distance(point, centroid_points) has (num_of_centroid_points,)
            # returns index of minimum distance as int. If minimum is not unique, returns first index
            nearest_centroid_idx = np.argmin(utl.distance(point, centroid_points)) 
            self.labels[nearest_centroid_idx].append(point)
    
    def fit(self, data, bandwidth):
        centroids = {}
        # start with all data points as centroids
        for i in range(data.shape[0]):
            centroids[i] = data[i]
        while True:
            new_centroids = []
            for i in centroids:
                # take first point
                centroid = centroids[i]
                # calculate weighted distance to all other points
                dist = utl.distance(centroid, data)
                weight = utl.gaussian(dist, bandwidth)
                tiled_weights = np.tile(weight, [centroid.shape[0], 1])
                denominator = sum(weight)
                new_centroids.append(np.multiply(tiled_weights.transpose(), data).sum(axis=0) / denominator)
            # take unique centroids only
            # round centroids to 2 decimals, this should be parameter since precision depends on dataset
            unique_idx = np.unique(np.array(new_centroids).round(decimals=1), return_index=True, axis=0)
            # filter centroids to take only unique ones but with full precision
            uniques = np.take(np.array(new_centroids), unique_idx[1], axis=0)
            # save previous centrouds
            prev_centroids = dict(centroids)
            centroids = {}
            # fill new centroids
            for i in range(uniques.shape[0]):
                centroids[i] = uniques[i]
            centroid_values = np.array(list(centroids.values()), dtype=float)
            prev_centroids_values = np.array(list(prev_centroids.values()), dtype=float)
            # compare centroids with previous centroids
            if centroid_values.shape[0] == prev_centroids_values.shape[0]:
                # compare centroid vectors if they have same dimension, if not they are not equal
                if np.allclose(centroid_values, prev_centroids_values):  # TODO: pass convergence criteria parameter
                    break
        # final centroids            
        self.centroids = centroids
        self._classify(data)