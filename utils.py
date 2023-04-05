import math
import numpy as np
import matplotlib.pyplot as plt


def distance(x, X):
    '''
    Calculates distance between one point and rest of the points
    inputs:
        x: np.array -> shape (1, 2)
        X: np.array -> shape (num_of_points, 2)
    returns:
        np.array -> shape (num_of_points,)
    '''
    return np.sqrt(((x - X)**2).sum(1))


def gaussian(dist, bandwith):
    return np.exp(-0.5 * ((dist/bandwith))**2) / (bandwith * math.sqrt(2 * math.pi))


def plot(labels):
    colors = 100*['r','g','b','c','k','y']
    for label in labels:
        color = colors[label]
        for point in labels[label]:
            plt.scatter(point[0], point[1], marker='o', color=color, s=50, linewidths=5, zorder=10)
    plt.show()