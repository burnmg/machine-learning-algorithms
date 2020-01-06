import numpy as np


def closest_centroids(distances):
    """
    return the closest centroids' indices for each data point.
    """
    # use broadcasting th compute the distances between each data point and centroid
    return np.argmin(distances, axis=1) # (k, )


def init_centroids(x, k):

    """
    :param x: (data_size, data_dim) np array
    :param k: int
    """
    # TODO(ruolin): I may improve the performance by not generating data_size random numbers. Only k numbers are required
    # This reduces the space and time complexity from data_size to k.
    k_ids = np.random.permutation(x.shape[0])[:k]

    return x[k_ids]


def distance(a, b):

    """

    :param a: (data_size_a, data_dim...)
    :param b: (data_size_b, data_dim...)
    :return: (data_size_a, data_size_b)
    """
    a = a[:, np.newaxis] # pad a dimension for broadcasting

    return np.sqrt(np.sum((a - b)**2, axis=a.shape[2:]))


def kmeans(x, k, epsilon= 1e-5):

    # cluster assignments, an 2D array. i row contains indices of data of cluster i.
    prev_centroids = init_centroids(x, k)

    while True:
        # assign centroids
        distances = distance(x, prev_centroids)

        closest_centroids_ids = closest_centroids(distances)

        assignments = [[] for _ in range(k)]
        for data_idx, c in enumerate(closest_centroids_ids):
            assignments[c].append(data_idx)

        # update centroids
        new_centroids = np.zeros([k, *x.shape[1:]])
        for c, data_ids in enumerate(assignments):
            new_centroids[c] = np.mean(x[data_ids], axis=0)

        if np.all(np.abs((new_centroids - prev_centroids)) < epsilon):
            return new_centroids, closest_centroids_ids
        else:
            prev_centroids = new_centroids



