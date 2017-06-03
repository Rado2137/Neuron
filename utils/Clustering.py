import random
import numpy as np
import sklearn.cluster as c

def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return mu, clusters

def spectral(X):
    cluster = c.SpectralClustering(n_clusters=7)
    return cluster.fit(X)

def kmeans(X):
    cluster = c.KMeans(n_clusters=7)
    return cluster.fit(X)

def affinity(X):
    cluster = c.AffinityPropagation()
    return cluster.fit(X)

def agglomerative(X):
    cluster = c.AgglomerativeClustering(n_clusters=7)
    return cluster.fit(X)

def birch(X):
    cluster = c.Birch(n_clusters=7)
    return cluster.fit(X)

def DBSCAN(X):
    cluster = c.DBSCAN()
    return cluster.fit(X)

def miniBatchKMeans(X):
    cluster = c.MiniBatchKMeans(n_clusters=7)
    return cluster.fit(X)

def meanShift(X):
    cluster = c.MeanShift()
    return cluster.fit(X)