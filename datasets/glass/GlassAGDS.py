import csv
import numpy as np

from agds.AGDS import AGDS
import utils.Clustering
# Data Set Information:

# 1. Id number: 1 to 214
# 2. RI: refractive index
# 3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 4. Mg: Magnesium
# 5. Al: Aluminum
# 6. Si: Silicon
# 7. K: Potassium
# 8. Ca: Calcium
# 9. Ba: Barium
# 10. Fe: Iron
# 11. Type of glass: (class attribute)
# -- 1 building_windows_float_processed 70
# -- 2 building_windows_non_float_processed 76
# -- 3 vehicle_windows_float_processed 56
# -- 4 vehicle_windows_non_float_processed 1
# -- 5 containers 13
# -- 6 tableware 9
# -- 7 headlamps 29

agds = AGDS()
agds.addParam("RI")
agds.addParam("Na")
agds.addParam("Mg")
agds.addParam("Al")
agds.addParam("Si")
agds.addParam("K")
agds.addParam("Ca")
agds.addParam("Ba")
agds.addParam("Fe")
agds.addParam("class")
agds.addParam("object")

Matrix = []

with open('glass_data.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        agds.addValue("RI", float(row[1]))
        agds.addValue("Na", float(row[2]))
        agds.addValue("Mg", float(row[3]))
        agds.addValue("Al", float(row[4]))
        agds.addValue("Si", float(row[5]))
        agds.addValue("K", float(row[6]))
        agds.addValue("Ca", float(row[7]))
        agds.addValue("Ba", float(row[8]))
        agds.addValue("Fe", float(row[9]))
        agds.addValue("class", float(row[10]))

        Matrix.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10])])

        agds.addObject(i, {'RI': float(row[1]), 'Na': float(row[2]), 'Mg': float(row[3]), 'Al': float(row[4]), 'Si': float(row[5]),
                           'K': float(row[6]), 'Ca': float(row[7]), 'Ba': float(row[8]), 'Fe': float(row[9]), 'class': float(row[10])})
        i += 1

    agds.initWeights("RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe")

# finding similarity to R163
agds.associativeInference("object", 162)
agds.inferenceVisualization()

X = np.array([(Matrix[i][0], Matrix[i][1], Matrix[i][2], Matrix[i][3], Matrix[i][4], Matrix[i][5], Matrix[i][6], Matrix[i][7], Matrix[i][8]) for i in range(214)])
result = utils.Clustering.find_centers(list(X), 7)

print("Mean values of parameters per cluster: ")
for cluster in result[0]:
    print(cluster)

for key, cluster in result[1].items():
    print("Cluster " + str(key + 1))
    print("Number of elements in cluster " + str(cluster.__len__()))
    matches = dict()
    # for element in cluster:
    #     if not matches.__contains__(str(element[9])):
    #         matches[str(element[9])] = 0
    #     matches[str(element[9])] += 1
    # print(matches)

labels = utils.Clustering.spectral(X).labels_
matched = 0
for i in range(labels.__len__()):
    if labels[i] == Matrix[i][9]:
        matched += 1

print("Spectral clustering accuracy: " + str(matched / 214))

labels = utils.Clustering.kmeans(X).labels_
matched = 0
for i in range(labels.__len__()):
    if labels[i] == Matrix[i][9]:
        matched += 1

print("K-means clustering accuracy: " + str(matched / 214))

labels = utils.Clustering.affinity(X).labels_
matched = 0
for i in range(labels.__len__()):
    if labels[i] == Matrix[i][9]:
        matched += 1

print("Affinity Propagation clustering accuracy: " + str(matched / 214))

labels = utils.Clustering.agglomerative(X).labels_
matched = 0
for i in range(labels.__len__()):
    if labels[i] == Matrix[i][9]:
        matched += 1

print("Agglomerative clustering accuracy: " + str(matched / 214))

labels = utils.Clustering.birch(X).labels_
matched = 0
for i in range(labels.__len__()):
    if labels[i] == Matrix[i][9]:
        matched += 1

print("Birch clustering accuracy: " + str(matched / 214))

labels = utils.Clustering.DBSCAN(X).labels_
matched = 0
for i in range(labels.__len__()):
    if labels[i] == Matrix[i][9]:
        matched += 1

print("DBSCAN clustering accuracy: " + str(matched / 214))

labels = utils.Clustering.miniBatchKMeans(X).labels_
matched = 0
for i in range(labels.__len__()):
    if labels[i] == Matrix[i][9]:
        matched += 1

print("Mini Batch K-means clustering accuracy: " + str(matched / 214))

labels = utils.Clustering.meanShift(X).labels_
matched = 0
for i in range(labels.__len__()):
    if labels[i] == Matrix[i][9]:
        matched += 1

print("Mean shift clustering accuracy: " + str(matched / 214))

# cluster.AffinityPropagation([damping, ...]) 	Perform Affinity Propagation Clustering of data.
# cluster.AgglomerativeClustering([...]) 	Agglomerative Clustering
# cluster.Birch([threshold, branching_factor, ...]) 	Implements the Birch clustering algorithm.
# cluster.DBSCAN([eps, min_samples, metric, ...]) 	Perform DBSCAN clustering from vector array or distance matrix.
# cluster.FeatureAgglomeration([n_clusters, ...]) 	Agglomerate features.
# cluster.KMeans([n_clusters, init, n_init, ...]) 	K-Means clustering
# cluster.MiniBatchKMeans([n_clusters, init, ...]) 	Mini-Batch K-Means clustering
# cluster.MeanShift([bandwidth, seeds, ...]) 	Mean shift clustering using a flat kernel.
# cluster.SpectralClustering([n_clusters, ...]) 	Apply clustering to a projection to the normalized laplacian.