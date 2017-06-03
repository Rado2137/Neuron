import csv
import random

import numpy as np

from agds.AGDS import AGDS
import utils.Clustering

agds = AGDS()
agds.addParam("sle")
agds.addParam("swi")
agds.addParam("ple")
agds.addParam("pwi")
agds.addParam("class")
agds.addParam("object")

Matrix = []

with open('../testing_data.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        agds.addValue("sle", float(row[0]))
        agds.addValue("swi", float(row[1]))
        agds.addValue("ple", float(row[2]))
        agds.addValue("pwi", float(row[3]))
        agds.addValue("class", row[4])

        Matrix.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])

        agds.addObject(i, {'sle': float(row[0]), 'swi': float(row[1]), 'ple': float(row[2]), 'pwi': float(row[3]), 'class': row[4]})
        i += 1

    agds.initWeights("sle", "swi", "ple", "pwi")

# finding similarity to R93
agds.associativeInference("object", 92)
#agds.associativeInference("object", 7)

# for item in agds.paramLayers["object"].values():
#     print(item.similarity)

agds.inferenceVisualization()
X = np.array([(Matrix[i][0], Matrix[i][1], Matrix[i][2], Matrix[i][3]) for i in range(150)])


utils.Clustering.find_centers(list(X), 3)