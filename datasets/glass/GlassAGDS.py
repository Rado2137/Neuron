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
# -- 1 building_windows_float_processed
# -- 2 building_windows_non_float_processed
# -- 3 vehicle_windows_float_processed
# -- 4 vehicle_windows_non_float_processed
# -- 5 containers
# -- 6 tableware
# -- 7 headlamps

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
        agds.addValue("class", row[10])

        Matrix.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9])])

        agds.addObject(i, {'RI': float(row[1]), 'Na': float(row[2]), 'Mg': float(row[3]), 'Al': float(row[4]), 'Si': float(row[5]),
                           'K': float(row[6]), 'Ca': float(row[7]), 'Ba': float(row[8]), 'Fe': float(row[9]), 'class': row[10]})
        i += 1

    agds.initWeights("RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe")

# finding similarity to R163
agds.associativeInference("object", 162)

agds.inferenceVisualization()
X = np.array([(Matrix[i][0], Matrix[i][1], Matrix[i][2], Matrix[i][3], Matrix[i][4], Matrix[i][5], Matrix[i][6], Matrix[i][7], Matrix[i][8]) for i in range(214)])

utils.Clustering.find_centers(list(X), 7)