import csv

from AGDS import AGDS


agds = AGDS()
agds.addParam("sle")
agds.addParam("swi")
agds.addParam("ple")
agds.addParam("pwi")
agds.addParam("class")
agds.addParam("object")

with open('testing_data.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        agds.addValue("sle", float(row[0]))
        agds.addValue("swi", float(row[1]))
        agds.addValue("ple", float(row[2]))
        agds.addValue("pwi", float(row[3]))
        agds.addValue("class", row[4])
        agds.addObject(i, {'sle': float(row[0]), 'swi': float(row[1]), 'ple': float(row[2]), 'pwi': float(row[3]), 'class': row[4]})
        i += 1

    agds.initWeights("sle", "swi", "ple", "pwi")

print(agds.paramLayers)