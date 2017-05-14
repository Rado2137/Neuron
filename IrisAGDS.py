import csv

from AGDS import AGDS


agds = AGDS()
agds.addParam("sle")
agds.addParam("swi")
agds.addParam("ple")
agds.addParam("pwi")
agds.addParam("class")

with open('testing_data.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        agds.addValue("sle", float(row[0]))
        agds.addValue("swi", float(row[1]))
        agds.addValue("ple", float(row[2]))
        agds.addValue("pwi", float(row[3]))
        agds.addValue("class", row[4])

    agds.initWeights("sle", "swi", "ple", "pwi")

print(agds.paramLayers)