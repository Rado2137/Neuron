import csv
import random

from mlp import MLP
from model.Layer import Layer
from model.Neuron import Neuron

global mlp
mlp = MLP()

# initialization of mlp
layer0 = Layer(0)
for i in range(4):
    layer0.addNeuron(Neuron())

# adding input layer
mlp.addLayer(layer0)

layer1 = Layer(1)
for i in range(3):
    layer1.addNeuron(Neuron())

mlp.addLayer(layer1)

# output layer
layer2 = Layer(2)

for i in range(2):
    layer2.addNeuron(Neuron())

mlp.addLayer(layer2)

layer3 = Layer(3)

for i in range(3):
    layer3.addNeuron(Neuron())

mlp.addLayer(layer3)

# initialization of weights
for i in range(0, 3):
    for j in mlp.layers[i].neurons:
        for l in mlp.layers[i + 1].neurons:
            #w = random.uniform(-5.0, 5.0)
            w = random.uniform(-0.1, 0.1)
            j.addOutput(l, w)
            l.addInput(j, w)

def test(iteration):
    with open('testing_data.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        Neuron.recognised = 0
        MLP.recognised = 0
        for row in spamreader:
            i = 0
            for input in mlp.getLayers()[0].getNeurons():
                input.outputValue = float(row[i])
                i = i + 1
                expectedValues = []
            if (row[4] == "Iris-setosa"):
                expectedValues = [1, 0, 0]
            elif (row[4] == "Iris-versicolor"):
                expectedValues = [0, 1, 0]
            else:
                expectedValues = [0, 0, 1]

            #mlp.backPropagation(0.5, expectedValues)
            mlp.backPropagation(0.2, expectedValues)

    print("Test number: " + iteration.__str__() + ".")
    print("Partial effectiveness in %: ")
    print(Neuron.recognised / 3 / 1.50)
    print("Full effectiveness in %: ")
    print(MLP.recognised / 1.50)

for i in range(1, 8):
    test(i)