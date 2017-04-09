import csv
import random

from Layer import Layer
from MLP import MLP
from Neuron import Neuron

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

for i in range(3):
    layer2.addNeuron(Neuron())

mlp.addLayer(layer2)

for i in range(0,1):
    for j in mlp.layers[i].neurons:
        for l in mlp.layers[i + 1].neurons:
            w = random.uniform(0.1, 0.5)
            j.addOutput(l, w)
            l.addInput(j, w)

mlp.layers[0].neurons[0].outputValue = 5.4
mlp.layers[0].neurons[1].outputValue = 4.4
mlp.layers[0].neurons[2].outputValue = 3.4
mlp.layers[0].neurons[3].outputValue = 4.7

mlp.layers[1].neurons[0].computeWeightedSum()
mlp.layers[1].neurons[0].computeOutputValue()
print(mlp.layers[1].neurons[0].outputValue)

with open('testing_data.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        i = 0;
        for input in mlp.getLayers():
            #input.setOutputValue(row[i])
            i = i + 1

            # launching bp algorithm for every entry