import math

from Neuron import Neuron


class SOMNeuron(Neuron):

    def __init__(self):
        self.distance = 0

    def computeDistance(self):
        self.distance = 0
        for k, v in self.inputConnections.items():
            self.distance += math.pow(k.outputValue - v, 2)
        self.distance = math.sqrt(self.distance)

    def updateWeight(self, learningRate, radius):
        for k, v in self.inputConnections.items():
            theta = math.exp(-math.pow(self.distance, 2) / (2 * math.pow(radius, 2)))
            self.inputConnections[k] = v - theta * learningRate * (k.outputValue - v)
