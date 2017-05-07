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