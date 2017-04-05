import math


class Neuron:

    def __init__(self):
        self.weightedSum = 0
        self.deltaParameter = 0
        self.outputValue = 0
        self.inputConnections = dict()
        self.outputConnections = []

    def addOutput(self, neuron):
        self.outputConnections.append(neuron)

    def removeOutput(self, neuron):
        self.outputConnections.remove(neuron)

    def addInput(self, neuron, weight):
        self.inputConnections[neuron] = weight

    def removeInput(self, neuron):
        self.inputConnections.pop(neuron)

    def computeWeightedSum(self):
        for k, v in self.inputConnections.items():
            self.weightedSum = self.weightedSum + k.outputValue * v

    def computeOutputValue(self):
        self.outputValue = (1 / (1 + math.pow(2.72, self.weightedSum)))

    def computeDeltaParameter(self, expectedValue):
        # when neuron is in last layer
        self.deltaParameter = expectedValue - self.outputValue

    #def updateWeight(self):
