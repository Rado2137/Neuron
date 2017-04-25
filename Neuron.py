import math

class Neuron:

    recognised = 0

    def __init__(self):
        self.weightedSum = 0
        self.deltaParameter = 0
        self.outputValue = 0
        self.inputConnections = dict()
        self.outputConnections = dict()

    def addOutput(self, neuron, weight):
        self.outputConnections[neuron] = weight

    def removeOutput(self, neuron):
        self.outputConnections.pop(neuron)

    def addInput(self, neuron, weight):
        self.inputConnections[neuron] = weight

    def removeInput(self, neuron):
        self.inputConnections.pop(neuron)

    def computeWeightedSum(self):
        self.weightedSum = 0
        for k, v in self.inputConnections.items():
            self.weightedSum = self.weightedSum + k.outputValue * v

    def computeOutputValue(self):
        self.outputValue = (1.0 / (1.0 + math.exp(-self.weightedSum)))

    def computeDeltaParameter(self, expectedValue):
        # when neuron is in last layer
        if (self.outputConnections.__len__() == 0):
            self.deltaParameter = (expectedValue - self.outputValue) * self.outputValue * (1.0 - self.outputValue)
            if (math.fabs(self.deltaParameter) < 0.1):
                Neuron.recognised += 1
                return 1
            else:
                return 0
        else:
            self.deltaParameter = 0
            for k, v in self.outputConnections.items():
                self.deltaParameter = self.deltaParameter + k.deltaParameter * v * (1 - k.outputValue) * k.outputValue
            return -1

    def updateWeight(self, learningRate):
        for k, v in self.outputConnections.items():
            self.outputConnections[k] = v - learningRate * k.deltaParameter * (1 - k.outputValue) * k.outputValue * self.outputValue
            k.inputConnections[self] = self.outputConnections[k]