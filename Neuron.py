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