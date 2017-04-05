class Neuron:

    def __init__(self):
        self.weightedSum = 0
        self.deltaParameter = 0
        self.outputValue = 0
        self.inputConnections = []
        self.outputConnections = dict()

    def addInput(self, neuron):
        self.inputConnections.append(neuron)

    def removeInput(self, neuron):
        self.inputConnections.remove(neuron)

    def addOutput(self, neuron, weight):
        self.outputConnections[neuron] = weight

    def removeOutput(self, neuron):
        self.outputConnections.pop(neuron)