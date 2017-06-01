from sortedcontainers import SortedDict

from model.Neuron import Neuron


class AGDS:

    def __init__(self):
        self.paramLayers = dict()

    def addParam(self, param):
        self.paramLayers[param] = SortedDict()

    def addValue(self, param, value):
        if not self.paramLayers[param].__contains__(value):
            neuron = Neuron()
            neuron.outputValue = value
            self.paramLayers[param][value] = neuron

    def initWeights(self, *args):
        for param in args:
            for i in range(0, self.paramLayers[param].__len__() - 2):
                weight = 1 - (abs(self.paramLayers[param].keys().__getitem__(i + 1) - self.paramLayers[param].keys().__getitem__(i))) / (self.paramLayers[param].keys().__getitem__(self.paramLayers[param].__len__() - 1) - self.paramLayers[param].keys().__getitem__(0))
                self.paramLayers[param][self.paramLayers[param].keys().__getitem__(i)].addInput(self.paramLayers[param][self.paramLayers[param].keys().__getitem__(i + 1)], weight)
                self.paramLayers[param][self.paramLayers[param].keys().__getitem__(i + 1)].addInput(self.paramLayers[param][self.paramLayers[param].keys().__getitem__(i)], weight)

    def addObject(self, index, classes):
        object = Neuron()
        for k, v in classes.items():
            self.paramLayers[k][v].addOutput(object, 1 / (self.paramLayers[k][v].outputConnections.__len__() + 1))
            object.addOutput(self.paramLayers[k][v], 1)
            self.paramLayers["object"][index] = object
            # updating occurence weigths
            for k1, v1 in self.paramLayers[k][v].outputConnections.items():
                self.paramLayers[k][v].outputConnections[k1] = 1 / self.paramLayers[k][v].outputConnections.__len__()

    def associativeInference(self, param, index):
        self.paramLayers[param][self.paramLayers[param].keys().__getitem__(index)].similarity = 1
        for neuron in self.paramLayers[param][self.paramLayers[param].keys().__getitem__(index)].outputConnections:
            neuron.similarity = 1
            neuron.computeSimilarity(self)

        comparedNode = self.paramLayers[param][self.paramLayers[param].keys().__getitem__(index)]
        similarityWeight = 1 / (self.paramLayers.__len__() - 1)
        for node in self.paramLayers[param].values():
            if node != comparedNode:
                for attribute in node.outputConnections.keys():
                    node.similarity += similarityWeight * attribute.similarity

