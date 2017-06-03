from sortedcontainers import SortedDict
import networkx as nx
from model.Neuron import Neuron
from utils.Visualization import Visualization


class AGDS:

    def __init__(self):
        self.paramLayers = dict()
        self.v = Visualization()
        self.v.addNode("param")

    def addParam(self, param):
        self.paramLayers[param] = SortedDict()
        if param != "object":
            self.v.addNode(param)
            self.v.addEdge("param", param)

    def addValue(self, param, value):
        if not self.paramLayers[param].__contains__(value):
            neuron = Neuron()
            neuron.outputValue = value
            neuron.classOfNeuron = param
            self.paramLayers[param][value] = neuron

            self.v.addNode(str(value) + " - " + neuron.classOfNeuron)
            self.v.addEdge(param, str(value) + " - " + neuron.classOfNeuron)

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
        i = 1
        for node in self.paramLayers[param].values():
            node.name = "R" + str(i)
            i += 1
            if node != comparedNode:
                for attribute in node.outputConnections.keys():
                    node.similarity += similarityWeight * attribute.similarity

    def inferenceVisualization(self):
        previous = None
        for node in self.paramLayers["object"].values():
            node.name += " " + str(round(node.similarity, 2))
            self.v.addNode(node.name)
            if previous is not None:
                self.v.addEdge(previous.name, node.name)
            previous = node

            for param in node.outputConnections.keys():
                self.v.addEdge(node.name, str(param.outputValue) + " - " + param.classOfNeuron)

        self.v.draw()
