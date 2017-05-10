class MLP:

    recognised = 0

    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def removeLayer(self, layer):
        self.layers.remove(layer)

    def getLayers(self):
        return self.layers

    def backPropagation(self, learningRate, expectedValues):
        for i in range(1, self.layers.__len__()):
            for j in self.layers[i].neurons:
                j.computeWeightedSum()
                j.computeOutputValue()

        recognised = 0
        for i in range(0, self.layers[self.layers.__len__() - 1].getNeurons().__len__()):
            recognised += self.layers[self.layers.__len__() - 1].getNeurons()[i].computeDeltaParameter(expectedValues[i])

        if (recognised == 3):
            MLP.recognised += 1

        for i in range(0, self.layers[self.layers.__len__() - 1].getNeurons().__len__()):
            self.layers[self.layers.__len__() - 1].getNeurons()[i].updateWeight(learningRate)

        for i in range(self.layers.__len__() - 2, -1, -1):
            for j in self.layers[i].neurons:
                j.updateWeight(learningRate)
            for j in self.layers[i].neurons:
                # dummy parameter
                if (i > 0):
                    j.computeDeltaParameter(None)
