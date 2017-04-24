class MLP:

    layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def removeLayer(self, layer):
        self.layers.remove(layer)

    def getLayers(self):
        return self.layers

    def backPropagation(self, learningRate, expectedValues):
        for i in self.layers:
            for j in i.neurons:
                j.computeWeightedSum()
                j.computeOutputValue()

        for i in range(0, self.layers[self.layers.__len__() - 1].getNeurons().__len__()):
            self.layers[self.layers.__len__() - 1].getNeurons()
            self.layers[self.layers.__len__() - 1].getNeurons()[i].computeDeltaParameter(expectedValues[i])

        for i in range(self.layers.__len__() - 2, -1, -1):
            for j in self.layers[i].neurons:
                # dummy parameter
                j.computeDeltaParameter(None)
            for j in self.layers[i].neurons:
                #TODO not sure
                j.updateWeight(learningRate)