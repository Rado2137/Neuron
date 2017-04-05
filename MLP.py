class MLP:

    layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def removeLayer(self, layer):
        self.layers.remove(layer)

    def getLayers(self):
        return self.layers