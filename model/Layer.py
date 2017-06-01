class Layer:

    def __init__(self, number):
        self.layerNumber = number
        self.neurons = []

    def addNeuron(self, neuron):
        self.neurons.append(neuron)

    def removeNeuron(self, neuron):
        self.neurons.remove(neuron)

    def getNeurons(self):
        return self.neurons