import math

from SOMNeuron import SOMNeuron


class SOMGrid:

    def __init__(self, height, width, initLearningRate):
        self.width = width
        self.height = height
        self.startingRadius = height / 2 if height > width else width / 2
        self.narrowing = 1000
        self.initLearningRate = initLearningRate
        self.matrix = [[SOMNeuron() for x in range(width)] for y in range(height)]

    def chooseWinner(self):
        #TODO to delete
        winnerCoords = (0, 0)
        for i in range(self.height):
            for j in range(self.width):
                if self.matrix[i][j].distance < self.matrix[winnerCoords[0]][winnerCoords[1]].distance:
                    winnerCoords = (i, j)
        winnerCoords

    def doLearnStep(self, iteration):
        radius = self.startingRadius * math.exp(- iteration / self.narrowing)
        learningRate = self.initLearningRate * math.exp(- iteration / self.narrowing)

        #TODO it needs to be computed only once
        for i in range(self.height):
            for j in range(self.width):
                self.matrix[i][j].computeDistance()

        for i in range(self.height):
            for j in range(self.width):
                self.matrix[i][j].updateWeight(learningRate, radius)