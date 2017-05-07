from SOMNeuron import SOMNeuron


class SOMGrid:

    def __init__(self, height, width, startingRadius, narrowing, initLearningRate):
        self.width = width
        self.height = height
        self.startingRadius = startingRadius
        self.narrowing = narrowing
        self.initLearningRate = initLearningRate
        self.matrix = [[SOMNeuron() for x in range(width)] for y in range(height)]

    def chooseWinner(self):
        winnerCoords = (0, 0)
        for i in range(self.height):
            for j in range(self.width):
                if self.matrix[i][j].distance < self.matrix[winnerCoords[0]][winnerCoords[1]].distance:
                    winnerCoords = (i, j)
        winnerCoords
