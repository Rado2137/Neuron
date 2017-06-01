import networkx as nx
import matplotlib.pyplot as plt

class Visualization:

    def __init__(self):
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7), (3, 8), (3, 9), (4, 10),
                          (5, 11), (5, 12), (6, 13)])
        nx.draw_networkx(self.G)
        plt.show()


Visualization()