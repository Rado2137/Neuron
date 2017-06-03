import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

class Visualization:

    def __init__(self):
        self.G = nx.Graph()
        # self.G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7), (3, 8), (3, 9), (4, 10),
        #                   (5, 11), (5, 12), (6, 13)])
        # nx.draw_networkx(self.G)
        # plt.show()

    def addNode(self, name, attributes=None):
        self.G.add_node(str(name), attr_dict=attributes)

    def addEdge(self, first, second):
        self.G.add_edge(str(first), str(second))

    def draw(self):
        # pos = nx.spring_layout(self.G, scale=20000)
        #
        # nx.draw_networkx(self.G, pos)
        nx.write_gml(self.G, path="../graph.gml")
        # plt.show()