import networkx as nx


class Visualization:

    def __init__(self):
        self.G = nx.Graph()

    def addNode(self, name, attributes=None):
        self.G.add_node(str(name), attr_dict=attributes)

    def addEdge(self, first, second):
        self.G.add_edge(str(first), str(second))

    def draw(self):
        nx.write_gml(self.G, path="graph.gml")