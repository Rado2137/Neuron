import unittest

from model.Neuron import Neuron


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = Neuron()
        x.addInput(Neuron())
        y = Neuron()
        x.addOutput(y, 1)
        x.outputValue = 6

        self.assertEqual(len(x.inputConnections), 1)
        self.assertEqual(len(x.outputConnections), 1)
        x.removeOutput(y)
        self.assertEqual(len(x.outputConnections), 0)
        self.assertEqual(x.outputValue, 6)

if __name__ == '__main__':
    unittest.main()