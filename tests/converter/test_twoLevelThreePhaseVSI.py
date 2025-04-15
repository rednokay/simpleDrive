import unittest

from simpleDrive.converter.twoLevelThreePhaseVSI import ConverterData


class TestTwoLevelThreePhaseVSI(unittest.TestCase):
    def setUp(self):
        self.V_DC = 400
        self.I_max = 500
        self.fs = 10e3

        self.cd = ConverterData(V_DC=self.V_DC, I_max=self.I_max, fs=self.fs)

    def test_instance(self):
        self.assertIsInstance(self.cd, ConverterData)
