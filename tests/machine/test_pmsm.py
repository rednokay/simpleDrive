import unittest

from simpleDrive.machine.pmsm import MachineData


class TestPmsm(unittest.TestCase):
    def setUp(self):
        self.Rs = 12e-3
        self.Ld = 1e-3
        self.Lq = 2e-3
        self.psi_f = 15e-3
        self.p = 4

        self.md = MachineData(self.Rs,
                              self.Ld,
                              self.Lq,
                              self.psi_f,
                              self.p)

    def test_instance(self):
        self.assertIsInstance(self.md, MachineData)

    def test_chi(self):
        chi = (self.Lq - self.Ld)/(2*self.Ld)
        self.assertEqual(self.md.chi, chi)
