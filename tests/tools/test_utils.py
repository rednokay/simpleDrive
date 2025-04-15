from scipy.constants import pi
import unittest

from simpleDrive.machine.pmsm import MachineData
from simpleDrive.tools.utils import n_to_omega_electric


class TestUtils(unittest.TestCase):
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

    def test_n_to_omega_electric(self):
        n = 2750
        omega = 2*pi*self.p*n/60
        self.assertEqual(n_to_omega_electric(n, self.md), omega)
