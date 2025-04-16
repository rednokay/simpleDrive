import unittest
import numpy as np

from simpleDrive.tools.transforms import alpha_beta_to_abc

# TODO: Tests for n and c


class TestAlphaBetaToAbc(unittest.TestCase):
    def test_number(self):
        alpha_beta = 0.5 + 1j
        abc = alpha_beta_to_abc(alpha_beta)

        self.assertEqual(abc[0], np.real(alpha_beta))

    def test_ndarray(self):
        phi = np.linspace(0, 2*np.pi, 6)
        alpha = np.cos(phi)
        beta = np.sin(phi)

        alpha_beta = np.array([alpha + 1j*beta])
        abc = alpha_beta_to_abc(alpha_beta)

        print(f"{abc[0]=}")
        print(f"{np.real(alpha_beta)[0]=}")
        np.testing.assert_array_equal(abc[0], np.real(alpha_beta).ravel())
