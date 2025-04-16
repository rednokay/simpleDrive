import unittest
import numpy as np
from simpleDrive.tools.coordinates import AlphaBetaCoordinates
from simpleDrive.tools.transforms import alpha_beta_to_abc


class TestAlphaBetaCoordinates(unittest.TestCase):
    def test_init_single_number_ndarray(self):
        alpha_beta_input = np.array([0.25 + 1j])
        alpha_beta = AlphaBetaCoordinates(alpha_beta_input)

        np.testing.assert_array_equal(alpha_beta_input, alpha_beta.alpha_beta)

    def test_init_single_number(self):
        alpha_beta_input = 0.25 + 1j
        alpha_beta = AlphaBetaCoordinates(alpha_beta_input)

        np.testing.assert_array_equal(
            np.array(alpha_beta_input), alpha_beta.alpha_beta)

    def test_init_single_number_list(self):
        alpha_beta_input = [0.25 + 1j]
        alpha_beta = AlphaBetaCoordinates(alpha_beta_input)

        np.testing.assert_array_equal(
            np.array(alpha_beta_input), alpha_beta.alpha_beta)

    def test_init_ndarray(self):
        alpha = [1, -5, 3, 0]
        beta = [-5, 0, 1, 9]

        alpha_beta_input = np.array(alpha) + 1j*np.array(beta)
        alpha_beta = AlphaBetaCoordinates(alpha_beta_input)

        np.testing.assert_array_equal(alpha_beta_input, alpha_beta.alpha_beta)

    def test_init_real_imag_lists(self):
        alpha = [1, -5, 3, 0]
        beta = [-5, 0, 1, 9]

        alpha_beta_input = np.array(alpha) + 1j*np.array(beta)
        alpha_beta = AlphaBetaCoordinates(alpha, beta)

        np.testing.assert_array_equal(alpha_beta_input, alpha_beta.alpha_beta)

        alpha = [1, -5, 3]
        beta = [-5, 0, 1, 9]

        self.assertRaises(ValueError, AlphaBetaCoordinates, alpha, beta)

    def test_init_real_imag_singular_numbers(self):
        alpha = -2
        beta = 8

        alpha_beta_input = np.array([alpha + 1j*beta])
        alpha_beta = AlphaBetaCoordinates(alpha, beta)

        np.testing.assert_array_equal(alpha_beta_input, alpha_beta.alpha_beta)
    
    def test_init_real_imag_ndarray(self):
        alpha = np.array([1, -5, 3, 0])
        beta = np.array([-5, 0, 1, 9])

        alpha_beta_input = alpha + 1j*beta
        alpha_beta = AlphaBetaCoordinates(alpha, beta)

        np.testing.assert_array_equal(alpha_beta_input, alpha_beta.alpha_beta)

        alpha = np.array([1, 3, 0])
        beta = np.array([-5, 0, 1, 9])

        self.assertRaises(ValueError, AlphaBetaCoordinates, alpha, beta)

    def test_init_unequal_types(self):
        alpha = np.array([1, -5, 3, 0])
        beta = [-5, 0, 1, 9]

        self.assertRaises(ValueError, AlphaBetaCoordinates, alpha, beta)
    
    def setUp(self):
        self.phi = np.linspace(0, 2*np.pi, 10)
        self.alpha = np.cos(self.phi)
        self.beta = np.sin(self.phi)
        self.alpha_beta = AlphaBetaCoordinates(self.alpha, self.beta)

    def test_to_abc(self):
        abc_ref = alpha_beta_to_abc(self.alpha_beta.alpha_beta)
        np.testing.assert_array_equal(abc_ref, self.alpha_beta.to_abc())