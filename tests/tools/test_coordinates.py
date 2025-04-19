import unittest
import numpy as np
from simpleDrive.tools.coordinates import AlphaBetaCoordinates, AbcCoordinates, alpha_beta_to_abc, abc_to_alpha_beta


class TestAlphaBetaToAbc(unittest.TestCase):
    def setUp(self):
        alpha = [1, 0.2, 3]
        beta = [-1, 3.2, 1]
        self.alpha_beta = AlphaBetaCoordinates(alpha, beta)

    def test_instance(self):
        self.assertTrue(isinstance(alpha_beta_to_abc(
            self.alpha_beta), AbcCoordinates))

    # TODO: test b and c phases
    def test_number(self):
        alpha_beta = AlphaBetaCoordinates(1 + 2j)
        abc = alpha_beta_to_abc(alpha_beta)

        self.assertEqual(abc.abc[0], np.real(alpha_beta.alpha_beta))

    # TODO: test b and c phases
    def test_ndarray(self):
        abc = alpha_beta_to_abc(self.alpha_beta)

        np.testing.assert_array_equal(
            abc.abc[0], np.real(self.alpha_beta.alpha_beta).ravel())


class TestAbcToAlphaBeta(unittest.TestCase):
    def setUp(self):
        a = [1, 3, -1]
        b = [-1, 5, 0]
        c = [7, 3, 3]
        self.abc = AbcCoordinates(a, b, c)

    def test_instance(self):
        self.assertTrue(isinstance(
            abc_to_alpha_beta(self.abc), AlphaBetaCoordinates))
    
    def test_ndarray(self):
        alpha_beta = abc_to_alpha_beta(self.abc)
        np.testing.assert_array_equal(np.real(alpha_beta.alpha_beta).ravel(), self.abc.abc[0])


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
        abc_ref = alpha_beta_to_abc(self.alpha_beta)
        abc = self.alpha_beta.to_abc()

        self.assertTrue(isinstance(abc, AbcCoordinates))
        np.testing.assert_array_equal(abc_ref.abc, abc.abc)


class TestAbcCoordinates(unittest.TestCase):
    def test_init_single_ndarray(self):
        abc_ref = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
        abc = AbcCoordinates(abc_ref)

        np.testing.assert_array_equal(abc_ref, abc.abc)

    def test_init_single_list(self):
        abc_ref = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
        abc = AbcCoordinates(abc_ref)

        np.testing.assert_array_equal(np.array(abc_ref), abc.abc)

    def test_init_mixed(self):
        al = [1, 2, 3]
        bl = [3, 1, 2]
        cl = [2, 3, 1]

        a = np.array(al)
        c = np.array(cl)

        abc_ref = np.array([al, bl, cl])

        abc = AbcCoordinates(a, bl, c)

        np.testing.assert_array_equal(np.array(abc_ref), abc.abc)

        self.assertRaises(ValueError, AbcCoordinates, 1, bl, cl)

    def test_wrong_amount_of_input_variables(self):
        al = [1, 2, 3]
        bl = [3, 1, 2]

        self.assertRaises(ValueError, AbcCoordinates, al, bl)
