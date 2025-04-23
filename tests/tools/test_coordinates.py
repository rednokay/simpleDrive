import pytest
import numpy as np
import simpleDrive.tools.coordinates as crds


@pytest.fixture
def sample_alpha_beta():
    alpha = [1, 0.2, 3]
    beta = [-1, 3.2, 1]
    return crds.AlphaBetaCoordinates(alpha, beta)


@pytest.fixture
def sample_abc():
    a = [1, 3, -1]
    b = [-1, 5, 0]
    c = [0, -8, 1]
    return crds.AbcCoordinates(a, b, c)


@pytest.fixture
def sample_cmplx():
    return crds.ComplexValuedCoordinates([1, -5, 0], [0, 10, -4])


@pytest.fixture
def sample_dq():
    d = [1, 0.2, 3]
    q = [-1, 3.2, 1]
    return crds.DqCoordinates(d, q)


# TODO: Add some test for B and C phase
class TestAlphaBetaToAbc():
    def test_instance(self, sample_alpha_beta):
        transform = crds.alpha_beta_to_abc(sample_alpha_beta)
        assert isinstance(transform, crds.AbcCoordinates)

    # Test multiple inputs because of the matrix multiplication
    @pytest.mark.parametrize(
        "args,expected_a",
        [
            (5 - 1j*3, [5]),
            (np.array([1, -3, 0] + 1j*np.array([-1, 0, 9])), [1, -3, 0])
        ],
        ids=["complex_scalar", "complex_ndarray"]
    )
    def test_correct_transform(self, args, expected_a):
        alpha_beta = crds.AlphaBetaCoordinates(args)
        a = crds.alpha_beta_to_abc(alpha_beta).abc[0]
        assert a.tolist() == expected_a


# TODO: Add some test for B and C phase
class TestAbcToAlphaBeta():
    def test_instance(self, sample_abc):
        transform = crds.abc_to_alpha_beta(sample_abc)
        assert isinstance(transform, crds.AlphaBetaCoordinates)

    # TODO: Add parameters like above when ABC can be created from three scalars
    def test_correct_transform(self, sample_abc):
        alpha = crds.abc_to_alpha_beta(sample_abc).real
        assert np.array_equal(alpha, sample_abc.abc[0])


class TestComplexValuedCoordinates():
    @pytest.mark.parametrize(
        "args,expected_values",
        [
            ((np.array([1, -3, 0] + 1j*np.array([-1, 0, 9])),),
             np.array([1, -3, 0] + 1j*np.array([-1, 0, 9]))),

            ((5 - 1j*10,), np.array([5]) + 1j*np.array([-10])),

            (([1 - 1j*4, -4 - 1j*1, 3, 1j*6],),
             np.array([1, -4, 3, 0]) + 1j*np.array([-4, -1, 0, 6])),

            ((4, -1), np.array([4]) + 1j*np.array([-1])),

            ((np.array([1, -4, 0]), np.array([0, -4, 50])),
             np.array([1, -4, 0]) + 1j*np.array([0, -4, 50])),

            (([1, -4, 0], [0, -4, 50]),
             np.array([1, -4, 0]) + 1j*np.array([0, -4, 50]))
        ],
        ids=["complex_ndarray", "complex_scalar",
             "complex_list", "real_imag_scalars",
             "real_imag_ndarray", "real_imag_list"]
    )
    def test_valid_init(self, args, expected_values):
        cmplx = crds.ComplexValuedCoordinates(*args)
        assert isinstance(cmplx, crds.ComplexValuedCoordinates)
        assert np.array_equal(cmplx._values, expected_values)

    @pytest.mark.parametrize(
        "args,expected_err",
        [
            (([1, 2], [-1, 0, 11]), ValueError),

            (([1, 0, 7], np.array([0, -1, 9])), ValueError)
        ],
        ids=["different_lengths", "different_types"]
    )
    def test_invalid_init(self, args, expected_err):
        with pytest.raises(expected_err):
            cmplx = crds.ComplexValuedCoordinates(*args)

    def test_cmplx_property(self, sample_cmplx):
        assert np.array_equal(sample_cmplx.cmplx, sample_cmplx._values)

    def test_real_property(self, sample_cmplx):
        assert np.array_equal(sample_cmplx.real, np.real(sample_cmplx._values))

    def test_imag_property(self, sample_cmplx):
        assert np.array_equal(sample_cmplx.imag, np.imag(sample_cmplx._values))

    def test_abs_property(self, sample_cmplx):
        expected_abs = np.sqrt(
            np.square(sample_cmplx.real) + np.square(sample_cmplx.imag))
        assert np.array_equal(sample_cmplx.abs, expected_abs)

    def test_phase_property(self, sample_cmplx):
        real = [1, 0, -1, 0, 1]
        imag = [0, 1, 0, -1, 1]
        expected_phase = np.array([0, np.pi/2, np.pi, -np.pi/2, np.pi/4])
        cmplx = crds.ComplexValuedCoordinates(real, imag)
        assert np.array_equal(cmplx.phase, expected_phase)

    def test_real_setter(self, sample_cmplx):
        new_real = np.array([-5, 0])
        with pytest.raises(ValueError):
            sample_cmplx.real = new_real

        new_real = [-5, 0, 1]
        with pytest.raises(ValueError):
            sample_cmplx.real = new_real

        imag = sample_cmplx.imag
        new_real = np.array([-5, 0, 1])
        sample_cmplx.real = new_real
        assert np.array_equal(sample_cmplx.real, new_real)
        assert np.array_equal(sample_cmplx.imag, imag)

    def test_imag_setter(self, sample_cmplx):
        new_imag = np.array([1, -2])
        with pytest.raises(ValueError):
            sample_cmplx.imag = new_imag

        new_imag = [1, 0, 3]
        with pytest.raises(ValueError):
            sample_cmplx.imag = new_imag

        real = sample_cmplx.real
        new_imag = np.array([-4, 0, 3])
        sample_cmplx.imag = new_imag
        assert np.array_equal(sample_cmplx.imag, new_imag)
        assert np.array_equal(sample_cmplx.real, real)

    @pytest.mark.parametrize(
        "dq,theta,expected_alpha_beta",
        [
            ([1 + 1J, -3, -1.5j], np.pi/2, [-1 + 1j, -1j*3, 1.5]),

            ([1 + 1J, -3, -1.5j], [np.pi/2, -np.pi/2, np.pi], [-1 + 1j, 1j*3, 1j*1.5]),

            ([1 + 1J, -3, -1.5j], np.array([np.pi/2, -np.pi/2, np.pi]),
             [-1 + 1j, 1j*3, 1j*1.5])
        ],
        ids=["ndarray_single_theta", "ndarray_list_theta",
             "ndarray_ndarray_theta"]
    )
    def test_to_alpha_beta(self, dq, theta, expected_alpha_beta):
        dq = crds.ComplexValuedCoordinates(dq)
        alpha_beta = dq.to_alpha_beta(theta)
        assert isinstance(alpha_beta, crds.AlphaBetaCoordinates)
        assert expected_alpha_beta == pytest.approx(
            alpha_beta.cmplx.tolist(), rel=1e-3, abs=1e-3)

    @pytest.mark.parametrize(
        "alpha_beta,theta,expected_dq",
        [
            ([1 + 1J, -3, -1.5j], -np.pi/2, [-1 + 1j, -1j*3, 1.5]),

            ([1 + 1J, -3, -1.5j], [-np.pi/2, np.pi/2, -np.pi], [-1 + 1j, 1j*3, 1j*1.5]),

            ([1 + 1J, -3, -1.5j], np.array([-np.pi/2, np.pi/2, -np.pi]),
             [-1 + 1j, 1j*3, 1j*1.5])
        ],
        ids=["ndarray_single_theta", "ndarray_list_theta",
             "ndarray_ndarray_theta"]
    )
    def test_to_dq(self, alpha_beta, theta, expected_dq):
        alpha_beta = crds.ComplexValuedCoordinates(alpha_beta)
        dq = alpha_beta.to_dq(theta)
        assert isinstance(dq, crds.DqCoordinates)
        assert expected_dq == pytest.approx(
            dq.cmplx.tolist(), rel=1e-3, abs=1e-3)


class TestAlphaBetaCoordinates():
    def test_instance(self, sample_alpha_beta):
        assert issubclass(crds.AlphaBetaCoordinates,
                          crds.ComplexValuedCoordinates)
        assert isinstance(sample_alpha_beta, crds.AlphaBetaCoordinates)
        assert isinstance(sample_alpha_beta, crds.ComplexValuedCoordinates)

    def test_to_abc(self, sample_alpha_beta):
        abc = sample_alpha_beta.to_abc()
        assert isinstance(abc, crds.AbcCoordinates)
        assert np.array_equal(
            abc.abc, crds.alpha_beta_to_abc(sample_alpha_beta).abc)


class TestDqCoordinates():
    def test_instance(self, sample_dq):
        assert issubclass(crds.DqCoordinates, crds.ComplexValuedCoordinates)
        assert isinstance(sample_dq, crds.DqCoordinates)
        assert isinstance(sample_dq, crds.ComplexValuedCoordinates)


class TestAbcCoordinates():
    def test_instance(self, sample_abc):
        assert isinstance(sample_abc, crds.AbcCoordinates)

    @pytest.mark.parametrize(
        "args,expected_values",
        [
            ((np.array([[4, 0, -3], [1, 7, -4], [-5, -7, 7]]),),
             np.array([[4, 0, -3], [1, 7, -4], [-5, -7, 7]])),

            (([[4, 0, -3], [1, 7, -4], [-5, -7, 7]],),
             np.array([[4, 0, -3], [1, 7, -4], [-5, -7, 7]])),

            (([4, 0, -3], [1, 7, -4], [-5, -7, 7]),
             np.array([[4, 0, -3], [1, 7, -4], [-5, -7, 7]])),

            ((np.array([4, 0, -3]), np.array([1, 7, -4]), np.array([-5, -7, 7])),
             np.array([[4, 0, -3], [1, 7, -4], [-5, -7, 7]])),

            ((np.array([4, 0, -3]), [1, 7, -4], np.array([-5, -7, 7])),
             np.array([[4, 0, -3], [1, 7, -4], [-5, -7, 7]]))
        ],
        ids=["single_array", "single_list",
             "seperate_lists", "seperate_ndarrays",
             "seperate_mixed"]
    )
    def test_valid_init(self, args, expected_values):
        abc = crds.AbcCoordinates(*args)
        assert isinstance(abc, crds.AbcCoordinates)
        assert np.array_equal(abc.abc, expected_values)

    @pytest.mark.parametrize(
        "args,expected_err",
        [
            (([1, 2, 0], [-1, 0, 11]), ValueError),

            (([1, 2, 0], [-1, 0, 11], [0, -2, 1]), ValueError),

            (([1, 0], [-1, 0, 11], [0, -2, 1]), ValueError)
        ],
        ids=["two_inputs", "not_symmetric", "unequal_length"]
    )
    def test_invalid_init(self, args, expected_err):
        with pytest.raises(expected_err):
            abc = crds.AbcCoordinates(*args)

    def test_to_alpha_beta(self, sample_abc):
        alpha_beta = sample_abc.to_alpha_beta()
        assert isinstance(alpha_beta, crds.AlphaBetaCoordinates)
        assert np.array_equal(
            alpha_beta.cmplx, crds.abc_to_alpha_beta(sample_abc).cmplx)


class TestDqToAlphaBeta():
    def test_instance(self, sample_dq):
        theta = np.pi/2
        transform = crds.dq_to_alpha_beta(sample_dq, theta)
        assert isinstance(transform, crds.AlphaBetaCoordinates)

    @pytest.mark.parametrize(
        "dq,theta,expected_alpha_beta",
        [
            ([1 + 1J, -3, -1.5j], np.pi/2, [-1 + 1j, -1j*3, 1.5]),

            ([1 + 1J, -3, -1.5j], [np.pi/2, -np.pi/2, np.pi], [-1 + 1j, 1j*3, 1j*1.5]),

            ([1 + 1J, -3, -1.5j], np.array([np.pi/2, -np.pi/2, np.pi]),
             [-1 + 1j, 1j*3, 1j*1.5])
        ],
        ids=["ndarray_single_theta", "ndarray_list_theta",
             "ndarray_ndarray_theta"]
    )
    def test_correct_transform(self, dq, theta, expected_alpha_beta):
        dq = crds.DqCoordinates(dq)
        transformed = crds.dq_to_alpha_beta(dq, theta).cmplx.tolist()
        assert expected_alpha_beta == pytest.approx(
            transformed, rel=1e-3, abs=1e-3)

    def test_invalid_transform(self, sample_dq):
        with pytest.raises(ValueError):
            alpha_beta = crds.dq_to_alpha_beta(sample_dq, [np.pi, np.pi/3])


class TestAlphaBetaToDq():
    def test_instance(self, sample_alpha_beta):
        theta = np.pi/2
        transform = crds.alpha_beta_to_dq(sample_alpha_beta, theta)
        assert isinstance(transform, crds.DqCoordinates)

    @pytest.mark.parametrize(
        "alpha_beta,theta,expected_dq",
        [
            ([1 + 1J, -3, -1.5j], -np.pi/2, [-1 + 1j, -1j*3, 1.5]),

            ([1 + 1J, -3, -1.5j], [-np.pi/2, np.pi/2, -np.pi], [-1 + 1j, 1j*3, 1j*1.5]),

            ([1 + 1J, -3, -1.5j], np.array([-np.pi/2, np.pi/2, -np.pi]),
             [-1 + 1j, 1j*3, 1j*1.5])
        ],
        ids=["ndarray_single_theta", "ndarray_list_theta",
             "ndarray_ndarray_theta"]
    )
    def test_correct_transform(self, alpha_beta, theta, expected_dq):
        alpha_beta = crds.AlphaBetaCoordinates(alpha_beta)
        transformed = crds.alpha_beta_to_dq(alpha_beta, theta).cmplx.tolist()
        assert expected_dq == pytest.approx(
            transformed, rel=1e-3, abs=1e-3)

    def test_invalid_transform(self, sample_alpha_beta):
        with pytest.raises(ValueError):
            dq = crds.alpha_beta_to_dq(sample_alpha_beta, [np.pi, np.pi/3])
