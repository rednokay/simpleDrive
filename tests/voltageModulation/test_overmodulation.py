import numpy as np
import simpleDrive.voltageModulation.overmodulation as ovm
import pytest


@pytest.fixture
def U_DC():
    return 400


@pytest.fixture
def tol():
    return {"rel": 1e-5, "abs": 1e-5}


@pytest.fixture
def phi30():
    return np.linspace(0, 2 * np.pi - np.pi / 6, 12)


class TestVoltageLimit:
    @pytest.mark.parametrize(
        "phase, normalized_limit",
        [
            (4 * np.pi / 3, 2 / 3),
            (
                np.array([4 * np.pi / 3, 2 * np.pi / 3, np.pi / 6]),
                np.array([2 / 3, 2 / 3, 1 / np.sqrt(3)]),
            ),
        ],
        ids=["float_phase", "ndarray_phase"],
    )
    def test_input_size(self, phase, normalized_limit, U_DC, tol):
        vl = ovm.voltage_limit(phase, U_DC)
        vl_result = vl.tolist() if isinstance(vl, np.ndarray) else vl
        vl_expected = (
            (U_DC * normalized_limit).tolist()
            if isinstance(normalized_limit, np.ndarray)
            else normalized_limit * U_DC
        )

        assert vl_expected == pytest.approx(vl_result, **tol)
        assert isinstance(vl, float) | isinstance(vl, np.ndarray)

    @pytest.mark.parametrize(
        "phase, normalized_y",
        [
            (
                np.array([60, 67.5, 90, 105, 120]) * np.pi / 180,
                np.array([1 / np.sqrt(3) for _ in range(5)]),
            ),
            (
                np.array([60, 67.5, 90, 105, 120]) * -np.pi / 180,
                np.array([-1 / np.sqrt(3) for _ in range(5)]),
            ),
        ],
        ids=["top_half", "bottom_half"],
    )
    def test_correct_by_imag(self, phase, normalized_y, U_DC, tol):
        vl_im = np.imag(ovm.voltage_limit(phase, U_DC) * np.exp(1j * phase))
        vl_im_result = vl_im.tolist() if isinstance(vl_im, np.ndarray) else vl_im
        vl_im_expected = (
            (U_DC * normalized_y).tolist()
            if isinstance(normalized_y, np.ndarray)
            else normalized_y * U_DC
        )

        assert vl_im_expected == pytest.approx(vl_im_result, **tol)

    def test_correct_by_known_values(self, U_DC, tol):
        phase = np.array([i * np.pi / 6 for i in range(12)])
        vl_result = ovm.voltage_limit(phase, U_DC).tolist()
        vl_expected = [
            2 / 3 * U_DC if i % 2 == 0 else U_DC / np.sqrt(3) for i in range(12)
        ]

        assert vl_expected == pytest.approx(vl_result, **tol)


class TestMinimumPhaseError:
    @pytest.mark.parametrize(
        "un",
        [
            (1.1 * np.exp(1j * np.pi / 5)),
            (
                np.array(
                    [
                        1.1 * np.exp(1j * np.pi / 5),
                        1.1 * np.exp(-1j * np.pi / 2),
                        1.1 * np.exp(1j * 3 * np.pi / 5),
                    ]
                )
            ),
        ],
        ids=["complex", "ndarray"],
    )
    def test_input_size(self, un, U_DC, tol):
        u = ovm.minimum_phase_error(U_DC * un, U_DC)
        u_result = u.tolist() if isinstance(u, np.ndarray) else u
        phase = np.atan2(np.imag(un), np.real(un))
        ue = ovm.voltage_limit(phase, U_DC) * np.exp(1j * phase)
        u_expected = ue.tolist() if isinstance(u, np.ndarray) else ue

        assert u_expected == pytest.approx(u_result, **tol)
        assert isinstance(u, complex) | isinstance(u, np.ndarray)

    def test_linear(self, phi30, tol):
        u_abs = 0.5
        u_ref = u_abs * np.exp(1j * phi30)
        u_abs_res = (np.abs(ovm.minimum_phase_error(u_ref))).tolist()
        u_abs_expected = [u_abs for _ in range(len(u_abs_res))]

        assert pytest.approx(u_abs_res, **tol) == u_abs_expected

    def test_sat(self, phi30, tol):
        u_abs = 1
        u_ref = u_abs * np.exp(1j * phi30)
        u_abs_res = (np.abs(ovm.minimum_phase_error(u_ref))).tolist()
        u_abs_expected = ovm.voltage_limit(phi30, 1).tolist()

        assert pytest.approx(u_abs_res, **tol) == u_abs_expected

    def test_ovm(self, phi30, tol):
        u_abs = 0.63
        u_ref = u_abs * np.exp(1j * phi30)
        u_abs_res = (np.abs(ovm.minimum_phase_error(u_ref))).tolist()
        u_abs_expected = [
            u_abs if i % 2 == 0 else 1 / np.sqrt(3) for i in range(len(u_abs_res))
        ]

        assert pytest.approx(u_abs_res, **tol) == u_abs_expected


class TestMinimumDistance:
    @pytest.mark.parametrize(
        "u_ref",
        [
            (1.1 * np.exp(1j * np.pi / 5)),
            (
                np.array(
                    [
                        1.1 * np.exp(1j * np.pi / 5),
                        1.1 * np.exp(-1j * np.pi / 2),
                        1.1 * np.exp(1j * 3 * np.pi / 5),
                    ]
                )
            ),
        ],
        ids=["complex", "ndarray"],
    )
    def test_input_size(self, u_ref, tol):
        u = ovm.minimum_distance(u_ref)
        u_result = np.abs(u).tolist() if isinstance(u, np.ndarray) else np.abs(u)
        phase = np.atan2(np.imag(u), np.real(u))
        ue = np.abs(ovm.voltage_limit(phase) * np.exp(1j * phase))
        u_expected = ue.tolist() if isinstance(u, np.ndarray) else ue

        assert pytest.approx(u_result, **tol) == u_expected
        assert isinstance(u, complex) | isinstance(u, np.ndarray)

    def test_linear(self, phi30, tol):
        u_abs = 0.5
        u_ref = u_abs * np.exp(1j * phi30)
        u_abs_res = (np.abs(ovm.minimum_distance(u_ref))).tolist()
        u_abs_expected = [u_abs for _ in range(len(u_abs_res))]

        assert pytest.approx(u_abs_res, **tol) == u_abs_expected

    def test_sat_abs(self, phi30, tol):
        u_abs = 1
        u_ref = u_abs * np.exp(1j * phi30)
        u_res = ovm.minimum_distance(u_ref)
        phase = np.atan2(np.imag(u_res), np.real(u_res))
        u_lim = ovm.voltage_limit(phase, 1).tolist()

        assert pytest.approx(np.abs(u_res).tolist(), **tol) == u_lim

    def test_ovm_abs(self, tol):
        u_abs = 0.63
        phi = np.array([0, 28, 61, 300, 267.75, 93]) * np.pi / 180
        u_ref = u_abs * np.exp(1j * phi)
        u_res = ovm.minimum_phase_error(u_ref)
        phase = np.atan2(np.imag(u_res), np.real(u_res))
        u_lim = ovm.voltage_limit(phase, 1).tolist()
        u_abs_expected = [u_abs, u_lim[1], u_abs, u_abs, u_lim[4], u_lim[5]]

        assert pytest.approx(np.abs(u_res).tolist(), **tol) == u_abs_expected

    @pytest.mark.parametrize(
        "phi",
        (
            (35 * np.pi / 180),
            (333 * np.pi / 180),
            (265 * np.pi / 180),
            (311.7 * np.pi / 180),
            (np.array([188, 95, 0, 18]) * np.pi / 180),
        ),
    )
    def test_geometric(self, phi, tol):
        u_abs = 1.5
        u_ref = u_abs * np.exp(1j * phi)
        u_res = ovm.minimum_distance(u_ref)

        phi_map = -np.floor(phi * 3 / np.pi) * np.pi / 3 - np.pi / 6
        u_ref_map = u_ref * np.exp(1j * phi_map)
        u_res_map = u_res * np.exp(1j * phi_map)

        if isinstance(u_res_map, np.ndarray):
            u_real_exp = (1 / np.sqrt(3) * np.ones(len(u_res))).tolist()
        else:
            u_real_exp = 1 / np.sqrt(3)

        u_imag_exp = np.imag(u_ref_map)
        if isinstance(u_imag_exp, np.ndarray):
            u_imag_exp[u_imag_exp > 1 / 3] = 1 / 3 * np.ones(len([u_imag_exp > 1 / 3]))
            u_imag_exp[u_imag_exp < -1 / 3] = (
                -1 / 3 * np.ones(len([u_imag_exp < -1 / 3]))
            )
            u_imag_exp = u_imag_exp.tolist()
        elif u_imag_exp > 1 / 3:
            u_imag_exp = 1 / 3
        elif u_imag_exp < -1 / 3:
            u_imag_exp = -1 / 3

        u_real_res = np.real(u_res_map)
        u_real_res = (
            u_real_res.tolist() if isinstance(u_real_res, np.ndarray) else u_real_res
        )
        u_imag_res = np.imag(u_res_map)
        u_imag_res = (
            u_imag_res.tolist() if isinstance(u_imag_res, np.ndarray) else u_imag_res
        )

        assert pytest.approx(u_real_res, **tol) == u_real_exp
        assert pytest.approx(u_imag_res, **tol) == u_imag_exp
