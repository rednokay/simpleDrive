import numpy as np
import simpleDrive.voltageModulation.overmodulation as ovm
import pytest


@pytest.fixture
def U_DC():
    return 400


@pytest.fixture
def tol():
    return {'rel': 1e-5, 'abs': 1e-5}


class TestVoltageLimit():
    @pytest.mark.parametrize(
        "phase, normalized_limit",
        [
            (4*np.pi/3, 2/3),

            (np.array([4*np.pi/3, 2*np.pi/3, np.pi/6]),
             np.array([2/3, 2/3, 1/np.sqrt(3)])),
        ],
        ids=["float_phase", "ndarray_phase"]
    )
    def test_input_size(self, phase, normalized_limit, U_DC, tol):
        vl = ovm.voltage_limit(phase, U_DC)
        vl_result = vl.tolist() if isinstance(vl, np.ndarray) else vl
        vl_expected = (U_DC*normalized_limit).tolist() if isinstance(
            normalized_limit, np.ndarray) else normalized_limit*U_DC

        assert vl_expected == pytest.approx(vl_result, **tol)
        assert isinstance(vl, float) | isinstance(vl, np.ndarray)

    @pytest.mark.parametrize(
        "phase, normalized_y",
        [
            (np.array([60, 67.5, 90, 105, 120])*np.pi/180,
             np.array([1/np.sqrt(3) for _ in range(5)])),

            (np.array([60, 67.5, 90, 105, 120])*-np.pi/180,
             np.array([-1/np.sqrt(3) for _ in range(5)])),
        ],
        ids=["top_half", "bottom_half"]
    )
    def test_correct_by_imag(self, phase, normalized_y, U_DC, tol):
        vl_im = np.imag(ovm.voltage_limit(phase, U_DC)*np.exp(1j*phase))
        vl_im_result = vl_im.tolist() if isinstance(vl_im, np.ndarray) else vl_im
        vl_im_expected = (U_DC*normalized_y).tolist() if isinstance(
            normalized_y, np.ndarray) else normalized_y*U_DC

        assert vl_im_expected == pytest.approx(vl_im_result, **tol)

    def test_correct_by_known_values(self, U_DC, tol):
        phase = np.array([i*np.pi/6 for i in range(12)])
        vl_result = ovm.voltage_limit(phase, U_DC).tolist()
        vl_expected = [2/3*U_DC if i %
                       2 == 0 else U_DC/np.sqrt(3) for i in range(12)]

        assert vl_expected == pytest.approx(vl_result, **tol)
