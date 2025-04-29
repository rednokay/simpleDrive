import numpy as np


def voltage_limit(phase: float | np.ndarray, U_DC: float) -> float | np.ndarray:
    """
    Computes the hexagonal voltage limit of an voltage-source inverter.

    Parameters
    ----------
    phase : float | np.ndarray
        Phase angle in question in rad
    U_DC : float
        DC-link voltage in V

    Returns
    -------
    float | np.ndarray
        The absolute value of the limit
    """
    phi = phase - np.floor(phase*3/np.pi)*np.pi/3 - np.pi/6

    x_max = 1/np.sqrt(3)*U_DC
    y_max = np.tan(phi)*x_max

    return np.abs(np.array(x_max + 1j*y_max))
