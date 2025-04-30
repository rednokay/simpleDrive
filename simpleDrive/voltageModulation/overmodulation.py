import numpy as np


def voltage_limit(phase: float | np.ndarray, U_DC: float = 1) -> float | np.ndarray:
    """
    Computes the hexagonal voltage limit of an voltage-source inverter.

    Parameters
    ----------
    phase : float | np.ndarray
        Phase angle in question in rad
    U_DC : float
        DC-link voltage in V, by default 1 which yields a normalized result

    Returns
    -------
    float | np.ndarray
        The absolute value of the limit
    """
    phi = phase - np.floor(phase*3/np.pi)*np.pi/3 - np.pi/6

    x_max = 1/np.sqrt(3)*U_DC
    y_max = np.tan(phi)*x_max

    return np.abs(np.array(x_max + 1j*y_max))


def minimum_phase_error(u_alphaBeta_ref: complex | np.ndarray, U_DC: float = 1) -> float | np.ndarray:
    """
    Computes a references voltage following the minimum phase error OVM.

    Parameters
    ----------
    u_alphaBeta_ref : complex | np.ndarray
        Input voltage to be truncated according to the OVM
    U_DC : float
        DC-link voltage in V, by default 1 which yields a normalized result

    Returns
    -------
    float | np.ndarray
        Truncated voltage following the OVM

    Literature
    -------
    [1] Y.-C. Kwon, S. Kim, and S.-K. Sul, “Six-Step Operation of PMSM With Instantaneous Current Control,” IEEE Transactions on Industry Applications, vol. 50, no. 4, pp. 2614–2625, Jul. 2014, doi: 10.1109/TIA.2013.2296652.
    """
    phi = np.atan2(np.imag(u_alphaBeta_ref), np.real(u_alphaBeta_ref))
    u_alphaBeta_lim = voltage_limit(phi, U_DC)

    if isinstance(u_alphaBeta_ref, np.ndarray):
        u_alphaBeta_ref[np.abs(u_alphaBeta_ref) > u_alphaBeta_lim] = u_alphaBeta_lim[np.abs(
            u_alphaBeta_ref) > u_alphaBeta_lim]*np.exp(1j*phi[np.abs(u_alphaBeta_ref) > u_alphaBeta_lim])
    elif u_alphaBeta_ref > u_alphaBeta_lim:
        u_alphaBeta_ref = u_alphaBeta_lim*np.exp(1j*phi)

    return u_alphaBeta_ref


def minimum_distance(u_alphaBeta_ref: complex | np.ndarray, U_DC: float = 1) -> float | np.ndarray:
    phase = np.atan2(np.imag(u_alphaBeta_ref), np.real(u_alphaBeta_ref))
    u_alphaBeta_lim = voltage_limit(phase, U_DC)

    phi = phase - np.floor(phase*3/np.pi)*np.pi/3 - np.pi/6

    if isinstance(u_alphaBeta_ref, np.ndarray):
        mask = np.abs(u_alphaBeta_ref) > u_alphaBeta_lim

        if any(mask):
            x = 1/np.sqrt(3)*U_DC*np.ones(np.size(u_alphaBeta_ref))
            y = np.imag(abs(u_alphaBeta_ref)*np.exp(1j*phi))

            y = np.minimum(np.maximum(y, -1/3*U_DC), 1/3*U_DC)
            new_phase = np.atan2(y, x)[mask]
            new_abs = np.abs(np.array(x + 1j*y))[mask]
            u_alphaBeta_ref[mask] = new_abs * \
                np.exp(1j*(new_phase + phase[mask] - phi[mask]))

    elif np.abs(u_alphaBeta_ref) > u_alphaBeta_lim:
        x = 1/np.sqrt(3)*U_DC
        y = np.imag(abs(u_alphaBeta_ref)*np.exp(1j*phi))

        y = np.minimum(np.maximum(y, -1/3*U_DC), 1/3*U_DC)
        new_phase = np.atan2(y, x)
        new_abs = np.abs(np.array(x + 1j*y))
        u_alphaBeta_ref = new_abs*np.exp(1j*(new_phase + phase - phi))

    return u_alphaBeta_ref
