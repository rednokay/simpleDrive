import numpy as np


def abc_to_alpha_beta(abc_matrix):
    pass


def alpha_beta_to_abc(alpha_beta: np.ndarray) -> np.ndarray:
    """
    Compute alpha-beta to abc transform.

    Parameters
    ----------
    alpha_beta_matrix : np.ndarray
        Matrix of values complex alpha-beta values

    Returns
    -------
    np.ndarray
        Matrix of a (first row), b (second row) and c (third row) values
    """
    T = np.array([[1, 0],
                  [-0.5, np.sqrt(3)/2],
                  [-0.5, -np.sqrt(3)/2]])
    return T@np.vstack((np.real(alpha_beta), np.imag(alpha_beta)))


def alpha_beta_to_dq(alpha_beta, theta):
    pass


def dq_to_alpha_beta(dq, theta):
    pass
