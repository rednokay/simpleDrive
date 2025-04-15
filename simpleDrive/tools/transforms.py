import numpy as np


def abc_to_alpha_beta(abc_matrix):
    pass


def alpha_beta_to_abc(alpha_beta_matrix: np.ndarray) -> np.ndarray:
    """
    Compute alpha-beta to abc transform.

    Parameters
    ----------
    alpha_beta_matrix : np.ndarray
        Matrix of alpha (first row) and beta (second row) values

    Returns
    -------
    np.ndarray
        Matrix of a (first row), b (second row) and c (third row) values
    """
    T = np.array([[1, 0],
                  [-0.5, np.sqrt(3)/2],
                  [-0.5, -np.sqrt(3)/2]])
    return T@alpha_beta_matrix


def alpha_beta_to_dq(alpha_beta_matrix, theta):
    pass

def dq_to_alpha_beta(dq_matrix, theta):
    pass