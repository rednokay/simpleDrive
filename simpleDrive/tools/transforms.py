# import numpy as np
# from simpleDrive.tools.coordinates import AlphaBetaCoordinates, AbcCoordinates
#
#
# def abc_to_alpha_beta(abc_matrix):
#     pass
#
#
# def alpha_beta_to_abc(alpha_beta: AlphaBetaCoordinates) -> AbcCoordinates:
#     """
#     Compute alpha-beta to abc transform.
#
#     Parameters
#     ----------
#     alpha_beta : AlphaBetaCoordinates
#         Alpha-beta values to transform
#
#     Returns
#     -------
#     AbcCoordinates
#         Transformed values in ABC-Coordinates
#     """
#     T = np.array([[1, 0],
#                   [-0.5, np.sqrt(3)/2],
#                   [-0.5, -np.sqrt(3)/2]])
#     return AbcCoordinates(T@np.vstack((np.real(alpha_beta.alpha_beta), np.imag(alpha_beta.alpha_beta))))
#
#
# def alpha_beta_to_dq(alpha_beta, theta):
#     pass
#
#
# def dq_to_alpha_beta(dq, theta):
#     pass
#
