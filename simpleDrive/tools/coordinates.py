import numpy as np
from simpleDrive.tools.transforms import alpha_beta_to_abc


class AlphaBetaCoordinates():
    """
    Stationary reference frame aka alpha-beta coordinates
    """

    def __init__(self, *args):
        self.alpha_beta = None

        # TODO: Test for both arguments instances
        # Complex number as ndarray
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            self.alpha_beta = args[0]

        # Complex number as list or scalar
        elif len(args) == 1 and not isinstance(args[0], np.ndarray):
            self.alpha_beta = np.array([args[0]]).ravel()

        # Real and imag scalars
        elif len(args) == 2 and not isinstance(args[0], list) and not isinstance(args[0], np.ndarray):
            self.alpha_beta = np.array([args[0] + 1j*args[1]])

        # Real and imag as ndarray
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            if len(args[0]) != len(args[1]):
                raise ValueError(
                    "Alpha and beta lists must be of equal length")
            self.alpha_beta = args[0] + 1j*args[1]

        # Real and imag as list
        elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list):
            if len(args[0]) != len(args[1]):
                raise ValueError(
                    "Alpha and beta lists must be of equal length")
            self.alpha_beta = np.array(
                np.array(args[0]) + 1j*np.array(args[1]))

        # Other
        else:
            raise ValueError(
                "Could not create alpha-beta-coordinates from given values")

    def __str__(self):
        """
        Returns the np.ndarray string

        Returns
        -------
        str
            String of alpha beta np.ndarray
        """
        return str(self.alpha_beta)

    def to_abc(self):
        """
        Transforms to three-phase abc coordinates

        Returns
        -------
        np.ndarray
            Matrix of a (first row), b (second row) and c (third row) values
        """
        return alpha_beta_to_abc(self.alpha_beta)


class AbcCoordinates():
    """
    Three-phase abc coordinate system
    """

    # TODO: Check for equal length of each array
    def __init__(self, *args):
        self.abc = None
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            self.abc = args[0]
        elif len(args) == 1 and isinstance(args[0], list):
            self.abc = np.array(args[0])
        elif len(args) == 3:
            abc = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    abc.append(arg)
                elif isinstance(arg, list):
                    abc.append(np.array(arg))
                else:
                    raise ValueError(
                        "At least one input is neither list nor ndarray.")
            self.abc = np.array(abc)
        else:
            raise ValueError("Could not create ABC-values from selection.")
