import numpy as np


class AbcCoordinates:
    pass


class AlphaBetaCoordinates:
    pass


def alpha_beta_to_abc(alpha_beta: AlphaBetaCoordinates) -> AbcCoordinates:
    """
    Compute alpha-beta to abc transform.

    Parameters
    ----------
    alpha_beta : AlphaBetaCoordinates
        Alpha-beta values to transform

    Returns
    -------
    AbcCoordinates
        Transformed values in ABC-coordinates
    """
    T = np.array([[1, 0],
                  [-0.5, np.sqrt(3)/2],
                  [-0.5, -np.sqrt(3)/2]])
    return AbcCoordinates(T@np.vstack((np.real(alpha_beta.alpha_beta), np.imag(alpha_beta.alpha_beta))))


def abc_to_alpha_beta(abc: AbcCoordinates) -> AlphaBetaCoordinates:
    """
    Compute abc to alpha-beta transform.

    Parameters
    ----------
    abc : AbcCoordinates
        ABC values to transform

    Returns
    -------
    AlphaBetaCoordinates
        Transformed values in alpha-beta-coordinates
    """
    T = np.array([[2/3, -1/3, -1/3], [0, 1/np.sqrt(3), -1/np.sqrt(3)]])
    transformed = T@abc.abc
    return AlphaBetaCoordinates(transformed[0], transformed[1])


# TODO: Move args checkin here
class ComplexValuedCoordinates():
    def __init__(self, *args):
        self._values = np.array(args[0]) + 1j*np.array(args[1])

    @property
    def cmplx(self) -> np.ndarray:
        """
        Property that returns the complex values of the class.

        Returns
        -------
        np.ndarray
            Complex values of the instance
        """
        return self._values

    @property
    def real(self) -> np.ndarray:
        """
        Property that returns the real vlaues of the class.

        Returns
        -------
        np.ndarray
            Real values of the instance
        """
        return np.real(self._values)

    @property
    def imag(self) -> np.ndarray:
        """
        Property that returns the imaginary vlaues of the class.

        Returns
        -------
        np.ndarray
            Imaginary values of the instance
        """
        return np.imag(self._values)

    @property
    def abs(self) -> np.ndarray:
        """
        Property that returns the absolute vlaues of the class' complex values.

        Returns
        -------
        np.ndarray
            Absolute values of the instance
        """
        return np.abs(self._values)

    @property
    def phase(self):
        """
        Property that returns the phase of the class' complex values.

        Returns
        -------
        np.ndarray
            Phase of the instance in rad
        """
        return np.atan2(np.imag(self._values), np.real(self._values))


class AlphaBetaCoordinates(ComplexValuedCoordinates):
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

    def to_abc(self) -> AbcCoordinates:
        """
        Transforms to three-phase abc coordinates

        Returns
        -------
        AbcCoordinates
            Transformed values in AbcCoordinates
        """
        return alpha_beta_to_abc(self)


# TODO: Handling of non-symmetric inputs!
# TODO: Float or int inputs
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

    def __str__(self):
        """
        Returns the np.ndarray string

        Returns
        -------
        str
            String of alpha beta np.ndarray
        """
        return str(self.abc)

    def to_alpha_beta(self):
        pass
