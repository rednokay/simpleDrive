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
    return AbcCoordinates(T@np.vstack((np.real(alpha_beta.cmplx), np.imag(alpha_beta.cmplx))))


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


class ComplexValuedCoordinates():
    def __init__(self, *args):
        self._values = None

        # TODO: Test for both arguments instances
        # Complex number as ndarray
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            self._values = args[0]

        # Complex number as list or scalar
        elif len(args) == 1 and not isinstance(args[0], np.ndarray):
            self._values = np.array([args[0]]).ravel()

        # Real and imag scalars
        elif len(args) == 2 and not isinstance(args[0], list) and not isinstance(args[0], np.ndarray):
            self._values = np.array([args[0] + 1j*args[1]])

        # Real and imag as ndarray
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            if len(args[0]) != len(args[1]):
                raise ValueError(
                    "Alpha and beta lists must be of equal length")
            self._values = args[0] + 1j*args[1]

        # Real and imag as list
        elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list):
            if len(args[0]) != len(args[1]):
                raise ValueError(
                    "Alpha and beta lists must be of equal length")
            self._values = np.array(
                np.array(args[0]) + 1j*np.array(args[1]))

        # Other
        else:
            raise ValueError(
                "Could not create alpha-beta-coordinates from given values")

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

    @cmplx.setter
    def cmplx(self, complex_values: np.ndarray):
        pass

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

    @real.setter
    def real(self, new_real_values: np.ndarray):
        """
        Sets the real values of the class' complex values.

        Parameters
        ----------
        new_real_values : np.ndarray
            Real values to set

        Raises
        ------
        ValueError
            Length of the values to set must match the other complex component
        ValueError
            The input must be formatted as np.ndarray
        """
        if len(new_real_values) != len(self.imag):
            raise ValueError(
                "The real values to set must be of the same length as the existing imaginary values.")
        if not isinstance(new_real_values, np.ndarray):
            raise ValueError("Real values must be formatted as np.ndarray.")
        self._values = new_real_values + 1j*self.imag

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

    @imag.setter
    def imag(self, new_imag_values: np.ndarray):
        """
        Sets the imaginary values of the class' complex values.

        Parameters
        ----------
        new_imag_values : np.ndarray
            Imaginary values to set

        Raises
        ------
        ValueError
            Length of the values to set must match the other complex component
        ValueError
            The input must be formatted as np.ndarray
        """
        if len(new_imag_values) != len(self.real):
            raise ValueError(
                "The imaginary values to set must be of the same length as the existing real values.")
        if not isinstance(new_imag_values, np.ndarray):
            raise ValueError(
                "Imaginary values must be formatted as np.ndarray.")
        self._values = self.real + 1j*new_imag_values

    # TODO: Add setter
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

    # TODO: Add setter
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
        super().__init__(*args)

    def to_abc(self) -> AbcCoordinates:
        """
        Transforms to three-phase abc coordinates

        Returns
        -------
        AbcCoordinates
            Transformed values in AbcCoordinates
        """
        return alpha_beta_to_abc(self)


class DqCoordinates(ComplexValuedCoordinates):
    """
    Rotating reference frame aka alpha-beta coordinates
    """

    def __init__(self, *args):
        super().__init__(*args)


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
