from scipy.constants import pi


def n_to_omega_electric(n, machine_data):
    """
    Converts speed to electric angular velocity

    Parameters
    ----------
    n : float
        mechanical rotor speed in rpm
    machine_data : MachineData
        machine data

    Returns
    -------
    float
        electric angular velocity omega in s⁻1
    """
    return 2*pi*machine_data.p*n/60
