"""Top-level axisymmetric simulation class."""


class Axisymmetric2dSimulation(object):
    r"""
    Top-level lcode2dPy simulation class for axisymmetric 2d geometry.

    This class contains configuration of simulation and controls diagnostics.

    Parameters
    ----------
    r_max : float
        The radius of the simulation window in plasma length units
    r_step : float
        The step of the simulation grid in radial direction
        in plasma length units
    window_length : float
        Length of the simulation window in plasma length units. Simulation
        window length is increased into negative direction, minimal
        longitudinal value equals to -`window_length`
    xi_step : float
        Absolute value of the step of simulation grid in longitudinal direction
        in plasma length units
    continuation_mode : {'beam_evolution', 'beam_sequence', 'long_plasma'}
        Mode of plasma continuation to multiple simulation windows
    time_step : float
        Base beam time step in plasma time units
    time_max : float
        Maximum beam time in plasma time units. If it is not multiple of
        `time_step`, it is rounded to the nearest multiple of `time_step`.

    """

    def __init__(
        self,
        r_max=5.0,
        r_step=0.05,
        window_length=15.0,
        xi_step=0.05,
        continuation_mode='beam_evolution',
        time_step=25.0,
        time_max=200.5,
    ):
        pass
