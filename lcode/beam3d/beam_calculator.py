import numba as nb
import numpy as np

from ..config.config import Config
from .data import BeamParticles
from .weights import get_beam_deposition_function
from .push_cpu import get_beam_pusher_numba
from .push_gpu import get_beam_pusher_cupy


def get_beam_pusher(config: Config):
    xi_step_size = config.getfloat('xi-step')
    grid_step_size = config.getfloat('transverse-step')
    grid_steps = config.getint('window-width-steps')

    # Calculate the radius that marks that a particle is lost.
    max_radius = grid_step_size * grid_steps / 2
    lost_radius = max(0.9 * max_radius, max_radius - 1)

    pu_type = config.get('processing-unit-type').lower()
    integration_method = config.get('beam-pusher').lower()
    if pu_type == 'cpu':
        _push_beam_particles = get_beam_pusher_numba(integration_method)
    if pu_type == 'gpu':
        _push_beam_particles = get_beam_pusher_cupy(integration_method)

    def push_beam_particles(plasma_slice_idx, beam_layer: BeamParticles,
                            fields_prev, fields,
                            lost_idxes, moved_idxes, fell_idxes):
        _push_beam_particles(
            xi_step_size, lost_radius, plasma_slice_idx, 
            grid_step_size, grid_steps, 
            fields_prev.Ex, fields_prev.Ey, fields_prev.Ez,
            fields_prev.Bx, fields_prev.By, fields_prev.Bz,
            fields.Ex, fields.Ey, fields.Ez,
            fields.Bx, fields.By, fields.Bz,

            beam_layer.q_m, beam_layer.dt,
            beam_layer.remaining_steps, beam_layer.id,
            beam_layer.x, beam_layer.y, beam_layer.xi,
            beam_layer.ux, beam_layer.uy, beam_layer.uz,
            lost_idxes, moved_idxes, fell_idxes,

            size=beam_layer.id.size)

    return push_beam_particles


def get_beam_t_step_calculator(config):
    """
    Generation of a function to calculate the correct time step
    for beam particles according to 'beam-substepping-energy'.
    """
    pu_type = config.get('processing-unit-type').lower()
    if pu_type == 'cpu':
        @nb.njit
        def calc_beam_t_step_numba(q_m, uz, substepping_energy):
            dt = np.ones_like(q_m, dtype=np.float64)
            max_dt = np.sqrt(np.sqrt(1 + uz**2) / substepping_energy / np.abs(q_m))
            for i in range(len(q_m)):
                while dt[i] > max_dt[i]:
                    dt[i] /= 2.0
            return dt
        return calc_beam_t_step_numba

    if pu_type == 'gpu':
        import cupy as cp

        calc_beam_t_step_cupy_kernel = cp.ElementwiseKernel(
            in_params="T q_m, T uz, float64 substepping_energy",
            out_params="T dt",
            operation="""
            T max_dt = sqrt(sqrt(1 + uz*uz) / substepping_energy / abs(q_m));
            while (dt > max_dt){
                dt /= 2;
            }
            """)

        def calc_beam_t_step_cupy(q_m, uz, substepping_energy):
            dt = cp.ones_like(q_m, dtype=cp.float64)
            calc_beam_t_step_cupy_kernel(q_m, uz, substepping_energy, dt)
            return dt

        return calc_beam_t_step_cupy


# ----- A class for a beam consisting of macroparticles -----

class BeamCalculator:
    """
    The main class for performing operations with a beam in 3d.
    """
    def __init__(self, config: Config):
        self.xp = config.xp

        self.grid_step_size = config.getfloat('transverse-step')
        self.grid_steps = config.getint('window-width-steps')
        self.time_step = config.getfloat('time-step')
        self.substep_energy = config.getfloat('beam-substepping-energy')

        self._deposit = get_beam_deposition_function(config)
        self._push_particles = get_beam_pusher(config)
        self._calc_beam_t_step = get_beam_t_step_calculator(config)


    def start_time_step(self):
        """
        Perform necessary operations before starting the time step.
        """
        # Get a grid for beam density
        self.rho_beam_next = self.xp.zeros((self.grid_steps, self.grid_steps),
                                           dtype=self.xp.float64)

    def deposit_beam_layer(self, beam_layer: BeamParticles, plasma_slice_idx):
        """
        Perform deposition of the beam layer on the density grid.

        Parameters
        ----------
        beam_layer : BeamParticles
            Beam particles to be deposited at the current xi-step.
        plasma_layer_idx : int
            The xi-step number, which is calculated next.

        Returns
        -------
        rho_beam : np.ndarray
            Beam density for xi = -dxi * plasma_layer_idx.
        """
        rho_beam = self.rho_beam_next
        self.rho_beam_next = self.xp.zeros_like(rho_beam)

        if beam_layer.id.size != 0:
            self._deposit(plasma_slice_idx, beam_layer.x, beam_layer.y,
                          beam_layer.xi, beam_layer.q_norm,
                          rho_beam, self.rho_beam_next)

        rho_beam /= self.grid_step_size**2

        return rho_beam

    def push_beam_layer(self, beam_layer: BeamParticles, fell_size,
                        plasma_slice_idx,
                        fields_prev, fields):
        """
        Integrate beam particles.

        Parameters
        ----------
        beam_layer : BeamParticles
            Beam particles to be pushed at the current xi-step.
        plasma_layer_idx : int
            The xi-step number, which have been calculated.
        fields_prev : Array
            Fields at (plasma_slice_idx - 1) step.
        fields : Array
            Fields at (plasma_slice_idx) step.

        Returns
        -------
        (lost, moved, fell) : tuple of BeamParticles
            lost - particles have left the simulation domain.
            moved - particles have completed the current time step.
            fell - particles should be integrated at the next xi-step.
        """
        lost_idxes  = self.xp.zeros(beam_layer.id.size, dtype=self.xp.bool_)
        moved_idxes = self.xp.zeros(beam_layer.id.size, dtype=self.xp.bool_)
        fell_idxes  = self.xp.zeros(beam_layer.id.size, dtype=self.xp.bool_)

        if beam_layer.id.size != 0:
            # Initialization of substepping for new particles.
            size = beam_layer.id.size - fell_size
            dt = self._calc_beam_t_step(beam_layer.q_m[:size],
                                        beam_layer.uz[:size],
                                        self.substep_energy)
            beam_layer.dt[:size] = dt * self.time_step
            beam_layer.remaining_steps[:size] = (1. / dt).astype(self.xp.int_)

            self._push_particles(plasma_slice_idx, beam_layer,
                                 fields_prev, fields,
                                 lost_idxes, moved_idxes, fell_idxes)

        lost  = beam_layer[lost_idxes]
        moved = beam_layer[moved_idxes]
        fell  = beam_layer[fell_idxes]

        return lost, moved, fell


# ----- A class for a rigid rigid beam -----

class RigidBeamCalculator:
    def __init__(self, config: Config):
        # Get main calculation parameters.
        self.xp = config.xp
        self.xi_step_size = config.getfloat('xi-step')
        
        # Creates a transversal grid
        grid_steps     = config.getint('window-width-steps')
        grid_step_size = config.getfloat('transverse-step')

        grid = ((self.xp.arange(grid_steps) - grid_steps // 2)
                * grid_step_size)
        self.y_grid, self.x_grid = self.xp.meshgrid(grid, grid)
    
    def start_time_step(self):
        """A dummy function for the rigid-beam mode."""
        pass
    
    def layout_beam_layer(self, beam_charge_distribution_function,
                          plasma_layer_idx):
        xi = -plasma_layer_idx * self.xi_step_size        
        return beam_charge_distribution_function(self.xp, xi,
                                                 self.x_grid, self.y_grid)

    def move_beam_layer(self, beam_layer, fell_size,
                        pl_layer_idx, fields_after_layer, fields_before_layer):
        """A dummy function for the rigid-beam mode."""
        return None, None, None