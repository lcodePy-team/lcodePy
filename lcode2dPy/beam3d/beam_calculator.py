import numba as nb
import numpy as np

from ..config.config import Config
from .data import BeamParticles
from .weights import get_deposit_beam
from .move import get_move_beam_particles


# Helper function #

# NOTE: We have to write these functions separately and we don't merge them,
#       because other implementation options (including from old commits) led
#       to an illegal memory access when computing on a GPU. The problem is
#       probably in the internals of the cupy library. The specific simulation
#       settings will still create the problem, but in different places.

@nb.njit
def beam_substepping_step_numba(q_m, pz, substepping_energy):
    dt = np.ones_like(q_m, dtype=np.float64)
    max_dt = np.sqrt(np.sqrt(1 / q_m ** 2 + pz ** 2) / substepping_energy)
    for i in range(len(q_m)):
        while dt[i] > max_dt[i]:
            dt[i] /= 2.0
    return dt


def get_beam_substepping_step_cupy():
    import cupy as cp

    calculate_substepping_step = cp.ElementwiseKernel(
        in_params="T q_m, T pz, float64 substepping_energy",
        out_params="T dt",
        operation="""
        T max_dt = sqrt(sqrt(1 / (q_m*q_m) + pz*pz) / substepping_energy);
        while (dt > max_dt){
            dt /= 2;
        }
        """)

    def beam_substepping_step(q_m, pz, substepping_energy):
        dt = cp.ones_like(q_m, dtype=cp.float64)
        calculate_substepping_step(q_m, pz, substepping_energy, dt)
        return dt

    return beam_substepping_step


class BeamCalculator:
    def __init__(self, config: Config):
        # Get main calculation parameters.
        self.xp = config.xp

        self.grid_step_size = config.getfloat('window-width-step-size')
        self.grid_steps = config.getint('window-width-steps')
        self.time_step = config.getfloat('time-step')
        self.substep_energy = config.getfloat('beam-substepping-energy')

        self.deposit = get_deposit_beam(config)
        self.move_particles = get_move_beam_particles(config)

        pu_type = config.get('processing-unit-type').lower()
        if pu_type == 'cpu':
            self.beam_substepping_step = beam_substepping_step_numba
        if pu_type == 'gpu':
            self.beam_substepping_step = get_beam_substepping_step_cupy()

    # Helper functions for one time step cicle:

    def start_time_step(self):
        """
        Perform necessary operations before starting the time step.
        """
        # Get a grid for beam rho density
        self.rho_layout = self.xp.zeros((self.grid_steps, self.grid_steps),
                                        dtype=self.xp.float64)

    # Helper functions for depositing beam particles of a layer:

    def layout_beam_layer(self, beam_layer: BeamParticles, plasma_layer_idx):
        rho_layout = self.xp.zeros_like(self.rho_layout)

        if beam_layer.id.size != 0:
            self.deposit(plasma_layer_idx, beam_layer.x, beam_layer.y,
                         beam_layer.xi, beam_layer.q_norm,
                         self.rho_layout, rho_layout)

        self.rho_layout, rho_layout = rho_layout, self.rho_layout
        rho_layout /= self.grid_step_size ** 2

        return rho_layout

    # Helper functions for moving beam particles of a layer:

    def start_moving_layer(self, beam_layer: BeamParticles, fell_size):
        """
        Perform necessary operations before moving a beam layer.
        """
        # TODO: Do we need to set dt and remaining_steps only for particles
        #       that have dt == 0?
        # mask = beam_layer.id[beam_layer.dt == 0] and idxes -> mask ???

        # NOTE: We perform the following callculations only for the particles
        #       that haven't fallen from the previous level. Because these
        #       particles have been appended (concatenated) to the end of the
        #       beam_layer, we get unfallen particles using [:size] with size:
        size = beam_layer.id.size - fell_size

        dt = self.beam_substepping_step(
            beam_layer.q_m[:size], beam_layer.pz[:size], self.substep_energy)
        beam_layer.dt[:size] = dt * self.time_step
        beam_layer.remaining_steps[:size] = (1. / dt).astype(self.xp.int_)

    def move_beam_layer(self, beam_layer: BeamParticles, fell_size,
                        pl_layer_idx, fields_after_layer, fields_before_layer):
        lost_idxes  = self.xp.zeros(beam_layer.id.size, dtype=self.xp.bool8)
        moved_idxes = self.xp.zeros(beam_layer.id.size, dtype=self.xp.bool8)
        fell_idxes  = self.xp.zeros(beam_layer.id.size, dtype=self.xp.bool8)

        if beam_layer.id.size != 0:
            self.start_moving_layer(beam_layer, fell_size)
            beam_layer_idx = pl_layer_idx - 1

            self.move_particles(
                beam_layer_idx, beam_layer, fields_after_layer,
                fields_before_layer, lost_idxes, moved_idxes, fell_idxes)

        lost  = beam_layer.get_layer(lost_idxes)
        moved = beam_layer.get_layer(moved_idxes)
        fell  = beam_layer.get_layer(fell_idxes)

        return lost, moved, fell
