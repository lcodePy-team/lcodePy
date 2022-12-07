import numba as nb
import numpy as np

from ..config.config import Config
from .data import BeamParticles
from .weights import get_deposit_beam
from .move import get_move_beam


# Helper function #

def get_beam_substepping_step(xp: np):
    if xp == np:
        @nb.njit
        def beam_substepping_step(q_m, pz, substepping_energy):
            dt = xp.ones_like(q_m, dtype=xp.float64)
            max_dt = xp.sqrt(
                xp.sqrt(1 / q_m ** 2 + pz ** 2) / substepping_energy)
            for i in range(len(q_m)):
                while dt[i] > max_dt[i]:
                    dt[i] /= 2.0
            return dt
    else:
        def beam_substepping_step(q_m, pz, substepping_energy):
            dt = xp.ones_like(q_m, dtype=xp.float64)
            max_dt = xp.sqrt(
                xp.sqrt(1 / q_m ** 2 + pz ** 2) / substepping_energy)

            a = xp.ceil(xp.log2(dt / max_dt))
            a[a < 0] = 0
            dt /= 2 ** a

            return dt

    return beam_substepping_step


class BeamCalculator:
    def __init__(self, xp: np, config: Config):
        # Get main calculation parameters.
        self.xp = xp

        self.grid_step_size = config.getfloat('window-width-step-size')
        self.grid_steps = config.getint('window-width-steps')
        self.time_step = config.getfloat('time-step')
        self.substep_energy = config.getfloat('beam-substepping-energy')

        self.deposit = get_deposit_beam(config)
        self.move_particles = get_move_beam(config)
        self.beam_substepping_step = get_beam_substepping_step(self.xp)

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

    def start_moving_layer(self, beam_layer: BeamParticles, idxes):
        """
        Perform necessary operations before moving a beam layer.
        """
        # TODO: Do we need to set dt and remaining_steps only for particles
        #       that have dt == 0?
        # mask = beam_layer.id[beam_layer.dt == 0] and idxes -> mask ???
        dt = self.beam_substepping_step(
            beam_layer.q_m[idxes], beam_layer.pz[idxes], self.substep_energy)
        beam_layer.dt[idxes] = dt * self.time_step
        beam_layer.remaining_steps[idxes] = (1. / dt).astype(self.xp.int_)

    def move_beam_layer(self, beam_layer: BeamParticles, fell_size,
                        pl_layer_idx, fields_after_layer, fields_before_layer):
        idxes_1 = self.xp.arange(beam_layer.id.size - fell_size)
        idxes_2 = self.xp.arange(beam_layer.id.size)

        size = idxes_2.size
        lost_idxes  = self.xp.zeros(size, dtype=self.xp.bool8)
        moved_idxes = self.xp.zeros(size, dtype=self.xp.bool8)
        fell_idxes  = self.xp.zeros(size, dtype=self.xp.bool8)

        if len(idxes_2) != 0:
            self.start_moving_layer(beam_layer, idxes_1)
            beam_layer_to_move_idx = pl_layer_idx - 1

            lost_idxes, moved_idxes, fell_idxes = self.move_particles(
                idxes_2, beam_layer_to_move_idx, beam_layer,
                fields_after_layer, fields_before_layer,
                lost_idxes, moved_idxes, fell_idxes)

        lost  = beam_layer.get_layer(idxes_2[lost_idxes])
        moved = beam_layer.get_layer(idxes_2[moved_idxes])
        fell  = beam_layer.get_layer(idxes_2[fell_idxes])

        return lost, moved, fell
