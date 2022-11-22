import cupy as cp

from ..config.config import Config
from .data import BeamParticles
from .weights import get_deposit_beam
from .move import get_move_beam


# Helper function #

def beam_substepping_step(q_m, pz, substepping_energy):
    dt = cp.ones_like(q_m, dtype=cp.float64)
    max_dt = cp.sqrt(cp.sqrt(1 / q_m ** 2 + pz ** 2) / substepping_energy)
    
    a = cp.ceil(cp.log2(dt / max_dt))
    a[a < 0] = 0
    dt /= 2 ** a

    return dt


class BeamCalculator:
    def __init__(self, config: Config):
        # Get main calculation parameters.
        xi_step_size = config.getfloat('xi-step')
        self.grid_step_size = config.getfloat('window-width-step-size')
        self.grid_steps = config.getint('window-width-steps')
        self.time_step = config.getfloat('time-step')
        self.substepping_energy = 2 #config.get("beam-substepping-energy")

        # Calculate the radius that marks that a particle is lost.
        max_radius = self.grid_step_size * self.grid_steps / 2
        self.lost_radius = max(0.9 * max_radius, max_radius - 1) # or just max_radius?

        self.deposit = get_deposit_beam(
            self.grid_steps, self.grid_step_size, xi_step_size)
        self.move = get_move_beam(
            self.grid_steps, self.grid_step_size, xi_step_size)

    # Helper functions for one time step cicle:

    def start_time_step(self):
        """
        Perform necessary operations before starting the time step.
        """
        # Get a grid for beam rho density
        self.rho_layout = cp.zeros((self.grid_steps, self.grid_steps),
                                    dtype=cp.float64)

    # Helper functions for depositing beam particles of a layer:

    def layout_beam_layer(self, beam_layer: BeamParticles, plasma_layer_idx):
        rho_layout = cp.zeros_like(self.rho_layout)

        if beam_layer.id.size != 0:
            self.deposit(
                plasma_layer_idx, beam_layer.x, beam_layer.y, beam_layer.xi,
                beam_layer.q_norm, self.rho_layout, rho_layout)

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
        dt = beam_substepping_step(beam_layer.q_m[idxes], beam_layer.pz[idxes],
                                   self.substepping_energy)
        beam_layer.dt[idxes] = dt * self.time_step
        beam_layer.remaining_steps[idxes] = (1. / dt).astype(cp.int_)

    def move_beam_layer(self, beam_layer: BeamParticles, fell_size,
                        pl_layer_idx, fields_after_layer, fields_before_layer):
        idxes_1 = cp.arange(beam_layer.id.size - fell_size)
        idxes_2 = cp.arange(beam_layer.id.size)

        size = idxes_2.size
        lost_idxes  = cp.zeros(size, dtype=cp.bool8)
        moved_idxes = cp.zeros(size, dtype=cp.bool8)
        fell_idxes  = cp.zeros(size, dtype=cp.bool8)

        if size != 0:
            self.start_moving_layer(beam_layer, idxes_1)
            beam_layer_to_move_idx = pl_layer_idx - 1

            lost_idxes, moved_idxes, fell_idxes = self.move(
                idxes_2, beam_layer_to_move_idx, beam_layer,
                fields_before_layer, fields_after_layer,
                lost_idxes, moved_idxes, fell_idxes)

        lost  = beam_layer.get_layer(idxes_2[lost_idxes])
        moved = beam_layer.get_layer(idxes_2[moved_idxes])
        fell  = beam_layer.get_layer(idxes_2[fell_idxes])

        return lost, moved, fell
