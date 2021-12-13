import numpy as np

import os

from lcode2dPy.plasma3d.solver import Plane2d3vPlasmaSolver
from lcode2dPy.beam3d_mod.beam import (BeamParticles, BeamSource, BeamDrain,
                                       BeamCalculator, concatenate_beam_layers)


class PushAndSolver3d:
    def __init__(self, config):
        self.config = config

        # Import plasma solver and beam pusher, pl = plasma
        self.pl_solver = Plane2d3vPlasmaSolver(config)
        self.beam_calc = BeamCalculator(config)

        self.xi_max = config.getfloat('window-length')
        self.xi_step_size = config.getfloat('xi-step')
        self.xi_steps = int(self.xi_max / self.xi_step_size)
        self.grid_steps = config.getint('window-width-steps')

        self.time_step_i = 0
        self.time_step_size = config.getfloat('time-step')

    def step_dt(self, pl_fields, pl_particles, pl_currents, pl_const_arrays,
                beam_source: BeamSource, beam_drain: BeamDrain):
        """
        Perform one time step of beam-plasma calculations.
        """
        self.beam_calc.start_time_step()
        beam_layer_to_move = BeamParticles(0)
        fell_size = 0

        # Arrays for simple diagnostics
        Ez_00_arr, xi_arr = [], []

        for xi_i in range(self.xi_steps + 1):
            beam_layer_to_layout = beam_source.get_beam_layer_to_layout(xi_i)
            ro_beam = (
                self.beam_calc.layout_beam_layer(beam_layer_to_layout, xi_i))

            prev_pl_fields = pl_fields.copy()

            pl_particles, pl_fields, pl_currents = self.pl_solver.step_dxi(
                pl_particles, pl_fields, pl_currents, pl_const_arrays, ro_beam)

            lost, moved, fell_to_next_layer = self.beam_calc.move_beam_layer(
                beam_layer_to_move, fell_size, xi_i, pl_fields, prev_pl_fields)

            # Beam layers operations:
            beam_layer_to_move = concatenate_beam_layers(beam_layer_to_layout,
                                                         fell_to_next_layer)
            fell_size = fell_to_next_layer.size

            beam_drain.push_beam_layer(moved)
            # beam_drain.push_beam_lost(lost)

            # Some diagnostics:
            Ez_00 = pl_fields.Ez[self.grid_steps//2, self.grid_steps//2]
            Ez_00_arr.append(Ez_00)
            xi_arr.append(-xi_i * self.xi_step_size)

            print(f'xi={-xi_i * self.xi_step_size:+.4f} {Ez_00:+.4e}')

        self.time_step_i += 1

        # if not os.path.isdir('xi_Ez_cpu'):
        #     os.mkdir('xi_Ez_cpu')
        # np.savez(f'''./xi_Ez_cpu/xi_Ez_{(self.time_step_i *
        #                              self.time_step_size):+09.2f}.npz''',
        #                              xi=xi_arr, Ez_00=Ez_00_arr)
