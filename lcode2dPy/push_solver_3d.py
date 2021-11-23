import time
import numpy as np

import os

from lcode2dPy.plasma3d.solver import Plane2d3vPlasmaSolver


class PushAndSolver3d:
    def __init__(self, config):
        self.config = config
        self.pl_solver = Plane2d3vPlasmaSolver(config)
        self.xi_max = config.getfloat('window-length')
        self.xi_step_size   = config.getfloat('xi-step')
        self.xi_steps = int(self.xi_max / self.xi_step_size)
        self.grid_steps = config.getint('window-width-steps')

        self.time_step_i = 0
        self.time_step_size = config.getfloat('time-step')

    def step_dt(self, pl_fields, pl_particles, pl_currents, pl_const_arrays,
                beam_source, beam_calculator):

        beam_calculator.start_time_step()

        Ez_00_arr = []
        xi_arr = []

        for xi_i in range(self.xi_steps + 1):
            beam_ro = beam_calculator.layout_beam()

            prev_pl_fields = pl_fields.copy()

            # time1 = time.time()          
            pl_particles, pl_fields, pl_currents = self.pl_solver.step_dxi(
                pl_particles, pl_fields, pl_currents, pl_const_arrays, beam_ro)
            # time2 = time.time()

            # print(f"Plasma solver done in {time2-time1:+.4f} s.", end='\t')

            # time1 = time.time()
            beam_calculator.move_beam(pl_fields, prev_pl_fields)
            # time2 = time.time()

            # print(f"Beam pusher done in {time2-time1:+.4f} s.")

            Ez_00 = pl_fields.Ez[self.grid_steps//2, self.grid_steps//2]
            Ez_00_arr.append(Ez_00)
            xi_arr.append(-xi_i * self.xi_step_size)

            print(f'xi={-xi_i * self.xi_step_size:+.4f} {Ez_00:+.4e}')

        beam_calculator.stop_time_step()

        self.time_step_i += 1
                
        if not os.path.isdir('xi_Ez'):
            os.mkdir('xi_Ez')
        np.savez(f'''./xi_Ez_test8/xi_Ez_{(self.time_step_i *
                                     self.time_step_size):+09.2f}.npz''',
                                     xi=xi_arr, Ez_00=Ez_00_arr)
