import time
import cupy as cp

from lcode2dPy.plasma3d_gpu.data import GPUArraysView
from lcode2dPy.plasma3d_gpu.solver import Plane2d3vPlasmaSolver


class PushAndSolver3d:
    def __init__(self, config):
        self.config = config
        self.pl_solver = Plane2d3vPlasmaSolver(config)
        self.xi_max = config.getfloat('window-length')
        self.xi_step_size   = config.getfloat('xi-step')
        self.xi_steps = int(self.xi_max / self.xi_step_size)
        self.grid_steps = config.getint('window-width-steps')

    def step_dt(self, pl_fields, pl_particles, pl_currents, pl_const_arrays,
                beam_source, beam_calculator):

        beam_calculator.start_time_step()

        gpu_index = 0
        with cp.cuda.Device(gpu_index):
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
                # beam_calculator.xi_layer += 1
                # time2 = time.time()

                # print(f"Beam pusher done in {time2-time1:+.4f} s.")
                
                view_pl_fields = GPUArraysView(pl_fields)
                Ez_00 = view_pl_fields.Ez[self.grid_steps//2, self.grid_steps//2]

                print(f'xi={xi_i * self.xi_step_size:+.4f} {Ez_00:+.4e}')

        beam_calculator.stop_time_step()
