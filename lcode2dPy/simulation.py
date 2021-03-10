from lcode2dPy.push_solver import PushAndSolver
from lcode2dPy.push_solver_3d import PushAndSolver3d
import numpy as np

class Simulation:
    def __init__(self, config, beam_generator, beam_pars):
        self.config = config
        if config.getstr('geometry') == 'circ':
            self.push_solver = None # circ
        elif config.getstr('geometry') == '3d':
            self.push_solver = None # 3d
        elif config.getstr('geometry') == '2d_plane':
            self.push_solver = None # 2d_plane

        self.beam_generator = beam_generator
        self.beam_pars = beam_pars

        self.current_time = 0.
        # self.push_solver = push_solver # должен сам решить, какой пуш_и_солвер
                                       # ему взять в зависимости от геометрии
        # self.beam = beam

    def step(self, N_steps):

        beam = self.beam_generator(self.beam_pars)

        for t_i in range(N_steps + 1):
            pl_fields, pl_particles, pl_currents, pl_const_arrays = init_plasma(config)

            # t = t_i * time_step_size # Need it to write diagnostics and nothing else

            pusher_solver.step_dt(pl_fields, pl_particles, pl_currents, pl_const_arrays,
                                beam_particles, beam_calulator, diagnostics)


class Diagnostics:
    def __init__(self, dt_diag, dxi_diag):
        self.dt_diag = dt_diag
        self.dxi_diag = dxi_diag
    
    def every_dt(self, ...):
        ....

    def every_dxi(self, ...):
        ....    
        

