from lcode2dPy.push_solver import PushAndSolver
from lcode2dPy.push_solver_3d import PushAndSolver3d
import numpy as np
from lcode2dPy.config.default_config import default_config
from beam3d.beam_generator import make_beam

class Simulation:
    def __init__(self, config=default_config, beam_generator=make_beam, beam_pars=None):
        self.config = config
        if config.getstr('geometry') == '3d':
            self.push_solver = PushAndSolver3d # 3d
        elif config.getstr('geometry') == 'circ':
            self.push_solver = PushAndSolver # circ
        elif config.getstr('geometry') == 'plane':
            self.push_solver = PushAndSolver # 2d_plane

        self.beam_generator = beam_generator
        self.beam_pars = beam_pars

        self.current_time = 0.
        # self.push_solver = push_solver # должен сам решить, какой пуш_и_солвер
                                       # ему взять в зависимости от геометрии
        # self.beam = beam

    def step(self, N_steps):

        beam = self.beam_generator(self.config, **self.beam_pars)

        for t_i in range(N_steps):
            pl_fields, pl_particles, pl_currents, pl_const_arrays = init_plasma(config)
            pusher_solver.step_dt(pl_fields, pl_particles, pl_currents, pl_const_arrays,
                                beam_particles, beam_calulator, diagnostics)
            self.current_time = self.current_time + (t_i + 1) * self.config.getfloat('time-step')



class Diagnostics2d:
    def __init__(self, dt_diag, dxi_diag):
        self.dt_diag = dt_diag
        self.dxi_diag = dxi_diag

    def every_dt(self):
        pass # TODO

    def every_dxi(self, plasma_particles, plasma_fields, beam_slice, rho_beam):
        for key in self.dxi_diag.keys():
            self.dxi_diag[key][0](plasma_particles, plasma_fields, beam_slice, rho_beam, **self.dxi_diag[key][1])
        return None

# Example
def E_z_diag(simulation, buffer, plasma_particles, plasma_fields, beam_slice, rho_beam, t_start, t_end, r_selected):
    if simulation.current_time < t_start or simulation.current_time > t_end:
        return
    r_grid_steps = simulation.config.getfloat('window-width') // simulation.config.getfloat('window-width-step-size')
    rs = np.linspace(0, simulation.config.getfloat('window-width'), r_grid_steps)
    E_z_selected = plasma_fields.E_z[rs == r_selected]
    buffer.append(E_z_selected)
    return E_z_selected

E_z_diag_pars = dict(
t_start = 0,
t_end = 10,
r_selected = 0
)

dxi_diag = dict(
E_z=[E_z_diag,
E_z_diag_pars]
)

diagnostics = Diagnostics2d(dt_diag=None, dxi_diag=dxi_diag)
