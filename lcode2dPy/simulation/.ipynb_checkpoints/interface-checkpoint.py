from lcode2dPy.push_solver import PusherAndSolver
from lcode2dPy.beam.beam_slice import BeamSlice
from lcode2dPy.beam.beam_io import MemoryBeamSource, MemoryBeamDrain
#from lcode2dPy.push_solver_3d import PushAndSolver3d
import numpy as np
from lcode2dPy.config.default_config import default_config
from lcode2dPy.beam.beam_generator import make_beam, Gauss, rGauss
from lcode2dPy.plasma.initialization import init_plasma

class Simulation:
    def __init__(self, config=default_config, beam_generator=make_beam, beam_pars=None, diagnostics=None):
        self.config = config
        if config.get('geometry') == '3d':
            pass
#            self.push_solver = PushAndSolver3d(self.config) # 3d
        elif config.get('geometry') == 'circ' or config.get('geometry') == 'c':
            self.push_solver = PusherAndSolver(self.config) # circ
        elif config.get('geometry') == 'plane':
            self.push_solver = PusherAndSolver(self.config) # 2d_plane

        self.beam_generator = beam_generator
        self.beam_pars = beam_pars

        self.current_time = 0.
        self.beam_source = None
        self.beam_drain = None
        
        self.diagnostics = diagnostics

    def step(self, N_steps):
        # t step function, makes N_steps time steps.
        # Beam generation
        if self.beam_source is None:
            beam_particles = self.beam_generator(self.config, **self.beam_pars)
            beam_particle_dtype = np.dtype([('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'), ('q_m', 'f8'),
                               ('q_norm', 'f8'), ('id', 'i8')])
            beam_particles = np.array(list(map(tuple, beam_particles.to_numpy())), dtype=beam_particle_dtype)

            beam_slice = BeamSlice(beam_particles.size, beam_particles)
            self.beam_source = MemoryBeamSource(beam_slice) #TODO mpi_beam_source
            self.beam_drain = MemoryBeamDrain()
        if self.diagnostics:
            self.diagnostics.config = self.config
        # Time loop
        for t_i in range(N_steps):
            fields, plasma_particles = init_plasma(self.config)
            new_plasma_particles, new_fields = self.push_solver.step_dt(plasma_particles, fields, self.beam_source, self.beam_drain, self.current_time, self.diagnostics)
            # Every t step diagnostics
            if self.diagnostics:
                self.diagnostics.every_dt()
            self.current_time = self.current_time + (t_i + 1) * self.config.getfloat('time-step')



class Diagnostics2d:
    def __init__(self, dt_diag, dxi_diag):
        self.config = None
        self.dt_diag = dt_diag
        self.dxi_diag = dxi_diag

    def every_dt(self):
        pass # TODO

    def every_dxi(self, t, layer_idx, plasma_particles, plasma_fields, rho_beam, beam_slice):
        for diag_name in self.dxi_diag.keys():
            diag, pars = self.dxi_diag[diag_name]
            diag(self, t, layer_idx, plasma_particles, plasma_fields, rho_beam, beam_slice, **pars)
        return None