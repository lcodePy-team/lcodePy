# General imports
import numpy as np
from lcode2dPy.config.default_config import default_config
from lcode2dPy.config.config import Config
from lcode2dPy.beam_generator.beam_generator import make_beam, Gauss, rGauss

# Diagnostics
from lcode2dPy.diagnostics.targets import MyDiagnostics

# Imports for 2d simulation
from lcode2dPy.push_solvers.push_solver import PusherAndSolver
from lcode2dPy.beam.beam_slice import BeamSlice
from lcode2dPy.beam.beam_slice import particle_dtype as beam_particle_dtype_2d
from lcode2dPy.beam.beam_io import MemoryBeamSource, MemoryBeamDrain
from lcode2dPy.plasma.initialization import init_plasma as init_plasma_2d


class Simulation:
    def __init__(self, config: Config=default_config, beam_generator=make_beam,
                 beam_pars=None, diagnostics=None):
        # Firstly, we set some instance variables:
        self.config = config
        self.time_limit = config.getfloat('time-limit') 
        self.time_step_size = config.getfloat('time-step')
        self.rigid_beam = config.get('rigid-beam')

        # Mode of plasma continuation:
        self.cont_mode = config.get('continuation')

        # Here we get information about the geometry of the simulation window
        # and the type of processing unit (CPU or GPU)
        geometry = config.get('geometry').lower()
        self.PU_type = config.get('processing-unit-type').lower()

        if geometry == 'circ' or geometry == 'c':
            self.push_solver = PusherAndSolver(self.config) # circ
            self.init_plasma = init_plasma_2d
            self.beam_particle_dtype = beam_particle_dtype_2d
            self.geomtry = '2d'

        elif geometry == 'plane':
            self.push_solver = PusherAndSolver(self.config) # 2d_plane
            self.init_plasma = init_plasma_2d
            self.beam_particle_dtype = beam_particle_dtype_2d
            self.geometry = '2d'

        else:
            raise Exception(f"{geometry} type of geometry is not supported.")

        # Here we set parameters for beam generation, where we will store beam
        # particles and where they will go after calculations
        self.beam_generator = beam_generator
        self.beam_pars = beam_pars

        self.current_time = 0.
        # TODO: We should be able to use a beam file as a beam source.
        #       For now, it will always generate a new beam.
        self.beam_source = None
        self.beam_drain = None
        
        # Finally, we set the diagnostics.
        self.diagnostics = MyDiagnostics(config, diagnostics)

    def step(self, N_steps):
        """Compute N time steps."""
        # t step function, makes N_steps time steps.
        if N_steps is None:
            N_steps = int(self.time_limit / self.time_step_size)

        # Beam generation
        self.diagnostics.config()
        if self.beam_source is None:
            beam_particles = self.beam_generator(self.config,
                                                    **self.beam_pars)
            beam_particles = np.array(list(map(tuple, beam_particles.to_numpy())),
                                    dtype=self.beam_particle_dtype)

            beam_slice = BeamSlice(beam_particles.size, beam_particles)
            self.beam_source = MemoryBeamSource(beam_slice) #TODO mpi_beam_source
            self.beam_drain = MemoryBeamDrain()
        if self.diagnostics:
            self.diagnostics.config = self.config
        # Time loop
        for t_i in range(N_steps):
            fields, plasma_particles = self.init_plasma(self.config)

            plasma_particles_new, fields_new = self.push_solver.step_dt(
                plasma_particles, fields, self.beam_source, self.beam_drain,
                self.current_time, self.diagnostics)
                
            beam_particles = self.beam_drain.beam_slice()
            beam_slice = BeamSlice(beam_particles.size, beam_particles)
            self.beam_source = MemoryBeamSource(beam_slice)
            self.beam_drain = MemoryBeamDrain()
            self.current_time = self.current_time + self.time_step_size
            # Every t step diagnostics 
            # if self.diagnostics:
            #     self.diagnostics.every_dt()


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
