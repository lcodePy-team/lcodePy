# General imports
import numpy as np
from lcode2dPy.config.default_config import default_config
from lcode2dPy.config.config import Config
from lcode2dPy.beam_generator.beam_generator import make_beam, Gauss, rGauss

# Diagnostics
from lcode2dPy.diagnostics.targets import MyDiagnostics

# Imports for 2d simulation
from lcode2dPy.push_solver import PusherAndSolver
from lcode2dPy.beam.beam_slice import BeamSlice
from lcode2dPy.beam.beam_slice import particle_dtype as beam_particle_dtype_2d
from lcode2dPy.beam.beam_io import MemoryBeamSource, MemoryBeamDrain
from lcode2dPy.plasma.initialization import init_plasma as init_plasma_2d

# Imports for 3d simulation
from lcode2dPy.alt_beam_generator.beam_generator import generate_beam
from lcode2dPy.alt_beam_generator.beam_generator import particle_dtype3d
from lcode2dPy.alt_beam_generator.beam_shape import BeamShape, BeamSegmentShape

from lcode2dPy.push_solver_3d import PushAndSolver3d as PushAndSolver3d_cpu
from lcode2dPy.beam3d import beam as beam3d_cpu
from lcode2dPy.plasma3d.initialization import init_plasma as init_plasma_3d_cpu

from lcode2dPy.push_solver_3d_gpu import PushAndSolver3d as PushAndSolver3d_gpu
from lcode2dPy.beam3d_gpu import beam as beam3d_gpu
from lcode2dPy.plasma3d_gpu.initialization import init_plasma as init_plasma_3d_gpu


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

        if geometry == '3d':
            if self.PU_type == 'cpu':
                self.push_solver = PushAndSolver3d_cpu(self.config)
                self.init_plasma = init_plasma_3d_cpu
                self.beam = beam3d_cpu
            elif self.PU_type == 'gpu':
                self.push_solver = PushAndSolver3d_gpu(self.config)
                self.init_plasma = init_plasma_3d_gpu
                self.beam = beam3d_gpu
            
            self.beam_particle_dtype = particle_dtype3d
            self.geometry = '3d'

        elif geometry == 'circ' or geometry == 'c':
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

        if self.geometry == '2d':
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

        elif self.geometry == '3d':
            # 1. If a beam source is empty (None), we generate
            #    a new beam according to set parameters:
            if self.beam_source is None and (
               self.rigid_beam == 'n' or self.rigid_beam == 'no'):
                # You can set 'beam_current' and 'particles_in_layer' parameters
                # here for the whole beam (beam_shape).
                beam_shape = BeamShape(**self.beam_pars)

                # Works only for one segment (for now).
                beam_segment = BeamSegmentShape(**self.beam_pars)
                beam_shape.add_segment(beam_segment)

                # Now we generate beam particles:
                beam_particles = self.beam.BeamParticles(0)
                beam_particles.init_generated(generate_beam(self.config,
                                                            beam_shape))

                # Here we create a beam source and a beam drain:
                self.beam_source = self.beam.BeamSource(self.config,
                                                        beam_particles)
                self.beam_drain  = self.beam.BeamDrain()
            else:
                raise Exception("Sorry, for now, only 'no' mode of rigid-beam is supported.")

            # 2. A loop that calculates N time steps:
            for t_i in range(N_steps):
                # Checks for plasma continuation mode:
                if self.cont_mode == 'n' or self.cont_mode == 'no':
                    pl_fields, pl_particles, pl_currents, pl_const_arrays =\
                        self.init_plasma(self.config)

                    # Calculates one time step:
                    self.push_solver.step_dt(pl_fields, pl_particles,
                                            pl_currents, pl_const_arrays,
                                            self.beam_source, self.beam_drain)

                    # Here we transfer beam particles from beam_buffer to
                    # beam_source for the next time step. And create a new beam
                    # drain that is empty.
                    self.beam_source = self.beam.BeamSource(self.config,
                                                        self.beam_drain.beam_buffer)
                    self.beam_drain  = self.beam.BeamDrain()

                    self.current_time = self.current_time + self.time_step_size

                else:
                    raise Exception("Sorry, for now, only 'no' mode of plasma continuation is supported.")


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
