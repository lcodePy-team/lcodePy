"""Top-level three-dimensional simulation class."""
# General imports
import numpy as np
from lcode2dPy.config.default_config import default_config
from lcode2dPy.config.config import Config

# Diagnostics
# from lcode2dPy.diagnostics.targets_3d import Diagnostics

# Imports for 3d simulation
from lcode2dPy.alt_beam_generator.beam_generator import generate_beam
from lcode2dPy.alt_beam_generator.beam_generator import particle_dtype3d
from lcode2dPy.alt_beam_generator.beam_shape import BeamShape, BeamSegmentShape

from lcode2dPy.push_solver_3d import PushAndSolver3d as PushAndSolver3d_cpu
from lcode2dPy.beam3d import beam as beam3d_cpu
from lcode2dPy.plasma3d.initialization import init_plasma as init_plasma_cpu

from lcode2dPy.push_solver_3d_gpu import PushAndSolver3d as PushAndSolver3d_gpu
from lcode2dPy.beam3d_gpu import beam as beam3d_gpu
from lcode2dPy.plasma3d_gpu.initialization import init_plasma as init_plasma_gpu


class Cartesian3dSimulation:
    """
    Top-level lcodePy simulation class for cartesian 3d geometry.

    This class contains configuration of simulation and controls diagnostics.
    """
    def __init__(self, config:Config=default_config,
                 beam_pars=None, diagnostics=None):
        # Firstly, we set some instance variables:
        self.config = config
        self.time_limit = config.getfloat('time-limit') 
        self.time_step_size = config.getfloat('time-step')
        self.rigid_beam = config.get('rigid-beam')

        # Mode of plasma continuation:
        self.cont_mode = config.get('continuation')

        # Here we get information about the type of processing unit (CPU or GPU)
        self.PU_type = config.get('processing-unit-type').lower()
        
        if self.PU_type == 'cpu':
            self.push_solver = PushAndSolver3d_cpu(self.config)
            self.init_plasma = init_plasma_cpu
            self.beam = beam3d_cpu
        elif self.PU_type == 'gpu':
            self.push_solver = PushAndSolver3d_gpu(self.config)
            self.init_plasma = init_plasma_gpu
            self.beam = beam3d_gpu
        
        # Here we set parameters for beam generation, where we will store beam
        # particles and where they will go after calculations
        self.beam_pars = beam_pars
        self.beam_particle_dtype = particle_dtype3d

        self.current_time = 0.
        # TODO: We should be able to use a beam file as a beam source.
        #       For now, it will always generate a new beam.
        self.beam_source = None
        self.beam_drain = None
        
        # Finally, we set the diagnostics.
        # self.diagnostics = Diagnostics(config, diagnostics)

    def step(self, N_steps):
        """Compute N time steps."""
        # t step function, makes N_steps time steps.
        if N_steps is None:
            N_steps = int(self.time_limit / self.time_step_size)

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


class Diagnostics3d:
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
