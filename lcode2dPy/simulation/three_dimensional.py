"""Top-level three-dimensional simulation class."""
# General imports
import numpy as np

# Config
from lcode2dPy.config.default_config import default_config
from lcode2dPy.config.config import Config

# Diagnostics
from lcode2dPy.diagnostics.diagnostics_3d import Diagnostics3d

# Imports for beam generating in 3d (can be used for 2d also)
from lcode2dPy.alt_beam_generator.beam_generator import generate_beam
from lcode2dPy.alt_beam_generator.beam_generator import particle_dtype3d
from lcode2dPy.alt_beam_generator.beam_shape import BeamShape, BeamSegmentShape

# Imports for 3d simulation
from lcode2dPy.push_solvers.push_solver_3d import PushAndSolver3d as PushAndSolver3d_cpu
from lcode2dPy.beam3d import beam as beam3d_cpu
from lcode2dPy.plasma3d.initialization import init_plasma as init_plasma_cpu
from lcode2dPy.plasma3d.initialization import load_plasma as load_plasma_cpu

from lcode2dPy.push_solvers.push_solver_3d_gpu import PushAndSolver3d as PushAndSolver3d_gpu
from lcode2dPy.beam3d_gpu import beam as beam3d_gpu
from lcode2dPy.plasma3d_gpu.initialization import init_plasma as init_plasma_gpu
from lcode2dPy.plasma3d_gpu.initialization import load_plasma as load_plasma_gpu


class Cartesian3dSimulation:
    """
    Top-level lcodePy simulation class for cartesian 3d geometry.

    This class contains configuration of simulation and controls diagnostics.
    """
    def __init__(self, config: Config=default_config, beam_parameters: dict={},
                 diagnostics: list=None):
        # 0. We check if 'config' is just a Python dictionary:
        if type(config) == dict:
            config = Config(config)

        # Firstly, we check that the geomtry was set right:
        geometry = config.get('geometry').lower()
        if geometry != '3d':
            print("Sorry, you set a wrong type of geometry. If you want to use",
                  f"Cartesian3dSimulation, change geometry from {geometry} to",
                  "3d in config. (your_config.set('geometry', '3d'))")
            raise Exception("Please, read the text above in your window.")

        # We set some instance variables:
        self.config = config
        self.time_limit = config.getfloat('time-limit')
        self.time_step_size = config.getfloat('time-step')
        self.rigid_beam = config.get('rigid-beam')

        # Mode of plasma continuation:
        self.cont_mode = config.get('continuation')

        # Here we get information about the type of processing unit (CPU or GPU)
        self.pu_type = config.get('processing-unit-type').lower()

        if self.pu_type == 'cpu':
            self.push_solver = PushAndSolver3d_cpu(self.config)
            self.init_plasma = init_plasma_cpu
            self.load_plasma = load_plasma_cpu
            self.beam_module = beam3d_cpu
        elif self.pu_type == 'gpu':
            self.push_solver = PushAndSolver3d_gpu(self.config)
            self.init_plasma = init_plasma_gpu
            self.load_plasma = load_plasma_gpu
            self.beam_module = beam3d_gpu

        # Here we set parameters for beam generation, where we will store beam
        # particles and where they will go after calculations
        self.beam_parameters = beam_parameters
        self.beam_particle_dtype = particle_dtype3d

        self.current_time = 0.
        # TODO: We should be able to use a beam file as a beam source.
        #       For now, it will always generate a new beam.
        self.beam_source = None
        self.beam_drain = None

        # We set a value for external starting plasma state:
        self.external_plasma_state = False

        # Finally, we set the diagnostics.
        if type(diagnostics) != list and diagnostics is not None:
            diagnostics = [diagnostics] # If a user set only one diag. class.
        self.diagnostics = Diagnostics3d(config, diagnostics)

    def load_beamfile(self, path_to_beamfile='beamfile.npz'):
        beam_particles = self.beam_module.BeamParticles()
        beam_particles.load(path_to_beamfile)

        self.beam_source = self.beam_module.BeamSource(self.config,
                                                       beam_particles)
        self.beam_drain  = self.beam_module.BeamDrain()

    def load_plasmastate(self, path_to_plasmastate='plasmastate.npz'):
        self.external_plasma_state = True

        self.ld_fields, self.ld_particles, self.ld_currents = self.load_plasma(
            self.config, path_to_plasmastate
        )

    def step(self, N_steps=None):
        """Compute N time steps."""
        # t step function, makes N_steps time steps.
        if N_steps is None:
            N_steps = int(self.time_limit / self.time_step_size)
            print("Since the number of time steps hasn't been set explicitly,",
                  f"the code will simulate {N_steps} time steps with a time",
                  f"step size = {self.time_step_size}.")

        # 0. Checks for plasma continuation mode:
        if self.cont_mode == 'n' or self.cont_mode == 'no':
            # 1. If a beam source is empty (None), we generate
            #    a new beam according to set parameters:
            if self.beam_source is None:
                # Check for a beam being not rigid.
                if self.rigid_beam == 'n' or self.rigid_beam == 'no':
                    # Generate all parameters for a beam:
                    beam_particles = generate_beam(self.config,
                                                   self.beam_parameters,
                                                   self.beam_module)

                    # Here we create a beam source and a beam drain:
                    self.beam_source = self.beam_module.BeamSource(self.config,
                                                                beam_particles)
                    self.beam_drain  = self.beam_module.BeamDrain()

                # A rigid beam mode has not been implemented yet. If you are
                # writing rigid beam mode, just use rigid_beam_current(...) from
                # lcode2dPy.alt_beam_generator.beam_generator
                else:
                    print("Sorry, for now, only 'no' mode of rigid-beam is",
                          "supported.")
                    raise Exception("Please, read the text above in your window.")

            # 2. A loop that calculates N time steps:
            for t_i in range(N_steps):
                if self.external_plasma_state:
                    _, _, _, pl_const_arrays = self.init_plasma(self.config)
                    pl_fields, pl_particles, pl_currents =\
                        self.ld_fields, self.ld_particles, self.ld_currents
                else:
                    pl_fields, pl_particles, pl_currents, pl_const_arrays =\
                        self.init_plasma(self.config)

                # Calculates one time step:
                self.push_solver.step_dt(pl_fields, pl_particles,
                                        pl_currents, pl_const_arrays,
                                        self.beam_source, self.beam_drain,
                                        self.current_time, self.diagnostics)

                # Perform diagnostics
                self.diagnostics.dt(self.current_time,
                    pl_particles, pl_fields, pl_currents, self.beam_drain)

                # Here we transfer beam particles from beam_buffer to
                # beam_source for the next time step. And create a new beam
                # drain that is empty.
                self.beam_source = self.beam_module.BeamSource(self.config,
                                                    self.beam_drain.beam_buffer)
                self.beam_drain  = self.beam_module.BeamDrain()

                self.current_time = self.current_time + self.time_step_size
                print()

            # As in lcode2d, we save the beam state on reaching the time limit:
            self.beam_source.beam.save('beamfile')
            print('\nThe work is done!')

        # Other plasma continuation mode has not been implemented yet.
        # If you are writing these modes, just change where you put
        # init_plasma(...) and generate_beam(...)
        else:
            print("Sorry, for now, only 'no' mode of plasma continuation is", 
                  "supported.")
            raise Exception("Please, read the text above in your window.")


# class Diagnostics3d:
#     def __init__(self, dt_diag: dict, dxi_diag: dict):
#         self.config = None
#         self.dt_diag: dict = dt_diag
#         self.dxi_diag: dict = dxi_diag

#     def every_dt(self):
#         pass # TODO

#     def every_dxi(self, t, layer_idx, pl_fields, pl_particles, pl_currents, rho_beam, beam_slice):
#         for diag_name in self.dxi_diag.keys():
#             diag, pars = self.dxi_diag[diag_name]
#             diag(self, t, layer_idx, pl_fields, pl_particles, pl_currents, rho_beam, beam_slice, **pars)
#         return None
