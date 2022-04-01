"""Top-level three-dimensional simulation class."""
# General imports
import numpy as np

# Config
from lcode2dPy.config.default_config_values import default_config_values
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
    def __init__(self, config=default_config_values, beam_parameters={},
                 diagnostics=None):
        self.config = config
        self.beam_parameters = beam_parameters
        self.diagnostics = diagnostics

        # Here we set parameters for beam generation, where we will store beam
        # particles and where they will go after calculations
        # self.beam_parameters = beam_parameters
        # self.beam_particle_dtype = particle_dtype3d

        # We use this time as a general time value:
        self.current_time = 0.

        # We initialize a beam source and a beam drain:
        self.beam_source = None
        self.beam_drain = None

        # We set that initially the code doesn't use an external plasma state:
        self.external_plasmastate = False
        self.path_to_plasmastate = 'plasmastate.npz'

    def pull_config(self):
        # 0. We set __config__ as a Config class instance:
        self.__config__ = Config(self.config)

        # Firstly, we check that the geomtry was set right:
        geometry = self.__config__.get('geometry').lower()
        if geometry != '3d':
            print("Sorry, you set a wrong type of geometry. If you want to use",
                  f"Cartesian3dSimulation, change geometry from {geometry} to",
                  "3d in config. (your_config.set('geometry', '3d'))")
            raise Exception("Please, read the text above in your window.")

        # We set some instance variables:
        self.time_limit = self.__config__.getfloat('time-limit')
        self.time_step_size = self.__config__.getfloat('time-step')
        self.rigid_beam = self.__config__.get('rigid-beam')

        # Mode of plasma continuation:
        self.cont_mode = self.__config__.get('continuation')

        # Here we get information about the type of processing unit (CPU or GPU)
        self.pu_type = self.__config__.get('processing-unit-type').lower()

        if self.pu_type == 'cpu':
            self.push_solver = PushAndSolver3d_cpu(self.__config__)
            self.init_plasma = init_plasma_cpu
            self.load_plasma = load_plasma_cpu
            self.beam_module = beam3d_cpu
        elif self.pu_type == 'gpu':
            self.push_solver = PushAndSolver3d_gpu(self.__config__)
            self.init_plasma = init_plasma_gpu
            self.load_plasma = load_plasma_gpu
            self.beam_module = beam3d_gpu

        # Finally, we set the diagnostics.
        if type(self.diagnostics) != list and self.diagnostics is not None:
            # If a user set only one diag. class:
            self.diagnostics = [self.diagnostics]
        self.__diagnostics__ = Diagnostics3d(self.__config__, self.diagnostics)

    def load_beamfile(self, path_to_beamfile='beamfile.npz'):
        beam_particles = self.beam_module.BeamParticles()
        beam_particles.load(path_to_beamfile)

        self.beam_source = self.beam_module.BeamSource(self.__config__,
                                                       beam_particles)
        self.beam_drain  = self.beam_module.BeamDrain()

    def load_plasmastate(self):
        self.loaded_fields, self.loaded_particles, self.loaded_currents =\
            self.load_plasma(self.__config__, self.path_to_plasmastate)
    
    def init_plasmastate(self):
        # Initializes a plasma state:
        pl_fields, pl_particles, pl_currents, pl_const_arrays =\
            self.init_plasma(self.__config__)

        # In case of an external plasma state, we set values
        # as the loaded values:
        if self.external_plasmastate:
            pl_fields, pl_particles, pl_currents =\
                self.loaded_fields, self.loaded_particles, self.loaded_currents

        return pl_fields, pl_particles, pl_currents, pl_const_arrays

    def step(self, N_steps=None):
        """Compute N time steps."""
        # 0. It analyzes config values:
        self.pull_config()

        # 1. If we use an external plasma state, we load it:
        if self.external_plasmastate:
            self.load_plasmastate()

        # t step function, makes N_steps time steps.
        if N_steps is None:
            N_steps = int(self.time_limit / self.time_step_size)
            print("Since the number of time steps hasn't been set explicitly,",
                  f"the code will simulate {N_steps} time steps with a time",
                  f"step size = {self.time_step_size}.")

        # 2. Checks for plasma continuation mode:
        if self.cont_mode == 'n' or self.cont_mode == 'no':
            # 3. If a beam source is empty (None), we generate
            #    a new beam according to set parameters:
            if self.beam_source is None:
                # Check for a beam being not rigid.
                if self.rigid_beam == 'n' or self.rigid_beam == 'no':
                    # Generate all parameters for a beam:
                    beam_particles = generate_beam(self.__config__,
                                                   self.beam_parameters,
                                                   self.beam_module)

                    # Here we create a beam source and a beam drain:
                    self.beam_source = self.beam_module.BeamSource(
                                                self.__config__, beam_particles)
                    self.beam_drain  = self.beam_module.BeamDrain()

                # A rigid beam mode has not been implemented yet. If you are
                # writing rigid beam mode, just use rigid_beam_current(...) from
                # lcode2dPy.alt_beam_generator.beam_generator
                else:
                    print("Sorry, for now, only 'no' mode of rigid-beam is",
                          "supported.")
                    raise Exception("Please, read the text above in your window.")

            # 4. A loop that calculates N time steps:
            for t_i in range(N_steps):
                pl_fields, pl_particles, pl_currents, pl_const_arrays =\
                    self.init_plasmastate()

                # Calculates one time step:
                self.push_solver.step_dt(pl_fields, pl_particles,
                                        pl_currents, pl_const_arrays,
                                        self.beam_source, self.beam_drain,
                                        self.current_time, self.__diagnostics__)

                # Perform diagnostics
                self.__diagnostics__.dt(self.current_time,
                    pl_particles, pl_fields, pl_currents, self.beam_drain)

                # Here we transfer beam particles from beam_buffer to
                # beam_source for the next time step. And create a new beam
                # drain that is empty.
                self.beam_source = self.beam_module.BeamSource(self.__config__,
                                                    self.beam_drain.beam_buffer)
                self.beam_drain  = self.beam_module.BeamDrain()

                self.current_time = self.current_time + self.time_step_size
                print()

            # 4. As in lcode2d, we save the beam state on reaching the time limit:
            self.beam_source.beam.save('beamfile') # Do we need it?
            print('\nThe work is done!')

        # Other plasma continuation mode has not been implemented yet.
        # If you are writing these modes, just change where you put
        # init_plasma(...) and generate_beam(...)
        else:
            print("Sorry, for now, only 'no' mode of plasma continuation is", 
                  "supported.")
            raise Exception("Please, read the text above in your window.")
