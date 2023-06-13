"""Top-level three-dimensional simulation class."""
import copy

# Imports config, diagnostics, alternative beam generator for 3d
# (can be used for 2d too)
from .config.default_config_values import default_config_values
from .config.config import Config
from .alt_beam_generator.beam_generator import generate_beam

# Imports plasma nodule, 3d:
from .push_solvers.push_solver import PusherAndSolver3D
from .plasma3d import init_plasma_3d, load_plasma_3d

# Imports beam module, 3d:
from .beam3d import BeamParticles3D, BeamSource3D, BeamDrain3D

# Imports plasma module, 2d:
from .push_solvers.push_solver import PusherAndSolver2D
from .plasma import init_plasma_2d, load_plasma_2d

# Imports beam module, 3d:
from .beam import BeamParticles2D, BeamSource2D, BeamDrain2D


class Simulation:
    """
    Top-level lcodePy simulation class that controls computational.

    This class contains configuration of simulation and controls diagnostics.
    """
    def __init__(self, config=default_config_values, beam_parameters={},
                 diagnostics=[]):
        """
            Initializes a simulation.

            Parametrs
            ---------
            config : Dict, optional
                    The set of base parameters to perform the simulation.
                    Default : default_config_values from lcodePy2d/config/
                    (TODO: Should we use Config class insted Dict by default?)

            beam_parametrs : Dict, otianal
                    Configuration of the charge beam.
                    The beam desibled by default.      
                    (TODO: set default beam)

            diagnostics : List, optianal
                    Collection of diagnostics that should be run.
                    By default, diagnostics are desibled.      
        """

        self.config = copy.copy(config)
        self.beam_parameters = copy.copy(beam_parameters)
        self.diagnostics_list = copy.copy(diagnostics)

        # Here we set parameters for beam generation, where we will store beam
        # particles and where they will go after calculations
        # self.beam_parameters = beam_parameters
        # self.beam_particle_dtype = particle_dtype3d

        # We use this time as a general time value:
        self.current_time = 0.
        # The user can change this value for diagnostic convenience.

        # We initialize a beam source and a beam drain:
        self.beam_source = None
        self.beam_drain = None

        # We set that initially the code doesn't use an external plasma state:
        self.external_plasmastate = False
        self.path_to_plasmastate = 'plasmastate.npz'

        #set geometry is None before reading config
        self.__geometry = None

        # Pull the config before the start of calculations:
        self.__pull_config()
    
    def __pull_config(self):
        # 0. We set __config__ as a Config class instance:
        self.__config = Config(self.config)

        # We set some instance variables:
        self.__time_limit = self.__config.getfloat('time-limit')
        self.__time_step_size = self.__config.getfloat('time-step')
        self.__rigid_beam = self.__config.get('rigid-beam')

        # Mode of plasma continuation:
        self.__cont_mode = self.__config.get('continuation')

        # Firstly, we check that the geomtry was set right:
        if self.__geometry is None:
            self.__geometry = self.__config.get('geometry').lower()
        elif self.__geometry != self.__config.get('geometry').lower():
            raise Exception("Sorry, update geometry does not support now.")
            
        if self.__geometry == '3d':
            self.__push_solver = PusherAndSolver3D(config=self.__config)
            self.init_plasma, self.__load_plasma = \
                init_plasma_3d, load_plasma_3d
            self.BeamParticles, self.BeamSource, self.BeamDrain = \
                BeamParticles3D, BeamSource3D, BeamDrain3D
        elif self.__geometry == '2d':
            self.__push_solver = PusherAndSolver2D(config=self.__config)
            self.init_plasma, self.__load_plasma = \
                init_plasma_2d, load_plasma_2d
            self.BeamParticles, self.BeamSource, self.BeamDrain = \
                BeamParticles2D, BeamSource2D, BeamDrain2D
        else:
            raise Exception("Sorry, you set a wrong type of geometry. " +
                            "For now, we support only 2d and 3d geometry.")
            

        # Finally, we set the diagnostics.
        for diagnostic in self.diagnostics_list:
            diagnostic.pull_config(config=self.__config)

    
    ## TODO add update beam and diagnostics
    ## def update(self, config):
    ##    """
    ##    Update the simulation according modification. 


    # TODO: Should we make load_beamfile just
    #       another method of beam genetration?
    def load_beamfile(self, path_to_beamfile='beamfile.npz'):
        beam_particles = self.BeamParticles(self.__config.xp)
        beam_particles.load(path_to_beamfile)

        self.beam_source = self.BeamSource(self.__config, beam_particles)
        self.beam_drain  = self.BeamDrain(self.__config)

    # def add_beamfile(self, path_to_beamfile='new_beamfile.npz'):
    #     """Add a new beam that is loaded from 'path_to_beamfile' to the beam source.
    #     """
    #     pass

    def __load_plasmastate(self):
        # We use this function to load plasma only once and then use
        # it while it is loaded into the device's memory (CPU or GPU).
        self.__loaded_plasmastate =\
            self.__load_plasma(self.__config, self.path_to_plasmastate)

    def __init_plasmastate(self, current_time):
        # In case of an external plasma state, we set values
        # as the loaded values:
        if self.external_plasmastate:
            return self.__loaded_plasmastate
        else:
            # Initializes a plasma state:
            return self.init_plasma(self.__config, current_time)
            # The init_plasma function must be public so that a user
            # can change it and generate a unique plasma.
            # TODO: make the insides of init_plasma accessible for
            #       modifications after copy-pasting.

    def step(self, N_steps=None):
        """
        Compute N time steps.

        Parametrs
        ---------

        N_steps : int, optional
                Number of time steps that will be done. 
                Default : N_steps = time_limit / time_step.
        """
        # 0. It analyzes config values:
        #TODO: explicit config update. If we change xi-step we must change beam.
        self.__pull_config()

        # 1. If we use an external plasma state, we load it:
        if self.external_plasmastate:
            self.__load_plasmastate()

        # t step function, makes N_steps time steps.
        if N_steps is None:
            N_steps = int(self.__time_limit / self.__time_step_size)
            print("Since the number of time steps hasn't been set explicitly,",
                  f"the code will simulate {N_steps} time steps with a time",
                  f"step size = {self.__time_step_size}.")
        else:
            self.__time_limit = \
                N_steps * self.__time_step_size + self.current_time
            print("Since the number of time steps has been set explicitly,",
                  f"the code will simulate till time limit = {self.__time_limit},",
                  f"with a time step size = {self.__time_step_size}.")

        # 2. Checks for plasma continuation mode:
        if self.__cont_mode == 'n' or self.__cont_mode == 'no':
            # 3. If a beam source is empty (None), we generate
            #    a new beam according to set parameters:
            if self.beam_source is None:
                # Check for a beam being not rigid.
                if self.__rigid_beam == 'n' or self.__rigid_beam == 'no':
                    # Generate all parameters for a beam:
                    beam_particles = generate_beam(self.__config,
                                                   self.beam_parameters)

                    # Here we create a beam source and a beam drain:
                    self.beam_source = self.BeamSource(self.__config,
                                                       beam_particles)
                    self.beam_drain  = self.BeamDrain(self.__config)

                # A rigid beam mode has not been implemented yet. If you are
                # writing rigid beam mode, just use rigid_beam_current(...) from
                # ..alt_beam_generator.beam_generator
                else:
                    raise Exception("Sorry, for now, only 'no' mode of" +
                                    "rigid-beam is supported.")

            # 4. A loop that calculates N time steps:
            for t_i in range(N_steps):
                self.current_time = self.current_time + self.__time_step_size

                plasmastate = self.__init_plasmastate(self.current_time)

                # Calculates one time step:
                self.__push_solver.step_dt(
                    *plasmastate, self.beam_source, self.beam_drain,
                    self.current_time, self.diagnostics_list
                )

                # Here we transfer beam particles from beam_buffer to
                # beam_source for the next time step. And create a new beam
                # drain that is empty.
                beam_particles = self.beam_drain.beam_slice()
                self.beam_source = self.BeamSource(self.__config, 
                                                   beam_particles)
                self.beam_drain  = self.BeamDrain(self.__config)

            # 4. As in lcode2d, we save the beam state on reaching the time limit:
            # self.beam_source.beam.save('beamfile') # Do we need it?
            # TODO: Make checkpoints where all simulation information,
            #       including beam and current time, is saved.
            print('The work is done!')

        # Other plasma continuation mode has not been implemented yet.
        # If you are writing these modes, just change where you put
        # init_plasma(...) and generate_beam(...)
        else:
            raise Exception("Sorry, for now, only 'no' mode of plasma" +
                            "continuation is supported.")
