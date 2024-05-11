
from math import inf
from pathlib import Path
import numpy as np

from lcode.config.config import Config
from .utils import Diagnostic, absremainder, get

class RunStateDiag(Diagnostic):

    def __init__(self, 
                 save_beam:bool = False, 
                 save_plasma:bool = False,
                 output_period: float = 1000,
                 saving_xi_period: float = inf,
                 directory_name: str = None
                 ):
        
        self.__save_beam = save_beam
        self.__save_plasma = save_plasma
        self.__output_period = output_period
        self.__saving_xi_period = saving_xi_period

        if directory_name is None:
            self.__directory = Path('diagnostics')
        else:
            self.__directory = Path('diagnostics') / directory_name

    def pull_config(self, config: Config):
        self.__geometry = config.get('geometry')
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size = config.getfloat('xi-step')

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size
        if self.__saving_xi_period < self.__xi_step_size:
            self.__saving_xi_period = self.__xi_step_size
        
        is_rigid_beam = config.getbool('rigid-beam')
        if is_rigid_beam and self.__save_beam:
            raise Exception(
                "We cannot save the beam in the case of a rigid beam."
                "Please, change save_beam to False."
                )
        
    def conditions_check(self, current_time, xi_plasma_layer):
        return (
            absremainder(current_time,
                         self.__output_period) <= self.__time_step_size / 2 and
            absremainder(xi_plasma_layer,
                         self.__saving_xi_period) <= self.__xi_step_size / 2)
    
    def after_step_dxi(self, current_time, xi_plasma_layer,
                plasma_particles, plasma_fields, plasma_currents,
                rho_beam):
        
        if not self.conditions_check(current_time, xi_plasma_layer):
            return
        if not self.__save_plasma:
            return
        
        Path(self.__directory / "snapshots").mkdir(parents=True, exist_ok=True)

        e_filename = self.__directory / "snapshots" /f"plasma_electrons_{current_time:08.2f}_{xi_plasma_layer:+09.2f}.npz"
        i_filename = self.__directory / "snapshots" / f"plasma_ions_{current_time:08.2f}_{xi_plasma_layer:+09.2f}.npz"
        
        plasma_particles['electrons'].save(e_filename)
        plasma_particles['ions'].save(i_filename)

    def dump(self, current_time, xi_plasma_layer, plasma_particles, plasma_fields, plasma_currents, beam_drain):
        if absremainder(current_time, self.__output_period) > self.__time_step_size / 2:
            return
        
        Path(self.__directory).mkdir(parents=True, exist_ok=True)

        if self.__save_beam:
            filename = self.__directory / f"beam_{current_time:08.2f}.npz"
            
            beam_drain.save(filename)
        
        if self.__save_plasma:
            e_filename = self.__directory / f"plasma_electrons_{current_time:08.2f}.npz"
            i_filename = self.__directory / f"plasma_ions_{current_time:08.2f}.npz"

            plasma_particles['electrons'].save(e_filename)
            plasma_particles['ions'].save(i_filename)
