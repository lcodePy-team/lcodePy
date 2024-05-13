
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from ..utils import Diagnostic, OutputType
from ...config.config import Config

from .Strategy3D import _3D_FXi
from .StrategyCircular import _CIRC_FXi
from .Strategy import Strategy


class FXiType:
    __slots__ = ()

    Ex = 0x1
    Ey = 0x2
    Ez = 0x4
    E = Ex | Ey | Ez
    Er = Ex
    Ef = Ey

    Bx = 0x8
    By = 0x10
    Bz = 0x20
    B = Bx | By | Bz
    Bf = Bx
    EB = E | B

    Phi = 0x40

    ne = 0x80
    ni = 0x100
    n = ne | ni

    rho_beam = 0x200

    Sf = 0x400
    dx_chaotic = 0x800
    dy_chaotic = 0x1000
    dx_chaotic_perp = 0x2000
    dy_chaotic_perp = 0x4000



class FXiDiag(Diagnostic):

    def __init__(self, output_period = 100, saving_xi_period = 100, f_xi = FXiType.Ez,
                 output_type = OutputType.NUMBERS, probe_lines = None, directory_name = None):
        
        self.__f_xi = f_xi
        self.__output_period = output_period
        self.__saving_xi_period = saving_xi_period # not implemented
        self.__output_type = output_type
        self._probe_lines = probe_lines

        if directory_name is None:
            self.__directory = Path('diagnostics')
        else:
            self.__directory = Path('diagnostics') / directory_name

        self._data = defaultdict(list)
        self._data['xi'] = []
        self.__is_first_step = True


    def pull_config(self, config: Config):
        if config.get('processing-unit-type') == 'gpu':
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        
        self.strategy: _CIRC_FXi | _3D_FXi = Strategy(config, self)
    
        self.__grid_step_size = config.getfloat('transverse-step')
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size   = config.getfloat('xi-step')

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size

        if self.__saving_xi_period < self.__xi_step_size:
            self.__saving_xi_period = self.__xi_step_size

    def after_step_dxi(self, current_time, xi_plasma_layer,
                       plasma_particles, plasma_fields, plasma_currents,
                       rho_beam):
        if self.__is_first_step or self.condition_check(current_time, self.__output_period, self.__time_step_size):
            
            if self.__f_xi & FXiType.Ex:
                self.strategy.process_field(plasma_fields, 'Ex')
            if self.__f_xi & FXiType.Ey:
                self.strategy.process_field(plasma_fields, 'Ey')
            if self.__f_xi & FXiType.Ez:
                self.strategy.process_field(plasma_fields, 'Ez')
            if self.__f_xi & FXiType.Bx:
                self.strategy.process_field(plasma_fields, 'Bx')
            if self.__f_xi & FXiType.By:
                self.strategy.process_field(plasma_fields, 'By')
            if self.__f_xi & FXiType.Bz:
                self.strategy.process_field(plasma_fields, 'Bz')
            
            if self.__f_xi & FXiType.ne:
                self.strategy.process_n(plasma_currents, 'ne')
            
            if self.__f_xi & FXiType.ni:
                self.strategy.process_n(plasma_currents, 'ni')
            
            if self.__f_xi & FXiType.rho_beam:
                self.strategy.process_rho(rho_beam)
            
            if self.__f_xi & FXiType.Phi:
                self.strategy.process_Phi(plasma_fields)

            if self.__f_xi & FXiType.Sf:
                self.strategy.process_Sf(self.__grid_step_size, plasma_particles, plasma_fields)
            if self.__f_xi & FXiType.dx_chaotic:
                self.strategy.process_chaotic('dx', plasma_particles)
            if self.__f_xi & FXiType.dy_chaotic:
                self.strategy.process_chaotic('dy', plasma_particles)
            if self.__f_xi & FXiType.dx_chaotic_perp:
                self.strategy.process_chaotic_perp('dx', plasma_particles)
            if self.__f_xi & FXiType.dy_chaotic_perp:
                self.strategy.process_chaotic_perp('dy', plasma_particles)


           
            self._data['xi'].append(xi_plasma_layer)

    def dump(self, current_time, xi_plasma_layer,
             plasma_particles, plasma_fields, plasma_currents,
             beam_drain):
        if not self.__is_first_step and not self.condition_check(current_time, self.__output_period, self.__time_step_size):
            return
        Path(self.__directory).mkdir(parents=True, exist_ok=True)
        dirname = self.__directory / f'f_xi_{current_time:08.2f}.npz'

        if self.__output_type & OutputType.NUMBERS:
            self.xp.savez(dirname, **self._data)
        
        if self.__output_type & OutputType.PICTURES:
            for name, data in self._data.items():
                if name == 'xi':
                    continue
                self.__make_picture(current_time, self._data['xi'], name, data)

        self._data = defaultdict(list)
        self.__is_first_step = False

    def __make_picture(self, current_time, xi, name, data):

        plt.plot(xi, data)
        plt.title(name)
        plt.savefig(self.__directory / f'{name}_{current_time:08.2f}.jpg')
        plt.close()

