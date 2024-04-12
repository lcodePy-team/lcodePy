
from ..config import Config
from .utils import Diagnostic, OUTPUT_TYPE
from pathlib import Path
import numpy as np

class Slice:
    __slots__ = ()
    
    XI_R = 0x1
    XI_X = 0x2
    XI_Y = 0x4
    X_Y = 0x8

class SliceType:
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

class BaseSlice(Diagnostic):

    def __new__ (self, slice = Slice.XI_R, *args, **kwargs):
        if slice == Slice.XI_R:
            return SliceXI_R(*args, **kwargs)
        if slice == Slice.XI_X:
            return SliceXI_X(*args, **kwargs)
        if slice == Slice.XI_Y:
            return SliceXI_Y(*args, **kwargs)
        if slice == Slice.X_Y:
            return SliceX_Y(*args, **kwargs)
        
        raise Exception("SliceType is not defined")
        

    def __init__(self, slice_type: SliceType, slice_limits = None, output_period = 100, output_type = OUTPUT_TYPE.NUMBERS, directory_name = None):
        self._output_period = output_period
        self._output_type = output_type
        self._slice_type = slice_type

        if directory_name is None:
            self._directory = Path('diagnostics')
        else:
            self._directory = Path('diagnostics') / directory_name

        self._limits = slice_limits

from collections import defaultdict

class SliceXI_R(BaseSlice):
    
    def __init__(self, slice_limits = None, *args, **kwargs):
        super().__init__(slice_limits, *args, **kwargs)
        self._data = defaultdict(list)

    def pull_config(self, config: Config):
        if config.get('geometry') != '2d':
            raise ValueError('SliceXI_R can be used only for 2d')
        
        self._time_step_size = config.getfloat('time-step')

        xi_step = config.getfloat('xi-step')
        xi_length = config.getfloat('window-length')
        r_step = config.getfloat('window-width-step-size')
        r_length = config.getfloat('window-width')
        

        if self._limits is None:
            self._limits = [[0, xi_length], [0, r_length]]    
        self._limits = np.array(self._limits)    
        if self._limits.shape != (2, 2):
            raise ValueError('limits must be 2x2 array')
        np.clip(self._limits, [0, 0], [xi_length-xi_step, r_length-r_step], self._limits) 
        self._limits[0] = self._limits[0] / xi_step
        self._limits[1] = self._limits[1] / r_step
        self._limits = self._limits.astype(int)
    
    def after_step_dxi(self, current_time, xi_plasma_layer,
                       plasma_particles, plasma_fields, plasma_currents,
                       rho_beam):
        if self.condition_check(current_time, self._output_period, self._time_step_size):
            if self._slice_type & SliceType.Er:
                self._process_field(plasma_fields, 'E_r')
            if self._slice_type & SliceType.Ef:
                self._process_field(plasma_fields, 'E_f')
            if self._slice_type & SliceType.Ez:
                self._process_field(plasma_fields, 'E_z')
            if self._slice_type & SliceType.Bf:
                self._process_field(plasma_fields, 'B_f')
            if self._slice_type & SliceType.Bz:
                self._process_field(plasma_fields, 'B_z')
            


    def _process_field(self, plasma_fields, field_name):
        pass


    
class SliceX_Y:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class SliceXI_X:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class SliceXI_Y:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
