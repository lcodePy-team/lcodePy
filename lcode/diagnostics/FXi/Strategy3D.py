import numpy as np

from ...config import Config
from ..utils import get



class _3D_FXi:
    def __init__(self, config: Config, diagnostic):
        self.__config = config
        self.__diagnostic = diagnostic

        self.__process_probe_lines(diagnostic._probe_lines)

    def __process_probe_lines(self, probe_lines):
        steps = self.__config.getint('window-width-steps')
        grid_step_size = self.__config.getfloat('transverse-step')

        if probe_lines is None:
            self.__ax_x = 0
            self.__ax_y = 0
        else:
            if type(probe_lines) == list or type(probe_lines) == np.ndarray:
                probe_lines = np.array(probe_lines)
            
            if probe_lines.ndim < 1 or probe_lines.ndim > 2:
                raise ValueError('probe_lines must be 1D or 2D')
            
            if probe_lines.ndim == 1:
                self.__ax_x = probe_lines
                self.__ax_y = np.zeros_like(probe_lines)
            else: # ndim == 2
                self.__ax_x = probe_lines[0]
                self.__ax_y = probe_lines[1]

        self.__ax_x = (steps // 2 +
                       np.round(self.__ax_x / grid_step_size)).astype(int)
        self.__ax_y = (steps // 2 +
                       np.round(self.__ax_y / grid_step_size)).astype(int)

    def process_field(self, plasma_fields, field):

        val = getattr(plasma_fields, field)[self.__ax_x, self.__ax_y]
        self.__diagnostic._data[field].append(get(val))
    
    def process_n(self, plasma_currents, n):
        idx = 0 if n == 'ne' else 1 # ni
        val = plasma_currents.ro[idx, self.__ax_x, self.__ax_y]
        self.__diagnostic._data[n].append(get(val))

    def process_rho(self, rho_beam):
        val = rho_beam[self.__ax_x, self.__ax_y]
        self.__diagnostic._data['rho_beam'].append(get(val))
    
    def process_Phi(self, plasma_fields):
        val = plasma_fields.Phi[self.__ax_x, self.__ax_y]
        self.__diagnostic._data['Phi'].append(get(val))