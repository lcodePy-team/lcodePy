import numpy as np

from ...config import Config


_convert_field_2d = {
        'Ex': 'E_r',
        'Ey': 'E_f',
        'Ez': 'E_z',
        'Bx': 'B_f',
        'Bz': 'B_z',
}

class _CIRC_FXi:

    def __init__(self, config: Config, diagnostic):
        self.__config = config
        self.__diagnostic = diagnostic

        self.__process_probe_lines(diagnostic._probe_lines)

    def __process_probe_lines(self, probe_lines):
        grid_step_size = self.__config.getfloat('transverse-step')

        if probe_lines is None:
            self.__ax_r = 0
        else: 
            if type(probe_lines) == list or type(probe_lines) == np.ndarray:
                probe_lines = np.array(probe_lines)
            if probe_lines.ndim < 1 or probe_lines.ndim > 2:
                raise ValueError('probe_lines must be 1D or 2D')
            if probe_lines.ndim == 1:
                self.__ax_r = probe_lines
            else: # ndim == 2
                self.__ax_r = np.sqrt(probe_lines[0]**2 + probe_lines[1]**2)

        self.__ax_r = (np.round(self.__ax_r / grid_step_size)).astype(int)

    
    def process_field(self, plasma_fields, field):
        if field == 'By':
            # print("y-field in a circular geometry: Skip")
            return
        
        field = _convert_field_2d[field]
        val = getattr(plasma_fields, field)[self.__ax_r]
        self.__diagnostic._data[field].append(val)
    
    def process_Phi(self, plasma_fields):
        raise NotImplementedError("Not ready yet")
    
    def process_n(self, plasma_currents, n):
        idx = 0 if n == 'ne' else 1 # ni
        val = getattr(plasma_currents, 'rho')[idx, self.__ax_r]
        self.__diagnostic._data[n].append(val)

    def process_rho(self, rho_beam):
        val = rho_beam[self.__ax_r]
        self.__diagnostic._data['rho_beam'].append(val)
