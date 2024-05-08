
from ..config import Config
from .utils import Diagnostic, OutputType, get, absremainder
from pathlib import Path
import numpy as np
from collections import defaultdict
# from math import remainder

class SliceType:
    __slots__ = ()
    
    XI_R = 0x1
    XI_X = 0x2
    XI_Y = 0x4
    X_Y = 0x8

class SliceValue:
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
    rho = 0x200

class SliceDiag(Diagnostic):
    def __init__(self,  slice_type, slice_value, output_type,
                 limits = None, offset = 0, output_period = 100, saving_xi_period = 100, directory_name = None):
        self._slice_type = slice_type
        self._slice_value = slice_value
        self._output_type = output_type
        self._limits = limits
        self._offset = offset
        self._output_period = output_period
        self._saving_xi_period = saving_xi_period

        if directory_name is None:
            self._directory = Path('diagnostics')
        else:
            self._directory = Path('diagnostics') / directory_name

        self._data = defaultdict(list) # TODO
        
    def pull_config(self, config: Config):

        self._config = config

        self._time_step_size = config.getfloat('time-step')
        self._xi_step_size   = config.getfloat('xi-step')

        if self._output_period < self._time_step_size:
            self._output_period = self._time_step_size

        # select xp
        if config.get('processing-unit-type') == 'gpu':
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np

        # select slicer
        geometry = config.get('geometry')
        if geometry == '3d':
            if self._slice_type == SliceType.XI_X:
                self._slicer = _3D_XI_X_Slicer(self)
            elif self._slice_type == SliceType.XI_Y:
                self._slicer = _3D_XI_Y_Slicer(self)
            elif self._slice_type == SliceType.X_Y:
                self._slicer = _3D_X_Y_Slicer(self)
            else:
                raise NotImplementedError

        elif geometry == '2d':
            if self._slice_type == SliceType.XI_R:
                self._slicer = _2D_XI_R_Slicer(self)
            elif self._slice_type == SliceType.X_Y:
                raise NotImplementedError
                self._slicer = _2D_X_Y_Slicer
            else:
                raise NotImplementedError
            
        # process limits


    def after_step_dxi(self, *args, **kwargs):
        self._slicer.process(*args, **kwargs)
    
    def dump(self, current_time, xi_plasma_layer,
             plasma_particles, plasma_fields, plasma_currents,
             beam_drain):
            
        if self.condition_check(current_time, self._output_period, self._time_step_size):
            Path(self._directory).mkdir(parents=True, exist_ok=True)
            dirname = self._directory / f'slice_{current_time:08.2f}.npz'

            for name in self._data.keys():
                self._data[name] = np.array(self._data[name])

            if self._output_type & OutputType.NUMBERS:
                np.savez(dirname, **self._data)
            
            # if self._output_type & OutputType.PICTURES:
            #     for name, data in self._data.items():
            #         if name == 'xi':
            #             continue
            #         self.__make_picture(current_time, self._data['xi'], name, data)

            self._data = defaultdict(list)
        
class _3D_XI_X_Slicer:
    def __init__(self, diag: SliceDiag):
        self._diag = diag
        window_length = diag._config.getfloat('window-length')
        xi_step = diag._config.getfloat('xi-step')
        x_steps = diag._config.getint('window-width-steps')
        window_width = diag._config.getfloat('window-width')
        grid_step_size = diag._config.getfloat('window-width-step-size')
        limits = diag._limits
        if limits is None:
            from_1 = 0
            to_1 = -window_length
            from_2 = -window_width / 2
            to_2 = window_width / 2
        else:
            if type(limits) == tuple or type(limits) == list or type(limits) == np.ndarray:
                limits = np.array(limits)
            
            if limits.ndim != 2:
                raise ValueError("limits must be 2D array")
            
            from_1 = limits[0, 0]
            to_1 = limits[0, 1]
            from_2 = limits[1, 0]
            to_2 = limits[1, 1]
        
        # self.from_1 = int(np.clip(from_1 // xi_step, 0, window_length//xi_step - 1))
        # self.to_1 = int(np.clip(to_1 // xi_step, 0, window_length//xi_step - 1))
        self.from_1 = from_1
        self.to_1 = to_1

        self.from_2 = int(np.clip(from_2 // grid_step_size + x_steps // 2, 0, x_steps - 1))
        self.to_2 = int(np.clip(to_2 // grid_step_size + x_steps // 2, 0, x_steps - 1))

        self.offset = int(np.clip(x_steps // 2 + diag._offset // grid_step_size, 0, x_steps - 1))
    def process(self, current_time, xi_plasma_layer,
                plasma_particles, plasma_fields, plasma_currents,
                rho_beam):
        
        if xi_plasma_layer < self.to_1 or xi_plasma_layer > self.from_1:
            return

        if self._diag._slice_value & SliceValue.Ex:
            val = getattr(plasma_fields, 'Ex')[self.from_2:self.to_2, self.offset]
            self._diag._data['Ex'].append(get(val))
        if self._diag._slice_value & SliceValue.Ey:
            val = getattr(plasma_fields, 'Ey')[self.from_2:self.to_2, self.offset]
            self._diag._data['Ey'].append(get(val))
        if self._diag._slice_value & SliceValue.Ez:
            val = getattr(plasma_fields, 'Ez')[self.from_2:self.to_2, self.offset]
            self._diag._data['Ez'].append(get(val))
        if self._diag._slice_value & SliceValue.Bx:
            val = getattr(plasma_fields, 'Bx')[self.from_2:self.to_2, self.offset]
            self._diag._data['Bx'].append(get(val))
        if self._diag._slice_value & SliceValue.By:
            val = getattr(plasma_fields, 'By')[self.from_2:self.to_2, self.offset]
            self._diag._data['By'].append(get(val))
        if self._diag._slice_value & SliceValue.Bz:
            val = getattr(plasma_fields, 'Bz')[self.from_2:self.to_2, self.offset]
            self._diag._data['Bz'].append(get(val))
        if self._diag._slice_value & SliceValue.ne:
            val = plasma_currents.ro[0, self.from_2:self.to_2, self.offset]
            self._diag._data['ne'].append(get(val))
        if self._diag._slice_value & SliceValue.ni:
            val = plasma_currents.ro[1, self.from_2:self.to_2, self.offset]
            self._diag._data['ni'].append(get(val))
        if self._diag._slice_value & SliceValue.rho:
            val = rho_beam[self.from_2:self.to_2, self.offset]
            self._diag._data['rho_beam'].append(get(val))
        if self._diag._slice_value & SliceValue.Phi:
            val = plasma_fields.Phi[self.from_2:self.to_2, self.offset]
            self._diag._data['Phi'].append(get(val))


class _3D_XI_Y_Slicer:
    def __init__(self, diag: SliceDiag):
        self._diag = diag
        window_length = diag._config.getfloat('window-length')
        xi_step = diag._config.getfloat('xi-step')
        x_steps = diag._config.getint('window-width-steps')
        window_width = diag._config.getfloat('window-width')
        grid_step_size = diag._config.getfloat('window-width-step-size')
        limits = diag._limits
        if limits is None:
            from_1 = 0
            to_1 = -window_length
            from_2 = -window_width / 2
            to_2 = window_width / 2
        else:
            if type(limits) == tuple or type(limits) == list or type(limits) == np.ndarray:
                limits = np.array(limits)
            
            if limits.ndim != 2:
                raise ValueError("limits must be 2D array")
            
            from_1 = limits[0, 0]
            to_1 = limits[0, 1]
            from_2 = limits[1, 0]
            to_2 = limits[1, 1]
        
        self.from_1 = from_1
        self.to_1 = to_1

        self.from_2 = int(np.clip(from_2 // grid_step_size + x_steps // 2, 0, x_steps - 1))
        self.to_2 = int(np.clip(to_2 // grid_step_size + x_steps // 2, 0, x_steps - 1))

        self.offset = int(np.clip(x_steps // 2 + diag._offset // grid_step_size, 0, x_steps - 1))
    def process(self, current_time, xi_plasma_layer,
                plasma_particles, plasma_fields, plasma_currents,
                rho_beam):
        if xi_plasma_layer < self.to_1 or xi_plasma_layer > self.from_1:
            return
        if self._diag._slice_value & SliceValue.Ex:
            val = getattr(plasma_fields, 'Ex')[self.offset, self.from_2:self.to_2]
            self._diag._data['Ex'].append(get(val))
        if self._diag._slice_value & SliceValue.Ey:
            val = getattr(plasma_fields, 'Ey')[self.offset, self.from_2:self.to_2]
            self._diag._data['Ey'].append(get(val))
        if self._diag._slice_value & SliceValue.Ez:
            val = getattr(plasma_fields, 'Ez')[self.offset, self.from_2:self.to_2]
            self._diag._data['Ez'].append(get(val))
        if self._diag._slice_value & SliceValue.Bx:
            val = getattr(plasma_fields, 'Bx')[self.offset, self.from_2:self.to_2]
            self._diag._data['Bx'].append(get(val))
        if self._diag._slice_value & SliceValue.By:
            val = getattr(plasma_fields, 'By')[self.offset, self.from_2:self.to_2]
            self._diag._data['By'].append(get(val))
        if self._diag._slice_value & SliceValue.Bz:
            val = getattr(plasma_fields, 'Bz')[self.offset, self.from_2:self.to_2]
            self._diag._data['Bz'].append(get(val))
        if self._diag._slice_value & SliceValue.ne:
            val = plasma_currents.ro[0, self.offset, self.from_2:self.to_2]
            self._diag._data['ne'].append(get(val))
        if self._diag._slice_value & SliceValue.ni:
            val = plasma_currents.ro[1, self.offset, self.from_2:self.to_2]
            self._diag._data['ni'].append(get(val))
        if self._diag._slice_value & SliceValue.rho:
            val = rho_beam[self.offset, self.from_2:self.to_2]
            self._diag._data['rho_beam'].append(get(val))
        if self._diag._slice_value & SliceValue.Phi:
            val = plasma_fields.Phi[self.offset, self.from_2:self.to_2]
            self._diag._data['Phi'].append(get(val))

class _3D_X_Y_Slicer:
    def __init__(self, diag):
        self._diag = diag
        window_length = diag._config.getfloat('window-length')
        xi_step = diag._config.getfloat('xi-step')
        xi_steps = window_length // xi_step
        x_steps = diag._config.getint('window-width-steps')
        window_width = diag._config.getfloat('window-width')
        grid_step_size = diag._config.getfloat('window-width-step-size')
        limits = diag._limits
        if limits is None:
            from_1 = -window_width / 2
            to_1 = window_width / 2
            from_2 = -window_width / 2
            to_2 = window_width / 2
        else:
            if type(limits) == tuple or type(limits) == list or type(limits) == np.ndarray:
                limits = np.array(limits)
            
            if limits.ndim != 2:
                raise ValueError("limits must be 2D array")
            
            from_1 = limits[0, 0]
            to_1 = limits[0, 1]
            from_2 = limits[1, 0]
            to_2 = limits[1, 1]
        
        self.from_1 = int(np.clip(from_1 // grid_step_size + x_steps // 2, 0, x_steps - 1))
        self.to_1 = int(np.clip(to_1 // grid_step_size + x_steps // 2, 0, x_steps - 1))

        self.from_2 = int(np.clip(from_2 // grid_step_size + x_steps // 2, 0, x_steps - 1))
        self.to_2 = int(np.clip(to_2 // grid_step_size + x_steps // 2, 0, x_steps - 1))

        # self.offset = int(np.clip(x_steps // 2 + diag._offset // grid_step_size, 0, xi_steps - 1))
    
    def conditions_check(self, current_time, xi_plasma_layer):
        return (
            absremainder(current_time,
                         self._diag._output_period) <= self._diag._time_step_size / 2 and
            absremainder(xi_plasma_layer,
                         self._diag._saving_xi_period) <= self._diag._xi_step_size / 2)
    
    def process(self, current_time, xi_plasma_layer,
                plasma_particles, plasma_fields, plasma_currents,
                rho_beam):
        if not self.conditions_check(current_time, xi_plasma_layer):
            return
        if self._diag._slice_value & SliceValue.Ex:
            val = getattr(plasma_fields, 'Ex')[self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['Ex'].append(get(val))
        if self._diag._slice_value & SliceValue.Ey:
            val = getattr(plasma_fields, 'Ey')[self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['Ey'].append(get(val))
        if self._diag._slice_value & SliceValue.Ez:
            val = getattr(plasma_fields, 'Ez')[self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['Ez'].append(get(val))
        if self._diag._slice_value & SliceValue.Bx:
            val = getattr(plasma_fields, 'Bx')[self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['Bx'].append(get(val))
        if self._diag._slice_value & SliceValue.By:
            val = getattr(plasma_fields, 'By')[self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['By'].append(get(val))
        if self._diag._slice_value & SliceValue.Bz:
            val = getattr(plasma_fields, 'Bz')[self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['Bz'].append(get(val))
        if self._diag._slice_value & SliceValue.ne:
            val = plasma_currents.ro[0, self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['ne'].append(get(val))
        if self._diag._slice_value & SliceValue.ni:
            val = plasma_currents.ro[1, self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['ni'].append(get(val))
        if self._diag._slice_value & SliceValue.rho:
            val = rho_beam[self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['rho_beam'].append(get(val))
        if self._diag._slice_value & SliceValue.Phi:
            val = plasma_fields.Phi[self.from_1:self.to_1, self.from_2:self.to_2]
            self._diag._data['Phi'].append(get(val))


class _2D_XI_R_Slicer:
    def __init__(self, diag: SliceDiag):
        self._diag = diag
        window_length = diag._config.getfloat('window-length')
        xi_step = diag._config.getfloat('xi-step')
        window_width = diag._config.getfloat('window-width')
        grid_step_size = diag._config.getfloat('window-width-step-size')
        x_steps = window_width//grid_step_size + 1
        limits = diag._limits
        if limits is None:
            from_1 = 0
            to_1 = -window_length
            from_2 = 0
            to_2 = window_width
        else:
            if type(limits) == tuple or type(limits) == list or type(limits) == np.ndarray:
                limits = np.array(limits)
            
            if limits.ndim != 2:
                raise ValueError("limits must be 2D array")
            
            from_1 = limits[0, 0]
            to_1 = limits[0, 1]
            from_2 = limits[1, 0]
            to_2 = limits[1, 1]
        
        self.from_1 = from_1
        self.to_1 = to_1
        self.from_2 = int(np.clip(from_2 // grid_step_size, 0, x_steps - 1))
        self.to_2 = int(np.clip(to_2 // grid_step_size, 0, x_steps - 1))

    def process(self, current_time, xi_plasma_layer,
                plasma_particles, plasma_fields, plasma_currents,
                rho_beam):
        if xi_plasma_layer < self.to_1 or xi_plasma_layer > self.from_1:
            return
        # print('here', self.from_2, self.to_2)
        if self._diag._slice_value & SliceValue.Er:
            val = getattr(plasma_fields, 'E_r')[self.from_2:self.to_2]
            self._diag._data['Er'].append(get(val))
        if self._diag._slice_value & SliceValue.Ef:
            val = getattr(plasma_fields, 'E_f')[self.from_2:self.to_2]
            self._diag._data['Ef'].append(get(val))
        if self._diag._slice_value & SliceValue.Ez:
            val = getattr(plasma_fields, 'E_z')[self.from_2:self.to_2]
            self._diag._data['Ez'].append(get(val))
        if self._diag._slice_value & SliceValue.Bf:
            val = getattr(plasma_fields, 'B_f')[self.from_2:self.to_2]
            self._diag._data['Bf'].append(get(val))
        if self._diag._slice_value & SliceValue.By:
            pass
        if self._diag._slice_value & SliceValue.Bz:
            val = getattr(plasma_fields, 'B_z')[self.from_2:self.to_2]
            self._diag._data['Bz'].append(get(val))
        if self._diag._slice_value & SliceValue.ne:
            val = plasma_currents.ro[0, self.from_2:self.to_2]
            self._diag._data['ne'].append(get(val))
        if self._diag._slice_value & SliceValue.ni:
            val = plasma_currents.ro[1, self.from_2:self.to_2]
            self._diag._data['ni'].append(get(val))
        if self._diag._slice_value & SliceValue.rho:
            val = rho_beam[self.from_2:self.to_2]
            self._diag._data['rho_beam'].append(get(val))
        if self._diag._slice_value & SliceValue.Phi:
            raise NotImplementedError("Phi not implemented in 2d yet")

# TODO _2D_X_Y_Slicer 