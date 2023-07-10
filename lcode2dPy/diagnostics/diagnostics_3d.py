from math import inf, ceil, remainder
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ..config.config import Config
from ..plasma3d.data import Arrays
from ..beam3d.beam_io import BeamDrain


# Auxiliary functions:

def get(array: np.ndarray):
    """
    A function to copy data from GPU to CPU when using cupy as the main data
    module. For now, this method is only used in diagnostics.
    """
    if 'numpy' in str(type(array)): # numpy
        return array
    else: # cupy
        return array.get() # auto-copies to host RAM


def from_str_into_list(names_str: str):
    """
    Makes a list of elements that it gets from a long string.
    """
    # For example: 'Ez, Ey, Ez ,Bx,,' becomes ['Ez', 'Ey', 'Ez', 'Bx']
    names = np.array(names_str.replace(' ', '').split(','))
    names = names[names != '']
    return names


def absremainder(x, y):
    """
    This is a universal modulo operator that behaves predictably not only for
    integers, but also for floating point numbers.
    """
    # Due to the nature of floating-number arithmetic, the regular % operator
    # doesn't work as expected for floats. Thus, we use math.remainder that 
    # behaves a little more predictably. Still, while 0.15 % 0.05 = 0.04999...,
    # math.remainder(0.15, 0.05) = -1.3877787807814457e-17. Also, math.remainder
    # isn't exactly the modulo operator like math.fmod and may produce results
    # such as math.remainder(14, 5) = -1.0, but it solves the puzzle of
    # floating-number arithmetic.
    return abs(remainder(x, y))


def conv_2d(arr: np.ndarray, merging_xi: int, merging_r: int):
    """
    Calculates strided convolution using a mean/uniform kernel.
    """
    new_arr = []
    for i in range(0, arr.shape[0], merging_xi):
        start_i, end_i = i, i + merging_xi
        if end_i > arr.shape[0]:
            end_i = arr.shape[0]

        for j in range(0, arr.shape[1], merging_r):
            start_j, end_j = j, j + merging_r
            if end_j > arr.shape[1]:
                end_j = arr.shape[1]

            new_arr.append(np.mean(arr[start_i:end_i, start_j:end_j]))

    return np.reshape(np.array(new_arr),
                      ceil(arr.shape[0] / merging_xi, -1))


def calculate_energy_fluxes(grid_step_size,
                            plasma_particles: Arrays, plasma_fields: Arrays):
    # xp is either np (numpy) or cp (cupy):
    xp = plasma_particles.xp

    # The perturbation to the flux density of the electromagnetic energy:
    Se = (plasma_fields.Ez ** 2 + plasma_fields.Bz ** 2 +
         (plasma_fields.Ex - plasma_fields.By) ** 2 +
         (plasma_fields.Ey + plasma_fields.Bx) ** 2) * grid_step_size ** 2 / 2

    # Additional array to calculate the motion energy of plasma particles:
    gamma_m = xp.sqrt(plasma_particles.m ** 2  + plasma_particles.px ** 2 +
                      plasma_particles.py ** 2 + plasma_particles.pz ** 2)
    motion_energy = gamma_m - plasma_particles.m

    # Calculate integral total energy flux and return it:
    psi_electromagnetic = xp.sum(Se)
    psi_total = xp.sum(motion_energy) + psi_electromagnetic
    return psi_total


# Diagnostics classes:
# TODO: Add a template class for diagnostics.

class DiagnosticsFXi:
    __allowed_f_xi = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho', 'rho_beam',
                      'Phi', 'Sf']
                    # 'ni']
                    # TODO: add them and functionality!
    __allowed_f_xi_type = ['numbers', 'pictures', 'both']
    #TODO: add 'pictures' and 'both' and functionality

    def __init__(self, output_period=100, saving_xi_period=100, f_xi='Ez',
                 f_xi_type='numbers', x_probe_lines=0, y_probe_lines=0,
                 unique_directory_name=''):
        # It creates a list of string elements such as Ez, Ex, By...
        self.__f_xi_names = from_str_into_list(f_xi)
        for name in self.__f_xi_names:
            if name not in self.__allowed_f_xi:
                raise Exception(f'{name} value is not supported as f(xi).')

        # Output mode for the functions of xi:
        if f_xi_type in self.__allowed_f_xi_type:
            self.__f_xi_type = f_xi_type
        else:
            raise Exception(f"{f_xi_type} type of f(xi) diagnostics is " +
                             "not supported.")

        # The position of `probe' lines from the center:
        if ((type(x_probe_lines) == list or type(x_probe_lines) == np.ndarray) and
            (type(y_probe_lines) == list or type(y_probe_lines) == np.ndarray)):
            self.__x_probe = np.array(x_probe_lines)
            self.__y_probe = np.array(y_probe_lines)
            if len(self.__x_probe) != len(self.__y_probe):
                raise Exception(f"Different number of x and y coordinates " +
                                 "for probe lines!")
        else:
            # If x_probe_lines and y_probe_lines are only a number,
            # as is the default.
            self.__x_probe = x_probe_lines
            self.__y_probe = y_probe_lines

        # Set time periodicity of detailed output and safving into a file:
        self.__output_period = output_period
        self.__saving_xi_period = saving_xi_period

        # Set unique name for data storing directory:
        self.__unique_directory_name = unique_directory_name

        # We store data as a simple Python dictionary of lists for f(xi) data.
        # But I'm not sure this is the best way to handle data storing!
        self.__data = {name: [] for name in self.__f_xi_names}
        self.__data['xi'] = []

    def __repr__(self):
        return (
            f"DiagnosticsFXi(output_period={self.__output_period}, " +
            f"saving_xi_period={self.__saving_xi_period}, " +
            f"f_xi='{','.join(self.__f_xi_names)}', " +
            f"f_xi_type='{self.__f_xi_type}', " +
            f"x_probe_lines={self.__x_probe}, " +
            f"y_probe_lines={self.__y_probe}, " +
            f"unique_directory_name={self.__unique_directory_name})")

    def pull_config(self, config: Config):
        """Pulls a config to get all required parameters."""
        steps = config.getint('window-width-steps')
        grid_step_size = config.getfloat('window-width-step-size')

        # We change how we store the positions of the 'probe' lines:
        self.__ax_x = (steps // 2 +
                       np.round(self.__x_probe / grid_step_size)).astype(int)
        self.__ax_y = (steps // 2 +
                       np.round(self.__y_probe / grid_step_size)).astype(int)

        # Here we check if the output period is less than the time step size.
        # In that case each time step is diagnosed. The first time step is
        # always diagnosed.
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size   = config.getfloat('xi-step')
        self.__grid_step_size = config.getfloat('window-width-step-size')

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size

        if self.__saving_xi_period < self.__xi_step_size:
            self.__saving_xi_period = self.__xi_step_size

    def after_step_dxi(self, current_time, xi_plasma_layer,
                       plasma_particles: Arrays, plasma_fields: Arrays,
                       plasma_currents: Arrays, ro_beam):
        if self.conditions_check(current_time, xi_plasma_layer):
            self.__data['xi'].append(xi_plasma_layer)

            for name in self.__f_xi_names:
                if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi']:
                    val = getattr(plasma_fields, name)[self.__ax_x, self.__ax_y]
                    self.__data[name].append(get(val))

                if name == 'rho':
                    # TODO: It's just a crutch!!!
                    val = getattr(plasma_currents, 'ro')[self.__ax_x, self.__ax_y]
                    self.__data[name].append(get(val))

                if name == 'rho_beam':
                    val = ro_beam[self.__ax_x, self.__ax_y]
                    self.__data[name].append(get(val))

                if name == 'Sf':
                    val = calculate_energy_fluxes(self.__grid_step_size,
                                                  plasma_particles,
                                                  plasma_fields)
                    self.__data[name].append(get(val))

        # We use dump here to save data not only at the end of the simulation
        # window, but with some period too.
        # TODO: Do we really need this? Does it work right?
        if absremainder(xi_plasma_layer,
                        self.__saving_xi_period) <= self.__xi_step_size / 2:
            self.dump(
                current_time, None, plasma_particles, None, None, None, False)

    def conditions_check(self, current_time, xi_plasma_layer):
        return absremainder(current_time,
                            self.__output_period) <= self.__time_step_size / 2

    def dump(self, current_time, xi_plasma_layer, plasma_particles: Arrays,
             plasma_fields: Arrays, plasma_currents: Arrays,
             beam_drain: BeamDrain, clean_data=True):
        if self.conditions_check(current_time, inf):
            Path(f'./diagnostics{self.__unique_directory_name}').mkdir(
                 parents=True, exist_ok=True)
            if 'numbers' in self.__f_xi_type or 'both' in self.__f_xi_type:
                plasma_particles.xp.savez(
                    f'./diagnostics{self.__unique_directory_name}/' + 
                    f'f_xi_{current_time:08.2f}.npz',
                    **self.__data)

            if 'pictures' in self.__f_xi_type or 'both' in self.__f_xi_type:
                for name in self.__f_xi_names:                        
                    plt.plot(self.__data['xi'], self.__data[name])
                    plt.savefig(
                        f'./diagnostics{self.__unique_directory_name}/' +
                        f'{name}_f_xi_{current_time:08.2f}.jpg')
                                # vmin=-1, vmax=1)
                    plt.close()

        # We now clean old data:
        if clean_data:
            for name in self.__data:
                self.__data[name] = []


class DiagnosticsColormaps:
    # By default, colormaps are taken in (y, xi) plane.
    # TODO: Make (x, xi) plane an option.
    __allowed_colormaps = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho', 'rho_beam',
                           'Phi', 'y_offt', 'y_plasma_particles']
                        # 'ni']
                    # TODO: add them and functionality!
    __allowed_colormaps_type = ['numbers']
    #TODO: add 'pictures' and 'both' and functionality

    def __init__(self, output_period=100, saving_xi_period=1000, colormaps='Ez',
                 colormaps_type='numbers', xi_from=inf, xi_to=-inf,
                 r_from=-inf, r_to=inf,
                 output_merging_r: int=1, output_merging_xi: int=1,
                 unique_directory_name=''):
        # It creates a list of string elements such as Ez, Ex, By...
        self.__colormaps_names = from_str_into_list(colormaps)
        for name in self.__colormaps_names:
            if name not in self.__allowed_colormaps:
                raise Exception(f'{name} value is not supported as a colormap.')

        # Output mode for the functions of xi:
        if colormaps_type in self.__allowed_colormaps_type:
            self.__colormaps_type = colormaps_type
        else:
            raise Exception(f"{colormaps_type} type of colormap diagnostics" +
                             "is not supported.")

        # Set time periodicity of detailed output:
        self.__output_period = output_period
        self.__saving_xi_period = saving_xi_period

        # Set borders of a subwindow:
        self.__xi_from, self.__xi_to = xi_from, xi_to
        self.__r_from, self.__r_to = r_from, r_to

        # Set values for merging functionality:
        self.__merging_r  = round(output_merging_r)
        self.__merging_xi = round(output_merging_xi)

        # Set unique name for data storing directory:
        self.__unique_directory_name = unique_directory_name

        # We store data as a Python dictionary of numpy arrays for colormaps
        # data. I'm not sure this is the best way to handle data storing.
        self.__data = {name: [] for name in self.__colormaps_names}
        self.__data['xi'] = []

    def __repr__(self) -> str:
        return (
            f"DiagnosticsColormaps(output_period={self.__output_period}, " +
            # f"saving_xi_period={self.__saving_xi_period}, " + 
            f"colormaps='{','.join(self.__colormaps_names)}', " +
            f"colormaps_type='{self.__colormaps_type}', " +
            f"xi_from={self.__xi_from}, xi_to={self.__xi_to}, " + 
            f"r_from={self.__r_from}, r_to={self.__r_to}, " +
            f"output_merging_r={self.__merging_r}, " +
            f"output_merging_xi={self.__merging_xi}, " +
            f"unique_directory_name={self.__unique_directory_name})")

    def pull_config(self, config: Config):
        """Pulls a config to get all required parameters."""
        self.__grid_steps = config.getint('window-width-steps')
        grid_step_size = config.getfloat('window-width-step-size')

        self.__grid = ((np.arange(self.__grid_steps) - self.__grid_steps // 2) *
                        grid_step_size)

        # Here we define subwindow borders in terms of number of steps:
        self.__r_f = self.__grid_steps // 2 + self.__r_from / grid_step_size
        self.__r_t = self.__grid_steps // 2 + self.__r_to   / grid_step_size

        # We check that r_f and r_t are in the borders of window.
        # Otherwise, we make them equal to the size of the window.
        if self.__r_t > self.__grid_steps:
            self.__r_t = self.__grid_steps
        else:
            self.__r_t = round(self.__r_t)

        if self.__r_f < 0:
            self.__r_f = 0
        else:
            self.__r_f = round(self.__r_f)

        # Here we check if the output period is less than the time step size.
        # In that case each time step is diagnosed. The first time step is
        # always diagnosed.
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size   = config.getfloat('xi-step')

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size

        if self.__saving_xi_period < self.__xi_step_size:
            self.__saving_xi_period = self.__xi_step_size

    def after_step_dxi(self, current_time, xi_plasma_layer,
                       plasma_particles: Arrays, plasma_fields: Arrays,
                       plasma_currents: Arrays, ro_beam):
        if self.conditions_check(current_time, xi_plasma_layer):
            # Firstly, it adds the current value of xi to data dictionary:
            self.__data['xi'].append(xi_plasma_layer)

            for name in self.__colormaps_names:
                if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi']:
                    val = getattr(plasma_fields, name)[
                        self.__grid_steps//2, self.__r_f:self.__r_t]
                    self.__data[name].append(get(val))

                if name == 'rho':
                    val = getattr(plasma_currents, 'ro')[
                        self.__grid_steps//2, self.__r_f:self.__r_t]
                    self.__data[name].append(get(val))

                if name == 'rho_beam':
                    val = ro_beam[self.__grid_steps//2, self.__r_f:self.__r_t]
                    self.__data[name].append(get(val))

                # NOTE: These two are very-very experimental diagnostics!
                if name == 'y_offt':
                    val = getattr(plasma_particles, 'y_offt')
                    val = val[val.shape[0] // 2, :]
                    # TODO: self.__r_f:self.__r_t won't work here, think about
                    #       other ways for trajectories diagnostics.
                    self.__data[name].append(get(val))

                if name == 'y_plasma_particles':
                    val = (getattr(plasma_particles, 'y_init') +
                           getattr(plasma_particles, 'y_offt'))
                    val = val[val.shape[0] // 2, :]
                    # TODO: self.__r_f:self.__r_t won't work here, think about
                    #       other ways for trajectories diagnostics.
                    self.__data[name].append(get(val))

                val = None

        # We use dump here to save data not only at the end of the simulation
        # window, but with some period too.
        # TODO: Do we really need this? Does it work right?
        if absremainder(xi_plasma_layer,
                        self.__saving_xi_period) <= self.__xi_step_size / 2:
            self.dump(
                current_time, None, plasma_particles, None, None, None, False)

        # We can save data and then clean the memory after
        # the end of a subwindow.
        if (xi_plasma_layer <= self.__xi_to and
            (xi_plasma_layer + self.__xi_step_size) >= self.__xi_to):
            self.dump(
                current_time, None, plasma_particles, None, None, None, False)

    def conditions_check(self, current_time, xi_plasma_layer):
        return (
            absremainder(current_time,
                         self.__output_period) <= self.__time_step_size / 2 and
            xi_plasma_layer <= self.__xi_from and
            xi_plasma_layer >= self.__xi_to)

    def dump(self, current_time, xi_plasma_layer, plasma_particles: Arrays,
             plasma_fields: Arrays, plasma_currents: Arrays,
             beam_drain: BeamDrain, clean_data=True):
        # In case of colormaps, we reshape every data list except for xi list.
        if absremainder(current_time,
                        self.__output_period) <= self.__time_step_size / 2:
            data_for_saving = (self.__data).copy()
            data_for_saving['transverse_grid'] = self.__grid

            size = len(self.__data['xi'])
            for name in self.__colormaps_names:
                data_for_saving[name] = np.reshape(
                    np.array(self.__data[name]), (size, -1))

            # Merging the data along r and xi axes:
            if self.__merging_r > 1 or self.__merging_xi > 1:
                for name in self.__colormaps_names:
                    data_for_saving[name] = conv_2d(
                        data_for_saving[name],
                        self.__merging_xi, self.__merging_r)

            # TODO: If there is a file with the same name with important data 
            #       and we want to save data there, we should just add new data,
            #       not rewrite a file.
            Path(f'./diagnostics{self.__unique_directory_name}').mkdir(
                 parents=True, exist_ok=True)
            if ('numbers' in self.__colormaps_type or
                'both' in self.__colormaps_type):
                np.savez(f'./diagnostics{self.__unique_directory_name}/' +
                         f'colormaps_{current_time:08.2f}.npz',
                         **data_for_saving)

            # Clean some memory. TODO: Better ways to do that?
            data_for_saving = 0

        # We now clean old data:
        if clean_data:
            for name in self.__data:
                self.__data[name] = []


class DiagnosticsTransverse:
    __allowed_colormaps = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho',
                           'rho_beam', 'Phi', # 'ni',
                           'px', 'py', 'pz', 'x_offt', 'y_offt',
                           'dx_chaotic', 'dy_chaotic',
                           'dx_chaotic_perp', 'dy_chaotic_perp']
    # NOTE: If you want to save x_init, y_init, q, or m, use the SaveRunState
    #       diagnostics, as these values are constant throughout the window.

    __allowed_colormaps_type = ['numbers', 'pictures', 'both']
    # NOTE: Currently, the code only supports picture plotting for values
    #       on the grid: fields and currents.

    def __init__(self, output_period=100, saving_xi_period=1000,
                 colormaps='rho', colormaps_type='pictures',
                 unique_directory_name=''):
        # It creates a list of string elements such as Ez, Ex, By...
        self.__colormaps_names = from_str_into_list(colormaps)
        for name in self.__colormaps_names:
            if name not in self.__allowed_colormaps:
                raise Exception(f'{name} value is not supported.')

        # Output mode for the functions of xi:
        if colormaps_type in self.__allowed_colormaps_type:
            self.__colormaps_type = colormaps_type
        else:
            raise Exception(f"{colormaps_type} type of colormap diagnostics" +
                             "is not supported.")

        # Set time periodicity of detailed output:
        self.__output_period = output_period
        self.__saving_xi_period = saving_xi_period

        # Set unique name for data storing directory:
        self.__unique_directory_name = unique_directory_name

    def __repr__(self) -> str:
        return (
            f"DiagnosticsTransverse(output_period={self.__output_period}, " +
            f"saving_xi_period={self.__saving_xi_period}, " +
            f"colormaps='{','.join(self.__colormaps_names)}', " +
            f"colormaps_type='{self.__colormaps_type}', " +
            f"unique_directory_name={self.__unique_directory_name})")

    def pull_config(self, config: Config):
        """Pulls a config to get all required parameters."""
        # Here we check if the output period is less than the time step size.
        # In that case each time step is diagnosed. The first time step is
        # always diagnosed.
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size   = config.getfloat('xi-step')

        grid_steps     = config.getint('window-width-steps')
        grid_step_size = config.getfloat('window-width-step-size')
        self.__grid = (np.arange(grid_steps) - grid_steps // 2) * grid_step_size

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size

        if self.__saving_xi_period < self.__xi_step_size:
            self.__saving_xi_period = self.__xi_step_size

    def draw_figures(self, xi_plasma_layer, plasma_fields,
                     plasma_currents, ro_beam):
        Path(f'./diagnostics{self.__unique_directory_name}').mkdir(
             parents=True, exist_ok=True)

        for name in self.__colormaps_names:
            fname = f'{name}_{xi_plasma_layer:+09.2f}.jpg'

            if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi']:
                plt.imsave(
                    f'./diagnostics{self.__unique_directory_name}/' + fname,
                    get(getattr(plasma_fields, name).T), origin='lower',
                    vmin=-0.1, vmax=0.1, cmap='bwr')

            if name == 'rho':
                plt.imsave(
                    f'./diagnostics{self.__unique_directory_name}/' + fname,
                    get(getattr(plasma_currents, 'ro').T), origin='lower',
                    vmin=-0.1, vmax=0.1, cmap='bwr')

            if name == 'rho_beam':
                plt.imsave(
                    f'./diagnostics{self.__unique_directory_name}/' + fname,
                    # getattr(ro_beam, 'ro_beam').T, origin='lower',
                    get(ro_beam.T), origin='lower',
                    vmin=-0.1, vmax=0.1, cmap='bwr')

    def save_to_file(self, current_time, xi_plasma_layer,
                     plasma_particles: Arrays, plasma_fields: Arrays,
                     plasma_currents: Arrays, ro_beam):
        Path(f'./diagnostics{self.__unique_directory_name}').mkdir(
             parents=True, exist_ok=True)

        data_for_saving = {'transverse_grid': self.__grid}

        for name in self.__colormaps_names:
            if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi']:
                data_for_saving[name] = getattr(plasma_fields, name)

            if name == 'rho':
                data_for_saving[name] = getattr(plasma_currents, 'ro')

            if name == 'rho_beam':
                data_for_saving[name] = ro_beam

            if name in ['px', 'py', 'pz', 'x_offt', 'y_offt',
                        'dx_chaotic', 'dy_chaotic',
                        'dx_chaotic_perp', 'dy_chaotic_perp']:
                data_for_saving[name] = getattr(plasma_particles, name)

        file_name = \
            f'transverse_{current_time:08.2f}_{xi_plasma_layer:+09.2f}.npz'
        plasma_particles.xp.savez(
            f'./diagnostics{self.__unique_directory_name}/' + file_name,
            **data_for_saving)

    def after_step_dxi(self, current_time, xi_plasma_layer,
                       plasma_particles: Arrays, plasma_fields: Arrays,
                       plasma_currents: Arrays, ro_beam):
        if self.conditions_check(current_time, xi_plasma_layer):
            # For xy diagnostics we save files to a file or plot a picture.
            if ('pictures' in self.__colormaps_type or
                'both' in self.__colormaps_type):
                self.draw_figures(xi_plasma_layer, plasma_fields,
                                  plasma_currents, ro_beam)

            if ('numbers' in self.__colormaps_type or
                'both' in self.__colormaps_type):
                self.save_to_file(current_time, xi_plasma_layer,
                                  plasma_particles, plasma_fields,
                                  plasma_currents, ro_beam)

    def conditions_check(self, current_time, xi_plasma_layer):
        return (
            absremainder(current_time,
                         self.__output_period) <= self.__time_step_size / 2 and
            absremainder(xi_plasma_layer,
                         self.__saving_xi_period) <= self.__xi_step_size / 2)

    def dump(self, current_time, xi_plasma_layer, plasma_particles: Arrays,
             plasma_fields: Arrays, plasma_currents: Arrays,
             beam_drain: BeamDrain, clean_data=True):
        pass


class SaveRunState:
    def __init__(self, output_period=1000, saving_xi_period=inf,
                 save_beam=False, save_plasma=False, unique_directory_name=''):
        self.__output_period = output_period
        self.__saving_xi_period = saving_xi_period
        self.__save_beam = bool(save_beam)
        self.__save_plasma = bool(save_plasma)

        # Set unique name for data storing directory:
        self.__unique_directory_name = unique_directory_name

    def __repr__(self) -> str:
        return(
            f"SaveRunState(output_period={self.__output_period}, " +
            f"saving_xi_period={self.__saving_xi_period}, " +
            f"save_beam={self.__save_beam}, save_plasma={self.__save_plasma}, " +
            f"unique_directory_name={self.__unique_directory_name})")

    def pull_config(self, config: Config):
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size   = config.getfloat('xi-step')

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size

        if self.__saving_xi_period < self.__xi_step_size:
            self.__saving_xi_period = self.__xi_step_size

    def after_step_dxi(self, current_time, xi_plasma_layer,
                       plasma_particles: Arrays, plasma_fields: Arrays,
                       plasma_currents: Arrays, ro_beam):
        if (self.conditions_check(current_time, xi_plasma_layer) and
            self.__save_plasma):
            Path(f'./snapshots{self.__unique_directory_name}').mkdir(
                 parents=True, exist_ok=True)

            file_name = \
                f'plasmastate_{current_time:08.2f}_{xi_plasma_layer:+09.2f}.npz'
            plasma_particles.xp.savez_compressed(
                file=f'./snapshots{self.__unique_directory_name}/' + file_name,
                xi_plasma_layer=plasma_particles.xp.array([xi_plasma_layer]),
                x_init=plasma_particles.x_init,
                y_init=plasma_particles.y_init,
                x_offt=plasma_particles.x_offt,
                y_offt=plasma_particles.y_offt,

                dx_chaotic=plasma_particles.dx_chaotic,
                dy_chaotic=plasma_particles.dy_chaotic,
                dx_chaotic_perp=plasma_particles.dx_chaotic_perp,
                dy_chaotic_perp=plasma_particles.dy_chaotic_perp,

                px=plasma_particles.px, py=plasma_particles.py,
                pz=plasma_particles.pz,
                q=plasma_particles.q, m=plasma_particles.m,

                Ex=plasma_fields.Ex, Bx=plasma_fields.Bx,
                Ey=plasma_fields.Ey, By=plasma_fields.By,
                Ez=plasma_fields.Ez, Bz=plasma_fields.Bz,

                Phi=plasma_fields.Phi, ro=plasma_currents.ro,
                jx=plasma_currents.jx, jy=plasma_currents.jy,
                jz=plasma_currents.jz)

    def conditions_check(self, current_time, xi_plasma_layer):
        return (
            absremainder(current_time,
                         self.__output_period) <= self.__time_step_size / 2 and
            absremainder(xi_plasma_layer,
                         self.__saving_xi_period) <= self.__xi_step_size / 2)

    def dump(self, current_time, xi_plasma_layer, plasma_particles: Arrays,
             plasma_fields: Arrays, plasma_currents: Arrays, 
             beam_drain: BeamDrain, clean_data=True):
        # The run is saved if the current_time differs from a multiple
        # of the saving period by less then dt/2.
        if absremainder(current_time,
                        self.__output_period) <= self.__time_step_size / 2:
            Path(f'./run_states{self.__unique_directory_name}').mkdir(
                 parents=True, exist_ok=True)

            if self.__save_beam:
                beam_drain.beam_buffer.save(
                    f'./run_states{self.__unique_directory_name}/' +
                    f'beamfile_{current_time:08.2f}')

            if self.__save_plasma:
                plasma_particles.xp.savez_compressed(
                    file=f'./run_states{self.__unique_directory_name}/' +
                    f'plasmastate_{current_time:08.2f}',
                    xi_plasma_layer=plasma_particles.xp.array([xi_plasma_layer]),
                    x_init=plasma_particles.x_init,
                    y_init=plasma_particles.y_init,
                    x_offt=plasma_particles.x_offt,
                    y_offt=plasma_particles.y_offt,
        
                    dx_chaotic=plasma_particles.dx_chaotic,
                    dy_chaotic=plasma_particles.dy_chaotic,
                    dx_chaotic_perp=plasma_particles.dx_chaotic_perp,
                    dy_chaotic_perp=plasma_particles.dy_chaotic_perp,

                    px=plasma_particles.px, py=plasma_particles.py,
                    pz=plasma_particles.pz,
                    q=plasma_particles.q, m=plasma_particles.m,

                    Ex=plasma_fields.Ex, Bx=plasma_fields.Bx,
                    Ey=plasma_fields.Ey, By=plasma_fields.By,
                    Ez=plasma_fields.Ez, Bz=plasma_fields.Bz,
                    
                    Phi=plasma_fields.Phi, ro=plasma_currents.ro,
                    jx=plasma_currents.jx, jy=plasma_currents.jy,
                    jz=plasma_currents.jz)
