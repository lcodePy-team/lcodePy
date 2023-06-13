from math import inf, ceil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ..config.config import Config
from ..plasma3d.data import ArraysView


# Auxiliary functions:

def from_str_into_list(names_str: str):
    """ Makes a list of elements that it gets from a long string."""
    # For example: 'Ez, Ey, Ez ,Bx,,' becomes ['Ez', 'Ey', 'Ez', 'Bx']
    names = np.array(names_str.replace(' ', '').split(','))
    names = names[names != '']
    return names


def conv_2d(arr: np.ndarray, merging_xi: int, merging_r: int):
    """Calculates strided convolution using a mean/uniform kernel."""
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


# Diagnostics classes:
# TODO: Add a template class for diagnostics.

class DiagnosticsFXi:
    __allowed_f_xi = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho', 'rho_beam',
                      'Ez2', 'Bz2', 'rho_beam2', 'Phi']
                    # 'ni']
                    # TODO: add them and functionality!
    __allowed_f_xi_type = ['numbers', 'pictures', 'both']
    #TODO: add 'pictures' and 'both' and functionality

    def __init__(self, output_period=100, saving_xi_period=100, f_xi='Ez',
                 f_xi_type='numbers', axis_x=0, axis_y=0,
                 auxiliary_x=0, auxiliary_y=1):
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

        # The position of a `probe' line 1 from the center:
        self.__axis_x, self.__axis_y = axis_x, axis_y
        # THe position of a `probe' line 2 from the center:
        self.__auxiliary_x, self.__auxiliary_y = auxiliary_x, auxiliary_y

        # Set time periodicity of detailed output and safving into a file:
        self.__output_period = output_period
        self.__saving_xi_period = saving_xi_period

        # We store data as a simple Python dictionary of lists for f(xi) data.
        # But I'm not sure this is the best way to handle data storing!
        self.__data = {name: [] for name in self.__f_xi_names}
        self.__data['xi'] = []

    def __repr__(self):
        return (f"DiagnosticsFXi(output_period={self.__output_period}, " +
                f"saving_xi_period={self.__saving_xi_period}, " +
                f"saving_xi_period={self.__saving_xi_period}, " +
                f"saving_xi_period={self.__saving_xi_period}, " +
                f"f_xi='{','.join(self.__f_xi_names)}', " +
                f"f_xi_type='{self.__f_xi_type}', " +
                f"axis_x={self.__axis_x}, axis_y={self.__axis_y}, " +
                f"auxiliary_x={self.__auxiliary_x}, " +
                f"auxiliary_y={self.__auxiliary_y})")

    def pull_config(self, config: Config):
        """Pulls a config to get all required parameters."""
        steps = config.getint('window-width-steps')
        grid_step_size = config.getfloat('window-width-step-size')

        # We change how we store the positions of the 'probe' lines:
        self.__ax_x  = steps // 2 + round(self.__axis_x / grid_step_size)
        self.__ax_y  = steps // 2 + round(self.__axis_y / grid_step_size)
        self.__aux_x = steps // 2 + round(self.__auxiliary_x / grid_step_size)
        self.__aux_y = steps // 2 + round(self.__auxiliary_y / grid_step_size)

        # Here we check if the output period is less than the time step size.
        # In that case each time step is diagnosed. The first time step is
        # always diagnosed. And we check if output_period % time_step_size = 0,
        # because otherwise it won't diagnosed anything.
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size   = config.getfloat('xi-step')

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size

        if self.__saving_xi_period < self.__xi_step_size:
            self.__saving_xi_period = self.__xi_step_size

        if self.__output_period % self.__time_step_size != 0:
            raise Exception(
                "The diagnostics will not work because " +
                f"{self.__output_period} % {self.__time_step_size} != 0")

    def after_step_dxi(self, current_time, xi_plasma_layer, plasma_particles,
                       plasma_fields, plasma_currents, ro_beam):
        if self.conditions_check(current_time, xi_plasma_layer):
            self.__data['xi'].append(xi_plasma_layer)

            for name in self.__f_xi_names:
                if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi']:
                    val = getattr(plasma_fields, name)[self.__ax_x, self.__ax_y]
                    self.__data[name].append(val)

                if name == 'rho':
                    # TODO: It's just a crutch!!!
                    val = getattr(plasma_currents, 'ro')[
                        self.__ax_x, self.__ax_y]
                    self.__data[name].append(val)

                if name == 'rho_beam':
                    val = ro_beam[self.__ax_x, self.__ax_y]
                    self.__data[name].append(val)

                if name in ['Ez2', 'Bz2']:
                    val = getattr(
                        plasma_fields, name[:2])[self.__aux_x, self.__aux_y]
                    self.__data[name].append(val)

                if name == 'rho_beam2':
                    val = ro_beam[self.__aux_x, self.__aux_y]
                    self.__data[name].append(val)

        # We use dump here to save data not only at the end of the simulation
        # window, but with some period too.
        # TODO: Do we really need this? Does it work right?
        # if xi_plasma_layer % self.__saving_xi_period <= self.__xi_step_size / 2:
        #     self.dump(current_time)

    def conditions_check(self, current_time, xi_plasma_layer):
        return current_time % self.__output_period == 0

    def dump(self, current_time, plasma_particles, plasma_fields,
             plasma_currents, beam_drain):
        if self.conditions_check(current_time, inf):
            Path('./diagnostics').mkdir(parents=True, exist_ok=True)
            if 'numbers' in self.__f_xi_type or 'both' in self.__f_xi_type:
                np.savez(f'./diagnostics/f_xi_{current_time:08.2f}.npz',
                            **self.__data)

            if 'pictures' in self.__f_xi_type or 'both' in self.__f_xi_type:
                for name in self.__f_xi_names:
                    plt.plot(self.__data['xi'], self.__data[name])
                    plt.savefig(
                        f'./diagnostics/{name}_f_xi_{current_time:08.2f}.jpg')
                                # vmin=-1, vmax=1)
                    plt.close()

        # We now clean old data:
        for name in self.__data:
            self.__data[name] = []


class DiagnosticsColormaps:
    # By default, colormaps are taken in (y, xi) plane.
    # TODO: Make (x, xi) plane an option.
    __allowed_colormaps = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho', 'rho_beam',
                           'Phi']
                        # 'ni']
                    # TODO: add them and functionality!
    __allowed_colormaps_type = ['numbers']
    #TODO: add 'pictures' and 'both' and functionality

    def __init__(self, output_period=100, saving_xi_period=1000, colormaps='Ez',
                 colormaps_type='numbers', xi_from=inf, xi_to=-inf,
                 r_from=-inf, r_to=inf,
                 output_merging_r: int=1, output_merging_xi: int=1):
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
            f"output_merging_xi={self.__merging_xi})")

    def pull_config(self, config: Config):
        """Pulls a config to get all required parameters."""
        self.__grid_steps = config.getint('window-width-steps')
        grid_step_size = config.getfloat('window-width-step-size')

        grid = ((np.arange(self.__grid_steps) - self.__grid_steps // 2) *
                grid_step_size)
        self.__data['transverse_grid'] = grid

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
        # always diagnosed. And we check if period % time_step_size = 0,
        # because otherwise it won't diagnosed anything.
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size   = config.getfloat('xi-step')

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size

        if self.__output_period % self.__time_step_size != 0:
            raise Exception(
                "The diagnostics will not work because " +
                f"{self.__output_period} % {self.__time_step_size} != 0")

    def after_step_dxi(self, current_time, xi_plasma_layer, plasma_particles,
                       plasma_fields, plasma_currents, ro_beam):
        if self.conditions_check(current_time, xi_plasma_layer):
            # Firstly, it adds the current value of xi to data dictionary:
            self.__data['xi'].append(xi_plasma_layer)

            for name in self.__colormaps_names:
                if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi']:
                    val = getattr(plasma_fields, name)[
                        self.__grid_steps//2, self.__r_f:self.__r_t]
                    self.__data[name].append(val)

                if name == 'rho':
                    # val = getattr(plasma_currents, 'ro')[ # It isn't right!
                    val = getattr(plasma_currents, 'ro')[
                        self.__grid_steps//2, self.__r_f:self.__r_t]
                    self.__data[name].append(val)

                if name == 'rho_beam':
                    val = ro_beam[self.__grid_steps//2, self.__r_f:self.__r_t]
                    self.__data[name].append(val)

                val = None

        # We use dump here to save data not only at the end of the simulation
        # window, but with some period too.
        # TODO: Do we really need this? Does it work right?
        # if xi_plasma_layer % self.__saving_xi_period <= self.__xi_step_size / 2:
        #     self.dump(current_time, None, None, None, None)

        # We can save data and then clean the memory after
        # the end of a subwindow.
        if (xi_plasma_layer <= self.__xi_to and
            (xi_plasma_layer + self.__xi_step_size) >= self.__xi_to):
            self.dump(current_time, None, None, None, None)

            for name in self.__data:
                self.__data[name] = []


    def conditions_check(self, current_time, xi_plasma_layer):
        return (current_time % self.__output_period == 0 and
                xi_plasma_layer <= self.__xi_from and
                xi_plasma_layer >= self.__xi_to)

    def dump(self, current_time, plasma_particles, plasma_fields,
             plasma_currents, beam_drain):
        # In case of colormaps, we reshape every data list except for xi list.
        if self.conditions_check(current_time, inf):
            data_for_saving = (self.__data).copy()

            size = len(self.__data['xi'])
            for name in self.__colormaps_names:
                data_for_saving[name] = np.reshape(
                    np.array(self.__data[name]), (size, -1))

            # Merging the data along r and xi axes:
            if self.__merging_r > 1 or self.__merging_xi > 1:
                for name in self.__colormaps_names:
                    data_for_saving[name] = conv_2d(
                        data_for_saving[name], self.__merging_xi,
                        self.__merging_r)

            # TODO: If there is a file with the same name with important data 
            #       and we want to save data there, we should just add new data,
            #       not rewrite a file.
            Path('./diagnostics').mkdir(parents=True, exist_ok=True)
            if ('numbers' in self.__colormaps_type or
                'both' in self.__colormaps_type):
                np.savez(f'./diagnostics/colormaps_{current_time:08.2f}.npz',
                    **data_for_saving)

            # Clean some memory. TODO: Better ways to do that?
            data_for_saving = 0

        # We now clean old data:
        for name in self.__data:
            self.__data[name] = []


class DiagnosticsTransverse:
    __allowed_colormaps = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho',
                           'rho_beam', 'Phi', # 'ni',
                           'px', 'py', 'pz', 'x_offt', 'y_offt']
    # NOTE: If you want to save x_init, y_init, q, or m, use the SaveRunState
    #       diagnostics, as these values are constant throughout the window.

    __allowed_colormaps_type = ['numbers', 'pictures', 'both']
    # NOTE: Currently, the code only supports picture plotting for values
    #       on the grid: fields and currents.

    def __init__(self, output_period=100, saving_xi_period=1000,
                 colormaps='rho', colormaps_type='pictures'):
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

    def __repr__(self) -> str:
        return (
            f"DiagnosticsTransverse(output_period={self.__output_period}, " +
            f"saving_xi_period={self.__saving_xi_period}, " +
            f"colormaps='{','.join(self.__colormaps_names)}', " +
            f"colormaps_type='{self.__colormaps_type}')")

    def pull_config(self, config: Config):
        """Pulls a config to get all required parameters."""
        # Here we check if the output period is less than the time step size.
        # In that case each time step is diagnosed. The first time step is
        # always diagnosed. And we check if period % time_step_size = 0,
        # because otherwise it won't diagnosed anything.
        self.__time_step_size = config.getfloat('time-step')
        self.__xi_step_size   = config.getfloat('xi-step')

        grid_steps     = config.getint('window-width-steps')
        grid_step_size = config.getfloat('window-width-step-size')
        self.__grid = (np.arange(grid_steps) - grid_steps // 2) * grid_step_size

        if self.__output_period < self.__time_step_size:
            self.__output_period = self.__time_step_size

        if self.__saving_xi_period < self.__xi_step_size:
            self.__saving_xi_period = self.__xi_step_size

        if self.__output_period % self.__time_step_size != 0:
            raise Exception(
                "The diagnostics will not work because " +
                f"{self.__output_period} % {self.__time_step_size} != 0")

    def draw_figures(self, xi_plasma_layer, plasma_fields,
                     plasma_currents, ro_beam):
        Path('./diagnostics').mkdir(parents=True, exist_ok=True)

        for name in self.__colormaps_names:
            fname = f'{name}_{xi_plasma_layer:+09.2f}.jpg'

            if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi']:
                plt.imsave(
                    './diagnostics/' + fname,
                    getattr(plasma_fields, name).T, origin='lower',
                    vmin=-0.1, vmax=0.1, cmap='bwr')

            if name == 'rho':
                plt.imsave(
                    './diagnostics/' + fname,
                    getattr(plasma_currents, 'ro').T, origin='lower',
                    vmin=-0.1, vmax=0.1, cmap='bwr')

            if name == 'rho_beam':
                plt.imsave(
                    './diagnostics/' + fname,
                    getattr(ro_beam, 'ro_beam').T, origin='lower',
                    vmin=-0.1, vmax=0.1, cmap='bwr')

    def save_to_file(self, current_time, xi_plasma_layer, plasma_particles,
                     plasma_fields, plasma_currents, ro_beam):
        Path('./diagnostics').mkdir(parents=True, exist_ok=True)

        data_for_saving = {'transverse_grid': self.__grid}

        for name in self.__colormaps_names:
            if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi']:
                data_for_saving[name] = getattr(plasma_fields, name)

            if name == 'rho':
                data_for_saving[name] = getattr(plasma_currents, 'ro')

            if name == 'rho_beam':
                data_for_saving[name] = ro_beam

            if name in ['px', 'py', 'pz', 'x_offt', 'y_offt']:
                data_for_saving[name] = getattr(plasma_particles, name)

        file_name = \
            f'transverse_{current_time:08.2f}_{xi_plasma_layer:+09.2f}.npz'
        np.savez('./diagnostics/' + file_name, **data_for_saving)

    def after_step_dxi(self, current_time, xi_plasma_layer, plasma_particles,
                       plasma_fields, plasma_currents, ro_beam):
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
            current_time % self.__output_period == 0 and
            xi_plasma_layer % self.__saving_xi_period <= self.__xi_step_size / 2)

    def dump(self, current_time, plasma_particles, plasma_fields,
             plasma_currents, beam_drain):
        pass


class SaveRunState:
    def __init__(self, saving_period=1000., save_beam=False, save_plasma=False):
        self.__saving_period = saving_period
        self.__save_beam = bool(save_beam)
        self.__save_plasma = bool(save_plasma)

    def __repr__(self) -> str:
        return(f"SaveRunState(saving_period={self.__saving_period}, " +
            f"save_beam={self.__save_beam}, save_plasma={self.__save_plasma})")

    def pull_config(self, config: Config):
        self.__time_step_size = config.getfloat('time-step')

        if self.__saving_period < self.__time_step_size:
            self.__saving_period = self.__time_step_size

    def after_step_dxi(self, *parameters):
        pass

    def dump(self, current_time, plasma_particles, plasma_fields,
             plasma_currents, beam_drain):
        # The run is saved if the current_time differs from a multiple
        # of the saving period by less then dt/2.
        if current_time % self.__saving_period <= self.__time_step_size / 2:
            Path('./run_states').mkdir(parents=True, exist_ok=True)

            if self.__save_beam:
                beam_drain.beam_buffer.save(
                    f'./run_states/beamfile_{current_time:08.2f}')

            if self.__save_plasma:
                # Important for saving arrays from GPU (is it really?)
                plasma_particles = ArraysView(plasma_particles)
                plasma_fields    = ArraysView(plasma_fields)
                plasma_currents  = ArraysView(plasma_currents)

                np.savez_compressed(
                    file=f'./run_states/plasmastate_{current_time:08.2f}',
                    x_init=plasma_particles.x_init,
                    y_init=plasma_particles.y_init,
                    x_offt=plasma_particles.x_offt,
                    y_offt=plasma_particles.y_offt,
                    px=plasma_particles.px, py=plasma_particles.py,
                    pz=plasma_particles.pz,
                    q=plasma_particles.q, m=plasma_particles.m,
                    Ex=plasma_fields.Ex, Bx=plasma_fields.Bx,
                    Ey=plasma_fields.Ey, By=plasma_fields.By,
                    Ez=plasma_fields.Ez, Bz=plasma_fields.Bz,
                    Phi=plasma_fields.Phi, ro=plasma_currents.ro,
                    jx=plasma_currents.jx, jy=plasma_currents.jy,
                    jz=plasma_currents.jz)
