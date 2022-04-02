import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from lcode2dPy.config.config import Config
from lcode2dPy.config.default_config import default_config

from lcode2dPy.plasma3d_gpu.data import GPUArraysView


def from_str_into_list(names_str: str):
    # Makes a list of elements that it gets from a long string.
    names = np.array(names_str.replace(' ', '').split(','))
    names = names[names != '']
    return names


class Diagnostics3d:
    def __init__(self, config: Config=default_config,
                 diag_list: list=None):
        """The main class of 3d diagnostics."""
        self.config = config
        if diag_list is None:
            self.diag_list = []
        else:
            self.diag_list = diag_list

        # Pushes a config to all choosen diagnostics classes:
        for diag in self.diag_list:
            try:
                diag.pull_config(self.config)
            except AttributeError:
                print(f'{diag} type of diagnostics is not supported.')

    def dxi(self, *parameters):
        for diag in self.diag_list:
            try:
                diag.dxi(*parameters)
            except AttributeError:
                print(f'{diag} type of diagnostics is not supported.')

    def dump(self, *parameters):
        for diag in self.diag_list:
            diag.dump(*parameters)

    def dt(self, *parameters):
        for diag in self.diag_list:
            try:
                diag.dt(*parameters)
            except AttributeError:
                print(f'{diag} type of diagnostics is not supported.')


class Diagnostics_f_xi:
    allowed_f_xi = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'ne', 'nb',
                    'Ez2', 'Bz2', 'nb2']
                    # 'Phi', 'ni']
                    # TODO: add them and functionality!
    allowed_f_xi_type = ['numbers', 'pictures', 'both']
    #TODO: add 'pictures' and 'both' and functionality

    def __init__(self, output_period=100, f_xi='Ez', f_xi_type='numbers',
                 axis_x=0, axis_y=0, auxiliary_x=1, auxiliary_y=1):
        # It creates a list of string elements such as Ez, Ex, By...
        self.f_xi_names = from_str_into_list(f_xi)
        for name in self.f_xi_names:
            if name not in self.allowed_f_xi:
                raise Exception(f'{name} value is not supported as f(xi).')

        # Output mode for the functions of xi:
        if f_xi_type in self.allowed_f_xi_type:
            self.f_xi_type = f_xi_type
        else:
            raise Exception(f'{f_xi_type} type of f(xi) diagnostics is not supported.')

        # The position of a `probe' line 1 from the center:
        self.ax_x, self.ax_y = axis_x, axis_y
        # THe position of a `probe' line 2 from the center:
        self.aux_x, self.aux_y = auxiliary_x, auxiliary_y

        # Set time periodicity of detailed output:
        self.period = output_period

        # We store data as a simple Python dictionary of lists for f(xi) data.
        # But I'm not sure this is the best way to handle data storing!
        self.data = {name: [] for name in self.f_xi_names}
        self.data['xi'] = []

    def pull_config(self, config=default_config):
        """Pulls a config to get all required parameters."""
        self.steps = config.getint('window-width-steps')
        self.step_size = config.getfloat('window-width-step-size')

        # We change how we store the positions of the 'probe' lines:
        self.ax_x  = self.steps // 2 + int(self.ax_x / self.step_size)
        self.ax_y  = self.steps // 2 + int(self.ax_y / self.step_size)
        self.aux_x = self.steps // 2 + int(self.aux_x / self.step_size)
        self.aux_y = self.steps // 2 + int(self.aux_y / self.step_size)

        # Here we check if the output period is less than the time step size.
        # In that case each time step is diagnosed. The first time step is
        # always diagnosed. And we check if period % time_step_size = 0,
        # because otherwise it won't diagnosed anything.
        self.time_step_size = config.getfloat('time-step')

        if self.period < self.time_step_size:
            self.period = self.time_step_size

        if self.period % self.time_step_size != 0:
            print("The diagnostics will not work because",
                  f"{self.period} % {self.time_step_size} != 0")

    def dxi(self, current_time, xi_plasma_layer,
            pl_particles, pl_fields, pl_currents, ro_beam):
        if current_time % self.period == 0:
            self.data['xi'].append(xi_plasma_layer)

            for name in self.f_xi_names:
                if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
                    field = getattr(pl_fields, name)[self.ax_x, self.ax_y]
                    self.data[name].append(field)

                if name == 'ne':
                    ro = getattr(pl_currents, 'ro')[self.ax_x, self.ax_y]
                    self.data[name].append(ro)

                if name == 'nb':
                    ro = ro_beam[self.ax_x, self.ax_y]
                    self.data[name].append(ro)

                if name in ['Ez2', 'Bz2']:
                    field = getattr(pl_fields, name[:2])[self.aux_x, self.aux_y]
                    self.data[name].append(field)

                if name == 'nb2':
                    ro = ro_beam[self.aux_x, self.aux_y]
                    self.data[name].append(ro)

    def dump(self, current_time):
        time_save = current_time + self.time_step_size

        Path('./diagnostics').mkdir(parents=True, exist_ok=True)
        if 'numbers' in self.f_xi_type or 'both' in self.f_xi_type:
            np.savez(f'./diagnostics/f_xi_{time_save:08.2f}.npz',
                     **self.data)

        if 'pictures' in self.f_xi_type or 'both' in self.f_xi_type:
            for name in self.f_xi_names:
                plt.plot(self.data['xi'], self.data[name])
                plt.savefig(f'./diagnostics/{name}_f_xi_{time_save:08.2f}.png')
                            # vmin=-1, vmax=1)
                plt.close()

    def dt(self, *params):
        # We use this function to clean old data:
        for name in self.data:
            self.data[name] = []


class Diagnostics_colormaps:
    allowed_colormaps = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'ne', 'nb',
                         'px', 'py', 'pz']
                        # 'Phi', 'ni']
                    # TODO: add them and functionality!
    allowed_colormaps_type = ['numbers']
    #TODO: add 'pictures' and 'both' and functionality

    def __init__(self, output_period=100,
                 colormaps='Ez', colormaps_type='numbers',
                 xi_from=float('inf'), xi_to=float('-inf'),
                 r_from=0, r_to=float('inf'),
                 output_merging_r: int=1, output_merging_xi: int=1):
        # It creates a list of string elements such as Ez, Ex, By...
        self.colormaps_names = from_str_into_list(colormaps)
        for name in self.colormaps_names:
            if name not in self.allowed_colormaps:
                raise Exception(f'{name} value is not supported as a colormap.')

        # Output mode for the functions of xi:
        if colormaps_type in self.allowed_colormaps_type:
            self.colormaps_type = colormaps_type
        else:
            raise Exception(f'{colormaps_type} type of colormap diagnostics is not supported.')

        # Set time periodicity of detailed output:
        self.period = output_period

        # Set borders of a subwindow:
        self.xi_from, self.xi_to = xi_from, xi_to
        self.r_from, self.r_to = r_from, r_to

        # Set values for merging functionality:
        self.merging_r  = int(output_merging_r)
        self.merging_xi = int(output_merging_xi)

        # We store data as a Python dictionary of numpy arrays for colormaps
        # data. I'm not sure this is the best way to handle data storing.
        self.data = {name: [] for name in self.colormaps_names}
        self.data['xi'] = []

    def pull_config(self, config=default_config):
        """Pulls a config to get all required parameters."""
        self.steps = config.getint('window-width-steps')
        self.step_size = config.getfloat('window-width-step-size')

        # Here we define subwindow borders in terms of number of steps:
        self.r_from = self.steps // 2 + self.r_from / self.step_size
        self.r_to   = self.steps // 2 + self.r_to   / self.step_size

        # We check that r_from and r_to are in the borders of window.
        # Otherwise, we make them equal to the size of the window.
        if self.r_to > self.steps:
            self.r_to = self.steps
        else:
            self.r_to = int(self.r_to)

        if self.r_from < 0:
            self.r_from = 0
        else:
            self.r_from = int(self.r_from)

        # Here we check if the output period is less than the time step size.
        # In that case each time step is diagnosed. The first time step is
        # always diagnosed. And we check if period % time_step_size = 0,
        # because otherwise it won't diagnosed anything.
        self.time_step_size = config.getfloat('time-step')

        if self.period < self.time_step_size:
            self.period = self.time_step_size   

        if self.period % self.time_step_size != 0:
            print("The diagnostics will not work because",
                  f"{self.period} % {self.time_step_size} != 0")

    def dxi(self, current_time, xi_plasma_layer,
            pl_particles, pl_fields, pl_currents, ro_beam):
        if (current_time % self.period == 0 and xi_plasma_layer <= self.xi_from
            and xi_plasma_layer >= self.xi_to):
            self.data['xi'].append(xi_plasma_layer)

            for name in self.colormaps_names:
                if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
                    val = getattr(pl_fields, name)[
                        self.steps//2, self.r_from:self.r_to]
                    self.data[name].append(val)

                if name == 'ne':
                    val = getattr(pl_currents, 'ro')[
                        self.steps//2, self.r_from:self.r_to]
                    self.data[name].append(val)

                if name == 'nb':
                    val = ro_beam[self.steps//2, self.r_from:self.r_to]
                    self.data[name].append(val)

                if name in ['px', 'py', 'pz']:
                    val = getattr(pl_particles, name)[
                        self.steps//2, self.r_from:self.r_to]
                    self.data[name].append(val)

    def dump(self, current_time):
        # In case of colormaps, we reshape every data list except for xi list.
        size = len(self.data['xi'])
        if size != 0:
            for name in self.colormaps_names:
                    self.data[name] = np.reshape(np.array(self.data[name]),
                                                (size, -1))

        # Merging the data along r and xi axes:
        if self.merging_r > 1 or self.merging_xi > 1:
            for name in self.colormaps_names:
                self.data[name] = conv_2d(self.data[name],
                                        self.merging_xi, self.merging_r)

        # Saving the data to a file:
        time_for_save = current_time + self.time_step_size

        Path('./diagnostics').mkdir(parents=True, exist_ok=True)
        if 'numbers' in self.colormaps_type or 'both' in self.colormaps_type:
            np.savez(f'./diagnostics/colormaps_{time_for_save:08.2f}.npz',
                     **self.data)

    def dt(self, *params):
        # We use this function to clean old data:
        for name in self.data:
            self.data[name] = []


class Save_run_state:
    def __init__(self, saving_period=1000., save_beam=False, save_plasma=False):
        self.saving_period = saving_period
        self.save_beam = bool(save_beam)
        self.save_plasma = bool(save_plasma)

    def pull_config(self, config=default_config):
        self.time_step_size = config.getfloat('time-step')

        if self.saving_period < self.time_step_size:
            self.saving_period = self.time_step_size

        # Important for saving arrays from GPU:
        self.pu_type = config.get('processing-unit-type').lower()

    def dxi(self, *parameters):
        pass

    def dump(self, *parameeters):
        pass

    def dt(self, current_time,
           pl_particles, pl_fields, pl_currents, beam_drain):
        time_for_save = current_time + self.time_step_size

        # The run is saved if the current_time differs from a multiple
        # of the saving period by less then dt/2.
        if current_time % self.saving_period <= self.time_step_size / 2:
            Path('./run_state').mkdir(parents=True, exist_ok=True)

            if self.save_beam:
                beam_drain.beam_buffer.save(
                    f'./run_state/beamfile_{time_for_save:08.2f}')

            if self.save_plasma:
                # Important for saving arrays from GPU (is it really?)
                if self.pu_type == 'gpu':
                    pl_particles = GPUArraysView(pl_particles)
                    pl_fields    = GPUArraysView(pl_fields)
                    pl_currents  = GPUArraysView(pl_currents)

                np.savez_compressed(
                    file=f'./run_state/plasmastate_{time_for_save:08.2f}',
                    x_offt=pl_particles.x_offt, y_offt=pl_particles.y_offt,
                    px=pl_particles.px, py=pl_particles.py, pz=pl_particles.pz,
                    Ex=pl_fields.Ex, Ey=pl_fields.Ey, Ez=pl_fields.Ez,
                    Bx=pl_fields.Bx, By=pl_fields.By, Bz=pl_fields.Bz,
                    ro=pl_currents.ro,
                    jx=pl_currents.jx, jy=pl_currents.jy, jz=pl_currents.jz)


def conv_2d(arr: np.ndarray, merging_xi, merging_r):
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

            new_arr.append(np.mean(arr[start_i:end_i,start_j:end_j]))

    return np.reshape(np.array(new_arr),
                      math.ceil(arr.shape[0] / merging_xi, -1))
