import numpy as np
import cupy as cp

import matplotlib.pyplot as plt
from pathlib import Path

from lcode2dPy.config.config import Config
from lcode2dPy.config.default_config import default_config


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
    
    def dump(self, current_time):
        for diag in self.diag_list:
            diag.dump(current_time)


class Diagnostics3d_f_xi:
    allowed_f_xi = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'ne', 'nb',
                    'Ez2', 'Bz2', 'nb2'] 
                    # 'Phi', 'ni']
                    # TODO: add them and functionality!
    allowed_f_xi_type = ['numbers']
    #TODO: add 'pictures' and 'both' and functionality

    def __init__(self, f_xi: str='Ez', f_xi_type='numbers', axis_x=0, axis_y=0,
                 auxiliary_x=1, auxiliary_y=1, output_period=100):
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

        self.period = output_period

        # We store data as a simple Python dictionary of lists for f(xi) data.
        # But I'm not sure this is the best way to handle data storing!
        self.data = {name: [] for name in self.f_xi_names}
        self.data['xi'] = []

    def pull_config(self, config: Config=default_config):
        """Pulls a config to get all required parameters."""
        self.steps = config.getint('window-width-steps')
        self.step_size = config.getfloat('window-width-step-size')

        # We change how we store the positions of the 'probe' lines:        
        self.ax_x = int(self.ax_x / self.step_size)
        self.ax_y = int(self.ax_y / self.step_size)
        self.aux_x = int(self.aux_x / self.step_size)
        self.aux_y = int(self.aux_y / self.step_size)

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
                pl_fields, pl_currents, ro_beam):
        if current_time % self.period == 0:
            self.data['xi'].append(xi_plasma_layer)

            for name in self.f_xi_names:
                if name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
                    field = getattr(pl_fields, name)[
                        self.steps//2 + self.ax_x, self.steps//2 + self.ax_y]
                    self.data[name].append(field)
                
                if name == 'ne':
                    ro = getattr(pl_currents, 'ro')[
                        self.steps//2 + self.ax_x, self.steps//2 + self.ax_y]
                    self.data[name].append(ro)
                
                if name == 'nb':
                    ro_beam = ro_beam[
                        self.steps//2 + self.ax_x, self.steps//2 + self.ax_y]
                    self.data[name].append(ro_beam)

                if name in ['Ez2', 'Bz2']:
                    field = getattr(pl_fields, name[:2])[
                        self.steps//2 + self.aux_x, self.steps//2 + self.aux_y]
                    self.data[name].append(field)
                
                if name == 'nb2':
                    ro_beam = ro_beam[
                        self.steps//2 + self.aux_x, self.steps//2 + self.aux_y]
                    self.data[name].append(ro_beam)

    def dump(self, current_time):
        time_for_save = current_time + self.time_step_size
        Path('./diagnostics').mkdir(parents=True, exist_ok=True)
        if 'numbers' in self.f_xi_type or 'both' in self.f_xi_type:
            np.savez(f'./diagnostics/f_xi_{time_for_save:08.2f}.npz',
                     **self.data)
