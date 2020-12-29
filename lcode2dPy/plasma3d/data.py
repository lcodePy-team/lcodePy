"""Module for setting classes for Fields, Currents and Partciles data types."""
import numpy as np
from numba import float64, jitclass

_float_array = float64[:, :]

_fields_spec = [
    ('E_x', _float_array),
    ('E_y', _float_array),
    ('E_z', _float_array),
    ('B_x', _float_array),
    ('B_y', _float_array),
    ('B_z', _float_array),
]


@jitclass(spec=_fields_spec)
class Fields(object):
    def __init__(self, n_cells: int) -> None:
        self.E_x = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.E_y = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.E_z = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.B_x = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.B_y = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.B_z = np.zeros((n_cells, n_cells), dtype=np.float64)

    # Average operation is necessary on intermediate steps to have
    # better approximations of all fields
    def average(self, other):
        fields = Fields((self.E_x).shape[0])
        fields.E_x = (self.E_x + other.E_x) / 2
        fields.E_y = (self.E_y + other.E_y) / 2
        fields.E_z = (self.E_z + other.E_z) / 2
        fields.B_x = (self.B_x + other.B_x) / 2
        fields.B_y = (self.B_y + other.B_y) / 2
        fields.B_z = (self.B_z + other.B_z) / 2
        return fields


_currents_spec = [
    ('rho', _float_array),
    ('j_x', _float_array),
    ('j_y', _float_array),
    ('j_z', _float_array),
]


@jitclass(spec=_currents_spec)
class Currents(object):
    def __init__(self, rho, j_x, j_y, j_z):
        self.rho = rho
        self.j_x = j_x
        self.j_y = j_y
        self.j_z = j_z


_particles_spec = [
    ('x_init', _float_array),
    ('y_init', _float_array),
    ('x_offt', _float_array),
    ('y_offt', _float_array),
    ('p_x', _float_array),
    ('p_y', _float_array),
    ('p_z', _float_array),
    ('q', _float_array),
    ('m', _float_array),
]


@jitclass(spec=_particles_spec)
class Particles(object):
    def __init__(self, x_init, y_init, x_offt, y_offt, p_x, p_y, p_z, q, m):
        self.x_init = np.copy(x_init)
        self.y_init = np.copy(y_init)
        self.x_offt = np.copy(x_offt)
        self.y_offt = np.copy(y_offt)
        self.p_x = np.copy(p_x)
        self.p_y = np.copy(p_y)
        self.p_z = np.copy(p_z)
        self.q = np.copy(q)
        self.m = np.copy(m)

    def copy(self):
        return Particles(self.x_init, self.y_init,
                         self.x_offt, self.y_offt,
                         self.p_x, self.p_y, self.p_z,
                         self.q, self.m)


_const_arr_spec = [
    ('ro_initial', _float_array),
    ('dirichlet_matrix', _float_array),
    ('field_mixed_matrix', _float_array),
    ('neumann_matrix', _float_array)
]


@jitclass(spec=_const_arr_spec)
class Const_Arrays(object):
    def __init__(self, ro_initial, dirichlet_matrix,
                 field_mixed_matrix, neumann_matrix):
        self.ro_initial = np.copy(ro_initial)
        self.dirichlet_matrix = np.copy(dirichlet_matrix)
        self.field_mixed_matrix = np.copy(field_mixed_matrix)
        self.neumann_matrix = np.copy(neumann_matrix)