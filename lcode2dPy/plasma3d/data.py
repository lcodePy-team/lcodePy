"""Module for setting classes for Fields, Currents and Partciles data types."""
import numpy as np
from numba import float64
from numba.experimental import jitclass

_float_array = float64[:, :]

_fields_spec = [
    ('Ex', _float_array),
    ('Ey', _float_array),
    ('Ez', _float_array),
    ('Bx', _float_array),
    ('By', _float_array),
    ('Bz', _float_array),
]


@jitclass(spec=_fields_spec)
class Fields(object):
    def __init__(self, n_cells: int) -> None:
        self.Ex = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.Ey = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.Ez = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.Bx = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.By = np.zeros((n_cells, n_cells), dtype=np.float64)
        self.Bz = np.zeros((n_cells, n_cells), dtype=np.float64)

    # Average operation is necessary on intermediate steps to have
    # better approximations of all fields
    def average(self, other):
        fields = Fields((self.Ex).shape[0])
        fields.Ex = (self.Ex + other.Ex) / 2
        fields.Ey = (self.Ey + other.Ey) / 2
        fields.Ez = (self.Ez + other.Ez) / 2
        fields.Bx = (self.Bx + other.Bx) / 2
        fields.By = (self.By + other.By) / 2
        fields.Bz = (self.Bz + other.Bz) / 2
        return fields

    def copy(self):
        new_fields = Fields((self.Ex).shape[0])
        new_fields.Ex = np.copy(self.Ex)
        new_fields.Ey = np.copy(self.Ey)
        new_fields.Ez = np.copy(self.Ez)
        new_fields.Bx = np.copy(self.Bx)
        new_fields.By = np.copy(self.By)
        new_fields.Bz = np.copy(self.Bz)
        return new_fields


_currents_spec = [
    ('ro', _float_array),
    ('jx', _float_array),
    ('jy', _float_array),
    ('jz', _float_array),
]


@jitclass(spec=_currents_spec)
class Currents(object):
    def __init__(self, ro, jx, jy, jz):
        self.ro = ro
        self.jx = jx
        self.jy = jy
        self.jz = jz


_particles_spec = [
    ('x_init', _float_array),
    ('y_init', _float_array),
    ('x_offt', _float_array),
    ('y_offt', _float_array),
    ('px', _float_array),
    ('py', _float_array),
    ('pz', _float_array),
    ('q', _float_array),
    ('m', _float_array),
]


@jitclass(spec=_particles_spec)
class Particles(object):
    def __init__(self, x_init, y_init, x_offt, y_offt, px, py, pz, q, m):
        self.x_init = np.copy(x_init)
        self.y_init = np.copy(y_init)
        self.x_offt = np.copy(x_offt)
        self.y_offt = np.copy(y_offt)
        self.px = np.copy(px)
        self.py = np.copy(py)
        self.pz = np.copy(pz)
        self.q = np.copy(q)
        self.m = np.copy(m)

    def copy(self):
        return Particles(self.x_init, self.y_init,
                         self.x_offt, self.y_offt,
                         self.px, self.py, self.pz,
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