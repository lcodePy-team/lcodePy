import numpy as np
from numba import float64 
from numba.experimental import jitclass
from lcode2dPy.plasma.weights import (
    interpolate_antisymmetric,
    interpolate_symmetric,
    particles_weights,
)

_float_array = float64[:]

_fields_spec = [
    ('E_r', _float_array),
    ('E_f', _float_array),
    ('E_z', _float_array),
    ('B_f', _float_array),
    ('B_z', _float_array),
]


@jitclass(spec=_fields_spec)
class Fields(object):
    def __init__(self, n_cells: int) -> None:
        self.E_r = np.zeros(n_cells, dtype=np.float64)
        self.E_f = np.zeros(n_cells, dtype=np.float64)
        self.E_z = np.zeros(n_cells, dtype=np.float64)
        self.B_f = np.zeros(n_cells, dtype=np.float64)
        self.B_z = np.zeros(n_cells, dtype=np.float64)

    # Average operation is necessary on intermediate steps to have
    # better approximations of E_r and E_f
    def average(self, other):
        fields = Fields(len(self.E_r))
        fields.E_r = (self.E_r + other.E_r) / 2
        fields.E_f = (self.E_f + other.E_f) / 2
        fields.E_z = (self.E_z + other.E_z) / 2
        fields.B_f = (self.B_f + other.B_f) / 2
        fields.B_z = (self.B_z + other.B_z) / 2
        return fields

    # Interpolate fields to other positions
    def interpolate(self, r, r_step):
        new_fields = Fields(r.size)
        grid_particle_parameters = particles_weights(r, r_step, self.E_r.size)
        new_fields.E_r = interpolate_antisymmetric(
            self.E_r, grid_particle_parameters,
        )
        new_fields.E_f = interpolate_antisymmetric(
            self.E_f, grid_particle_parameters,
        )
        new_fields.E_z = interpolate_symmetric(
            self.E_z, grid_particle_parameters,
        )
        new_fields.B_f = interpolate_antisymmetric(
            self.B_f, grid_particle_parameters,
        )
        new_fields.B_z = interpolate_symmetric(
            self.B_z, grid_particle_parameters,
        )

        return new_fields


_currents_spec = [
    ('rho', _float_array),
    ('j_r', _float_array),
    ('j_f', _float_array),
    ('j_z', _float_array),
]


@jitclass(spec=_currents_spec)
class Currents(object):
    def __init__(self, rho, j_r, j_f, j_z):
        self.rho = rho
        self.j_r = j_r
        self.j_f = j_f
        self.j_z = j_z

    def normalize(self, cell_volume):
        rho = self.rho / cell_volume
        j_r = self.j_r / cell_volume
        j_f = self.j_f / cell_volume
        j_z = self.j_z / cell_volume
        return Currents(rho, j_r, j_f, j_z)

_particles_spec = [
    ('r', _float_array),
    ('p_r', _float_array),
    ('p_f', _float_array),
    ('p_z', _float_array),
    ('q', _float_array),
    ('m', _float_array),
    ('age', _float_array),
]


@jitclass(spec=_particles_spec)
class Particles(object):
    def __init__(self, r, p_r, p_f, p_z, q, m, age):
        self.r = np.copy(r)
        self.p_r = np.copy(p_r)
        self.p_f = np.copy(p_f)
        self.p_z = np.copy(p_z)
        self.q = np.copy(q)
        self.m = np.copy(m)
        self.age = np.copy(age)

    def copy(self):
        return Particles(self.r, self.p_r, self.p_f, self.p_z, self.q, self.m, self.age)
