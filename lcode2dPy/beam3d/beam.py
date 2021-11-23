import numpy as np
import numba as nb

particle_dtype = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
                           ('p_x', 'f8'), ('p_y', 'f8'), ('p_z', 'f8'),
                           ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'f8')])

spec = [
    ('length', nb.int64),
    ('size', nb.int64),
    ('xi', nb.float64[:]),
    ('x', nb.float64[:]),
    ('y', nb.float64[:]),
    ('p_x', nb.float64[:]),
    ('p_y', nb.float64[:]),
    ('p_z', nb.float64[:]),
    ('q_m', nb.float64[:]),
    ('q_norm', nb.float64[:]),

    ('id', nb.int64[:]),
    ('dt', nb.float64[:]),
    ('remaining_steps', nb.int64[:])
]


# @nb.jitclass(spec=spec)
class BeamParticles:
    def __init__(self, size):
        self.length = 0
        self.size = size
        self.xi = np.zeros(size)
        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.p_x = np.zeros(size)
        self.p_y = np.zeros(size)
        self.p_z = np.zeros(size)
        self.q_m = np.zeros(size)
        self.q_norm = np.zeros(size)
        self.id = np.zeros(size, dtype=np.int64)
        self.dt = np.zeros(size)
        self.remaining_steps = np.zeros(size, dtype=np.int_)

    def load(self, *args, **kwargs):
        with np.load(*args, **kwargs) as loaded:
            self.length = self.size = len(loaded['xi'])
            self.xi = loaded['xi']
            self.x = loaded['x']
            self.y = loaded['y']
            self.p_x = loaded['p_x']
            self.p_y = loaded['p_y']
            self.p_z = loaded['p_z']
            self.q_m = loaded['q_m']
            self.q_norm = loaded['q_norm']
            self.id = loaded['id']
            self.dt = np.zeros(self.size)
            self.remaining_steps = np.zeros(self.size)

    def save(self, *args, **kwargs):
        np.savez_compressed(*args, **kwargs, 
                            xi = self.xi,
                            x = self.x, 
                            y = self.y,
                            p_x = self.p_x,
                            p_y = self.p_y,
                            p_z = self.p_z,
                            q_m = self.q_m,
                            q_norm = self.q_norm,
                            id = self.id)


# def save(self, *args, **kwargs):
#     particle_type = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
#                               ('p_x', 'f8'), ('p_y', 'f8'), ('p_z', 'f8'),
#                               ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])
#     combined = np.zeros_like(self.xi, dtype=particle_type)
#     combined['xi'] = self.xi
#     combined['x'] = self.x
#     combined['y'] = self.y
#     combined['p_x'] = self.p_x
#     combined['p_y'] = self.p_y
#     combined['p_z'] = self.p_z
#     combined['q_m'] = self.q_m
#     combined['q_norm'] = self.q_norm
#     combined['id'] = self.id
#     return combined.tofile(*args, **kwargs)

# def load(self, *args, **kwargs):
#     particle_type = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
#                               ('p_x', 'f8'), ('p_y', 'f8'), ('p_z', 'f8'),
#                               ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])
#     loaded = np.fromfile(*args, dtype=particle_type, **kwargs)
#     self.length = self.size = len(loaded)
#     self.xi = loaded['xi']
#     self.x = loaded['x']
#     self.y = loaded['y']
#     self.p_x = loaded['p_x']
#     self.p_y = loaded['p_y']
#     self.p_z = loaded['p_z']
#     self.q_m = loaded['q_m']
#     self.q_norm = loaded['q_norm']
#     self.id = loaded['id']
#     self.dt = np.zeros(self.size)
#     self.remaining_steps = np.zeros(self.size)


# Functions like in weights.py in lcode2dPy


@nb.njit(cache=True)
def add_at_numba(out, idx_x, idx_y, value):
    for i in np.arange(len(idx_x)):
        out[idx_x[i], idx_y[i]] += value[i]

# TODO: get rid of these two quite simple functions

@nb.njit(parallel=True)
def particles_weights(x, y, dxi, grid_steps, grid_step_size):  # dxi = (xi_prev - xi)/D_XIP
    x_h, y_h = x / grid_step_size + .5, y / grid_step_size + .5
    i = (np.floor(x_h) + grid_steps // 2).astype(np.int_)
    j = (np.floor(y_h) + grid_steps // 2).astype(np.int_)
    dx, dy = x_h - np.floor(x_h) - .5, y_h - np.floor(y_h) - .5

    a0_xi = dxi
    a1_xi = (1. - dxi)
    a0_x = dx
    a1_x = (1. - dx)
    a0_y = dy
    a1_y = (1. - dy)

    a000 = a0_xi * a0_x * a0_y
    a001 = a0_xi * a0_x * a1_y
    a010 = a0_xi * a1_x * a0_y
    a011 = a0_xi * a1_x * a1_y
    a100 = a1_xi * a0_x * a0_y
    a101 = a1_xi * a0_x * a1_y
    a110 = a1_xi * a1_x * a0_y
    a111 = a1_xi * a1_x * a1_y

    return i, j, a000, a001, a010, a011, a100, a101, a110, a111


@nb.njit(cache=True)
def single_particle_weights(x, y, dxi, grid_steps, grid_step_size):  # dxi = (xi_prev - xi)/D_XIP
    x_h, y_h = x / grid_step_size + .5, y / grid_step_size + .5
    i = int(np.floor(x_h) + grid_steps // 2)
    j = int(np.floor(y_h) + grid_steps // 2)
    dx, dy = x_h - np.floor(x_h) - .5, y_h - np.floor(y_h) - .5

    a0_xi = dxi
    a1_xi = (1 - dxi)
    a0_x = dx
    a1_x = (1 - dx)
    a0_y = dy
    a1_y = (1 - dy)

    a000 = a0_xi * a0_x * a0_y
    a001 = a0_xi * a0_x * a1_y
    a010 = a0_xi * a1_x * a0_y
    a011 = a0_xi * a1_x * a1_y
    a100 = a1_xi * a0_x * a0_y
    a101 = a1_xi * a0_x * a1_y
    a110 = a1_xi * a1_x * a0_y
    a111 = a1_xi * a1_x * a1_y

    return i, j, a000, a001, a010, a011, a100, a101, a110, a111

# TODO: we have similar functions for GPU in lcode3d code

@nb.njit(cache=True)
def deposit_particles(value, out0, out1, i, j, a000, a001, a010, a011,
                                               a100, a101, a110, a111):
    add_at_numba(out0, i + 0, j + 0, a000 * value)
    add_at_numba(out0, i + 0, j + 1, a001 * value)
    add_at_numba(out0, i + 1, j + 0, a010 * value)
    add_at_numba(out0, i + 1, j + 1, a011 * value)
    add_at_numba(out1, i + 0, j + 0, a100 * value)
    add_at_numba(out1, i + 0, j + 1, a101 * value)
    add_at_numba(out1, i + 1, j + 0, a110 * value)
    add_at_numba(out1, i + 1, j + 1, a111 * value)


@nb.njit(cache=True)
def interpolate_particle(value0, value1, i, j, a000, a001, a010, a011,
                                               a100, a101, a110, a111):
    return a000 * value0[i + 0, j + 0] \
         + a001 * value0[i + 0, j + 1] \
         + a010 * value0[i + 1, j + 0] \
         + a011 * value0[i + 1, j + 1] \
         + a100 * value1[i + 0, j + 0] \
         + a101 * value1[i + 0, j + 1] \
         + a110 * value1[i + 1, j + 0] \
         + a111 * value1[i + 1, j + 1]


@nb.njit
def particle_fields(r_vec, grid_steps, grid_step_size, xi_step_size, xi_k,
                    fields_k_1, fields_k):
    dxi = (xi_k - r_vec[2]) / xi_step_size

    i, j, a000, a001, a010, a011, a100, a101, a110, a111 =\
        single_particle_weights(r_vec[0], r_vec[1], dxi, grid_steps, grid_step_size)

    e_x = interpolate_particle(fields_k.E_x, fields_k_1.E_x, i, j, 
                               a000, a001, a010, a011, a100, a101, a110, a111)
    e_y = interpolate_particle(fields_k.E_y, fields_k_1.E_y, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
    e_z = interpolate_particle(fields_k.E_z, fields_k_1.E_z, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
    b_x = interpolate_particle(fields_k.B_x, fields_k_1.B_x, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
    b_y = interpolate_particle(fields_k.B_y, fields_k_1.B_y, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
    b_z = interpolate_particle(fields_k.B_z, fields_k_1.B_z, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
                               
    e_vec = np.array((e_x, e_y, e_z))
    b_vec = np.array((b_x, b_y, b_z))
    return e_vec, b_vec


# Functions like in beam_calculate.py in lcode2dPy


@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64)], cache=True)
def beam_substepping_step(q_m, p_z, substepping_energy):
    dt = 1.0
    max_dt = np.sqrt(np.sqrt(1 / q_m ** 2 + p_z ** 2) / substepping_energy)
    while dt > max_dt:
        dt /= 2.0
    return dt


@nb.njit
def cross_nb(a, b):
    c = np.zeros_like(a)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


@nb.njit
def in_layer(r_vec, xi_k_1):
    return xi_k_1 <= r_vec[2]


@nb.njit
def is_lost(r_vec, r_max):
    return r_vec[0] ** 2 + r_vec[1] ** 2 >= r_max ** 2


# Moves one particle as far as possible on current xi layer

@nb.njit
def try_move_particle(beam, idx, fields_k_1, fields_k, max_radius, grid_steps,
                      r_step, xi_step_size, xi_layer):
    xi_k = xi_layer * -xi_step_size  # xi_{k}
    xi_k_1 = (xi_layer + 1) * -xi_step_size  # xi_{k+1}
    q_m: float = beam.q_m[idx]
    dt: float = beam.dt[idx]
    while beam.remaining_steps[idx] > 0:
        # Initial impulse and position vectors
        p_vec_0 = np.array([beam.p_x[idx], beam.p_y[idx], beam.p_z[idx]])
        r_vec_0 = np.array([beam.x[idx], beam.y[idx], beam.xi[idx]])

        # Compute approximate position of the particle in the middle of the step
        gamma_0: float = np.sqrt((1 / q_m) ** 2 + np.sum(p_vec_0 ** 2))
        r_vec_1_2 = r_vec_0 + dt / 2 * p_vec_0 / gamma_0
        r_vec_1_2[2] -= dt / 2  # Add time shift correction (dxi = (v_z - c)*dt)

        if not in_layer(r_vec_1_2, xi_k_1):
            return
        if is_lost(r_vec_1_2, max(0.9 * max_radius, max_radius - 1)):
            beam.id[idx] *= -1  # Particle hit the wall and is now lost
            return

        # Interpolate fields and compute new impulse
        e_vec, b_vec = particle_fields(r_vec_1_2, grid_steps, r_step,
                                       xi_step_size, xi_k,
                                       fields_k_1, fields_k)
        p_vec_1_2 = (p_vec_0 + np.sign(q_m) * dt / 2 *
                    (e_vec + cross_nb(p_vec_0 / gamma_0, b_vec)))

        # Compute final coordinates and impulses
        gamma_1_2: float = np.sqrt((1 / q_m) ** 2 + np.sum(p_vec_1_2 ** 2))
        r_vec_1 = r_vec_0 + dt * p_vec_1_2 / gamma_1_2
        r_vec_1[2] -= dt  # Add time shift correction (dxi = (v_z - c)*dt)
        p_vec_1 = 2 * p_vec_1_2 - p_vec_0

        beam.x[idx] = r_vec_1[0]
        beam.y[idx] = r_vec_1[1]
        beam.xi[idx] = r_vec_1[2]
        beam.p_x[idx] = p_vec_1[0]
        beam.p_y[idx] = p_vec_1[1]
        beam.p_z[idx] = p_vec_1[2]
        beam.remaining_steps[idx] -= 1
        if is_lost(r_vec_1, max(0.9 * max_radius, max_radius - 1)):
            beam.id[idx] *= -1  # Particle hit the wall and is now lost


class BeamCalculator:
    def __init__(self, config, beam):
        self.beam = beam
        # Get used configuration
        self.xi_step_size = config.getfloat('xi-step')
        self.grid_step_size = config.getfloat('window-width-step-size')
        self.grid_steps = config.getint('window-width-steps')
        self.max_radius = self.grid_step_size * self.grid_steps / 2
        self.time_step = config.getfloat('time-step')
        self.substepping_energy = 2 #config.get("beam-substepping-energy")

        self.rho_layout = np.zeros((self.grid_steps, self.grid_steps),
                                    dtype=np.float64)
        self.xi_layer = -1  # xi_k
        self.stable_count = 0
        self.touched_count = 0
        self.layout_count = 0

    def layout_next_xi_layer(self):
        xi_1 = -self.xi_step_size * (self.xi_layer + 1)
        xi_2 = -self.xi_step_size * (self.xi_layer + 2)
        for i in np.arange(self.layout_count, len(self.beam.xi)):
            if self.beam.xi[i] - xi_2 < 0:
                return self.layout_count, i, xi_1, xi_2
        return self.layout_count, len(self.beam.xi), xi_1, xi_2

    def move_next_xi_layer(self):
        xi = -self.xi_step_size * self.xi_layer
        xi_1 = -self.xi_step_size * (self.xi_layer + 1)
        for i in np.arange(self.touched_count, self.layout_count):
            if self.beam.xi[i] - xi_1 < 0:
                return self.touched_count, i, xi, xi_1
        return self.touched_count, self.layout_count, xi, xi_1

    def start_time_step(self):
        self.rho_layout = np.zeros((self.grid_steps, self.grid_steps),
                                    dtype=np.float64)
        self.stable_count = 0
        self.touched_count = 0
        self.layout_count = 0
        self.xi_layer = -1

    def stop_time_step(self):
        sort_idxes = np.argsort(-self.beam.xi, kind='mergesort')
        self.beam.xi = self.beam.xi[sort_idxes]
        self.beam.x = self.beam.x[sort_idxes]
        self.beam.y = self.beam.y[sort_idxes]
        self.beam.p_x = self.beam.p_x[sort_idxes]
        self.beam.p_y = self.beam.p_y[sort_idxes]
        self.beam.p_z = self.beam.p_z[sort_idxes]
        self.beam.q_m = self.beam.q_m[sort_idxes]
        self.beam.q_norm = self.beam.q_norm[sort_idxes]
        self.beam.id = self.beam.id[sort_idxes]
        self.beam.dt = self.beam.dt[sort_idxes]
        self.beam.remaining_steps = self.beam.remaining_steps[sort_idxes]

    # Layout particles from [xi_{k+2}, xi_{k+1})
    # and get beam current densityfor k+1 step
    def layout_beam(self):
        start, end, xi_1, xi_2 = self.layout_next_xi_layer()
        rho_layout: np.ndarray = np.zeros_like(self.rho_layout)
        idxes: np.ndarray = np.arange(start, end)[self.beam.id[start:end] > 0]
        if len(idxes) != 0:
            x: np.ndarray = self.beam.x[idxes]
            y: np.ndarray = self.beam.y[idxes]
            dxi: np.ndarray = (xi_1 - self.beam.xi[idxes]) / self.xi_step_size
            i, j, a000, a001, a010, a011, a100, a101, a110, a111 = \
                particles_weights(x, y, dxi, self.grid_steps, self.grid_step_size)
            deposit_particles(self.beam.q_norm[idxes], self.rho_layout,
                              rho_layout, i, j,
                              a000, a001, a010, a011, a100, a101, a110, a111)
        self.rho_layout, rho_layout = rho_layout, self.rho_layout
        rho_layout /= self.grid_step_size ** 2
        self.layout_count += end - start
        return rho_layout

    def move_beam(self, fields_k_1, fields_k):
        self.start_moving_layer()
        idxes: np.ndarray = np.arange(self.stable_count, self.touched_count)[
            self.beam.id[self.stable_count:self.touched_count] > 0]
        for i in idxes:
            try_move_particle(self.beam, i, fields_k_1, fields_k,
                              self.max_radius, self.grid_steps,
                              self.grid_step_size, self.xi_step_size,
                              self.xi_layer)
        self.stop_moving_layer()
        self.xi_layer += 1

    def start_moving_layer(self):
        start, end, xi_k, xi_k_1 = self.move_next_xi_layer()
        idxes: np.ndarray = np.arange(start, end)[self.beam.id[start:end] > 0]
        self.touched_count += end - start
        if len(idxes) == 0:
            return
        dt = beam_substepping_step(self.beam.q_m[idxes], self.beam.p_z[idxes],
                                   self.substepping_energy)
        steps = (1. / dt).astype(np.int_)
        self.beam.dt[idxes] = dt * self.time_step
        self.beam.remaining_steps[idxes] = steps

    # Sorts beam to preserve order
    # Sort order: remaining_steps ascending
    def stop_moving_layer(self):
        idxes = np.arange(self.stable_count, self.touched_count)
        if len(idxes) == 0:
            return
        s = np.argsort(self.beam.remaining_steps[idxes])
        sort_idxes = idxes[s]
        self.beam.xi[idxes] = self.beam.xi[sort_idxes]
        self.beam.x[idxes] = self.beam.x[sort_idxes]
        self.beam.y[idxes] = self.beam.y[sort_idxes]
        self.beam.p_x[idxes] = self.beam.p_x[sort_idxes]
        self.beam.p_y[idxes] = self.beam.p_y[sort_idxes]
        self.beam.p_z[idxes] = self.beam.p_z[sort_idxes]
        self.beam.q_m[idxes] = self.beam.q_m[sort_idxes]
        self.beam.q_norm[idxes] = self.beam.q_norm[sort_idxes]
        self.beam.id[idxes] = self.beam.id[sort_idxes]
        self.beam.dt[idxes] = self.beam.dt[sort_idxes]
        self.beam.remaining_steps[idxes] = self.beam.remaining_steps[sort_idxes]
        self.stable_count += np.sum(self.beam.remaining_steps[idxes] == 0)