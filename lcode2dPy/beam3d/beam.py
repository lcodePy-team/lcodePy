import numpy as np
import numba as nb

# from numba import int64, float64
# from numba.experimental import jitclass

# _int_array = int64[:]
# _float_array = float64[:]

# spec = [
#     ('length', int64),
#     ('size', int64),
#     ('xi', _float_array),
#     ('x', _float_array),
#     ('y', _float_array),
#     ('px', _float_array),
#     ('py', _float_array),
#     ('pz', _float_array),
#     ('q_m', _float_array),
#     ('q_norm', _float_array),

#     ('id', _int_array),
#     ('dt', _float_array),
#     ('remaining_steps', _int_array)
# ]


# @jitclass(spec=spec)
class BeamParticles:
    def __init__(self, size):
        self.length = 0
        self.size = size
        self.xi = np.zeros(size)
        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.px = np.zeros(size)
        self.py = np.zeros(size)
        self.pz = np.zeros(size)
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
            self.px = loaded['px']
            self.py = loaded['py']
            self.pz = loaded['pz']
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
                            px = self.px,
                            py = self.py,
                            pz = self.pz,
                            q_m = self.q_m,
                            q_norm = self.q_norm,
                            id = self.id)

#TODO: The BeamParticles class makes jitting harder. And we don't really need
#      this class. Get rid of the class.

# def save(self, *args, **kwargs):
#     particle_type = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
#                               ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
#                               ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])
#     combined = np.zeros_like(self.xi, dtype=particle_type)
#     combined['xi'] = self.xi
#     combined['x'] = self.x
#     combined['y'] = self.y
#     combined['px'] = self.px
#     combined['py'] = self.py
#     combined['pz'] = self.pz
#     combined['q_m'] = self.q_m
#     combined['q_norm'] = self.q_norm
#     combined['id'] = self.id
#     return combined.tofile(*args, **kwargs)

# def load(self, *args, **kwargs):
#     particle_type = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
#                               ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
#                               ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])
#     loaded = np.fromfile(*args, dtype=particle_type, **kwargs)
#     self.length = self.size = len(loaded)
#     self.xi = loaded['xi']
#     self.x = loaded['x']
#     self.y = loaded['y']
#     self.px = loaded['px']
#     self.py = loaded['py']
#     self.pz = loaded['pz']
#     self.q_m = loaded['q_m']
#     self.q_norm = loaded['q_norm']
#     self.id = loaded['id']
#     self.dt = np.zeros(self.size)
#     self.remaining_steps = np.zeros(self.size)


# Functions like in weights.py in lcode2dPy

@nb.njit
def add_at_numba(out, idx_x, idx_y, value):
    for i in np.arange(len(idx_x)):
        out[idx_x[i], idx_y[i]] += value[i]

# TODO: get rid of this quite simple function.

@nb.njit
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


@nb.njit
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

@nb.njit
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


@nb.njit
def interpolate_particle(value0, value1, i, j, a000, a001, a010, a011,
                                               a100, a101, a110, a111):
    return (
        a000 * value0[i + 0, j + 0] +
        a001 * value0[i + 0, j + 1] +
        a010 * value0[i + 1, j + 0] +
        a011 * value0[i + 1, j + 1] +
        a100 * value1[i + 0, j + 0] +
        a101 * value1[i + 0, j + 1] +
        a110 * value1[i + 1, j + 0] +
        a111 * value1[i + 1, j + 1]
    )


@nb.njit
def particle_fields(x, y, xi, grid_steps, grid_step_size, xi_step_size, xi_k,
                    fields_k_1, fields_k):
    dxi = (xi_k - xi) / xi_step_size

    i, j, a000, a001, a010, a011, a100, a101, a110, a111 =\
        single_particle_weights(x, y, dxi, grid_steps, grid_step_size)

    Ex = interpolate_particle(fields_k.Ex, fields_k_1.Ex, i, j, 
                               a000, a001, a010, a011, a100, a101, a110, a111)
    Ey = interpolate_particle(fields_k.Ey, fields_k_1.Ey, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
    Ez = interpolate_particle(fields_k.Ez, fields_k_1.Ez, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
    Bx = interpolate_particle(fields_k.Bx, fields_k_1.Bx, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
    By = interpolate_particle(fields_k.By, fields_k_1.By, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
    Bz = interpolate_particle(fields_k.Bz, fields_k_1.Bz, i, j,
                               a000, a001, a010, a011, a100, a101, a110, a111)
                               
    return Ex, Ey, Ez, Bx, By, Bz


# Functions like in beam_calculate.py in lcode2dPy


@nb.njit
def beam_substepping_step(q_m, pz, substepping_energy):
    dt = np.ones_like(q_m, dtype=np.float64)
    max_dt = np.sqrt(np.sqrt(1 / q_m ** 2 + pz ** 2) / substepping_energy)
    for i in range(len(q_m)):
        while dt[i] > max_dt[i]:
            dt[i] /= 2.0
    return dt


@nb.njit
def not_in_layer(xi, xi_k_1):
    return xi_k_1 > xi


@nb.njit
def is_lost(x, y, r_max):
    return x ** 2 + y ** 2 >= r_max ** 2
    # return abs(x) >= walls_width or abs(y) >= walls_width


# Moves one particle as far as possible on current xi layer

@nb.njit
def try_move_particle(beam_q_m, beam_dt, beam_remaining_steps,
                      beam_x, beam_y, beam_xi, beam_px, beam_py, beam_pz,
                      beam_id,
                      idx, fields_k_1, fields_k, max_radius, grid_steps,
                      r_step, xi_step_size, xi_layer):
    xi_k = xi_layer * -xi_step_size  # xi_{k}
    xi_k_1 = (xi_layer + 1) * -xi_step_size  # xi_{k+1}
    q_m: float = beam_q_m[idx]
    dt: float = beam_dt[idx]
    while beam_remaining_steps[idx] > 0:
        # Initial impulse and position vectors
        opx, opy, opz = beam_px[idx], beam_py[idx], beam_pz[idx]
        ox, oy, oxi = beam_x[idx], beam_y[idx], beam_xi[idx]

        # Compute approximate position of the particle in the middle of the step
        gamma_m: float = np.sqrt((1 / q_m) ** 2 + opx ** 2 + opy ** 2 + opz ** 2)

        x_halfstep  = ox  + dt / 2 * (opx / gamma_m)
        y_halfstep  = oy  + dt / 2 * (opy / gamma_m)
        xi_halfstep = oxi + dt / 2 * (opz / gamma_m - 1)
        # Add time shift correction (dxi = (v_z - c)*dt)

        if not_in_layer(xi_halfstep, xi_k_1):
            return

        if is_lost(x_halfstep, y_halfstep, max(0.9 * max_radius, max_radius - 1)):
            beam_id[idx] *= -1  # Particle hit the wall and is now lost
            beam_x[idx]  = x_halfstep
            beam_y[idx]  = y_halfstep
            beam_xi[idx] = xi_halfstep
            beam_remaining_steps[idx] = 0
            return

        # Interpolate fields and compute new impulse
        Ex, Ey, Ez, Bx, By, Bz = particle_fields(x_halfstep, y_halfstep, xi_halfstep,
                                       grid_steps, r_step,
                                       xi_step_size, xi_k,
                                       fields_k_1, fields_k)
        
        vx, vy, vz = opx / gamma_m, opy / gamma_m, opz / gamma_m
        px_halfstep = (opx + np.sign(q_m) * dt / 2 * (Ex + vy * Bz - vz * By))
        py_halfstep = (opy + np.sign(q_m) * dt / 2 * (Ey + vz * Bx - vx * Bz))
        pz_halfstep = (opz + np.sign(q_m) * dt / 2 * (Ez + vx * By - vy * Bx))

        # Compute final coordinates and impulses
        gamma_m: float = np.sqrt((1 / q_m) ** 2
                                 + px_halfstep ** 2 + py_halfstep ** 2 + pz_halfstep ** 2)

        x_fullstep  = ox  + dt * (px_halfstep / gamma_m)
        y_fullstep  = oy  + dt * (py_halfstep / gamma_m)
        xi_fullstep = oxi + dt * (pz_halfstep / gamma_m - 1)

        px_fullstep = 2 * px_halfstep - opx
        py_fullstep = 2 * py_halfstep - opy
        pz_fullstep = 2 * pz_halfstep - opz

        beam_x[idx]  = x_fullstep
        beam_y[idx]  = y_fullstep
        beam_xi[idx] = xi_fullstep
        beam_px[idx] = px_fullstep
        beam_py[idx] = py_fullstep
        beam_pz[idx] = pz_fullstep
  
        # TODO: Do we need to add it here?
        if not_in_layer(xi_halfstep, xi_k_1):
            return

        if is_lost(x_fullstep, y_fullstep, max(0.9 * max_radius, max_radius - 1)):
            beam_id[idx] *= -1  # Particle hit the wall and is now lost
            beam_remaining_steps[idx] = 0
            return

        beam_remaining_steps[idx] -= 1


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
        # Does it find all particles that lay in the layer?
        # Doesn't look like this. Check it.
        for i in np.arange(self.layout_count, len(self.beam.xi)):
            if self.beam.xi[i] - xi_2 < 0:
                return self.layout_count, i, xi_1, xi_2
        return self.layout_count, len(self.beam.xi), xi_1, xi_2

    def start_time_step(self):
        self.rho_layout = np.zeros((self.grid_steps, self.grid_steps),
                                    dtype=np.float64)
        self.stable_count = 0
        self.touched_count = 0
        self.layout_count = 0
        self.xi_layer = -1 #Why?

    def stop_time_step(self):
        sort_idxes = np.argsort(-self.beam.xi)
        self.beam.xi = self.beam.xi[sort_idxes]
        self.beam.x = self.beam.x[sort_idxes]
        self.beam.y = self.beam.y[sort_idxes]
        self.beam.px = self.beam.px[sort_idxes]
        self.beam.py = self.beam.py[sort_idxes]
        self.beam.pz = self.beam.pz[sort_idxes]
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
            try_move_particle(self.beam.q_m, self.beam.dt,
                              self.beam.remaining_steps,
                              self.beam.x, self.beam.y, self.beam.xi,
                              self.beam.px, self.beam.py, self.beam.pz,
                              self.beam.id,
                              i, fields_k_1, fields_k,
                              self.max_radius, self.grid_steps,
                              self.grid_step_size, self.xi_step_size,
                              self.xi_layer)
        self.stop_moving_layer()
        self.xi_layer += 1

    def move_next_xi_layer(self):
        xi = -self.xi_step_size * self.xi_layer
        xi_1 = -self.xi_step_size * (self.xi_layer + 1)
        for i in np.arange(self.touched_count, self.layout_count):
            if self.beam.xi[i] - xi_1 < 0:
                return self.touched_count, i, xi, xi_1
        return self.touched_count, self.layout_count, xi, xi_1

    def start_moving_layer(self):
        start, end, xi_k, xi_k_1 = self.move_next_xi_layer()
        idxes: np.ndarray = np.arange(start, end)[self.beam.id[start:end] > 0]
        self.touched_count += end - start
        if len(idxes) == 0:
            return
        dt = beam_substepping_step(self.beam.q_m[idxes], self.beam.pz[idxes],
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
        self.beam.px[idxes] = self.beam.px[sort_idxes]
        self.beam.py[idxes] = self.beam.py[sort_idxes]
        self.beam.pz[idxes] = self.beam.pz[sort_idxes]
        self.beam.q_m[idxes] = self.beam.q_m[sort_idxes]
        self.beam.q_norm[idxes] = self.beam.q_norm[sort_idxes]
        self.beam.id[idxes] = self.beam.id[sort_idxes]
        self.beam.dt[idxes] = self.beam.dt[sort_idxes]
        self.beam.remaining_steps[idxes] = self.beam.remaining_steps[sort_idxes]
        self.stable_count += np.sum(self.beam.remaining_steps[idxes] == 0)