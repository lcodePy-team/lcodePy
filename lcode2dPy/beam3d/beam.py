import numpy as np
import numba as nb

from math import floor, sqrt


particle_dtype = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
                           ('p_x', 'f8'), ('p_y', 'f8'), ('p_z', 'f8'),
                           ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'f8')])


class BeamParticles:
    def __init__(self, size):
        self.size = size
        self.xi = np.zeros(size,     dtype=np.float64)
        self.x = np.zeros(size,      dtype=np.float64)
        self.y = np.zeros(size,      dtype=np.float64)
        self.px = np.zeros(size,     dtype=np.float64)
        self.py = np.zeros(size,     dtype=np.float64)
        self.pz = np.zeros(size,     dtype=np.float64)
        self.q_m = np.zeros(size,    dtype=np.float64)
        self.q_norm = np.zeros(size, dtype=np.float64)
        self.id = np.zeros(size,     dtype=np.int64)
        self.dt = np.zeros(size,     dtype=np.float64)
        self.remaining_steps = np.zeros(size,
                                     dtype=np.int64)

    def load(self, *args, **kwargs):
        with np.load(*args, **kwargs) as loaded:
            self.size = len(loaded['xi'])
            self.xi = loaded['xi']
            self.x = loaded['x']
            self.y = loaded['y']
            self.px = loaded['px']
            self.py = loaded['py']
            self.pz = loaded['pz']
            self.q_m = loaded['q_m']
            self.q_norm = loaded['q_norm']
            self.id = loaded['id']
            self.dt = np.zeros(self.size, dtype=np.float64)
            self.remaining_steps = np.zeros(self.size,
                                          dtype=np.int64)

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


# Deposition and interpolation helper function #

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
def weights(x, y, xi_loc, grid_steps, grid_step_size):
    """
    """
    x_h, y_h = x / grid_step_size + .5, y / grid_step_size + .5
    i, j = int(floor(x_h) + grid_steps // 2), int(floor(y_h) + grid_steps // 2)
    x_loc, y_loc = x_h - floor(x_h) - .5, y_h - floor(y_h) - .5
    # xi_loc = dxi = (xi_prev - xi) / D_XIP

    wxi0 = xi_loc
    wxiP = (1 - xi_loc)
    wx0 = x_loc
    wxP = (1 - x_loc)
    wy0 = y_loc
    wyP = (1 - y_loc)

    w000, w00P = wxi0 * wx0 * wy0, wxi0 * wx0 * wyP
    w0P0, w0PP = wxi0 * wxP * wy0, wxi0 * wxP * wyP
    wP00, wP0P = wxiP * wx0 * wy0, wxiP * wx0 * wyP
    wPP0, wPPP = wxiP * wxP * wy0, wxiP * wxP * wyP

    return i, j, w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP

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
def interp(value_0, value_1, i, j,
                w000, w00P, w0P0, w0PP,
                wP00, wP0P, wPP0, wPPP):
    """
    Collect value from a cell and surrounding cells (using `weights` output).
    """
    return (
        w000 * value_0[i + 0, j + 0] +
        w00P * value_0[i + 0, j + 1] +
        w0P0 * value_0[i + 1, j + 0] +
        w0PP * value_0[i + 1, j + 1] +
        wP00 * value_1[i + 0, j + 0] +
        wP0P * value_1[i + 0, j + 1] +
        wPP0 * value_1[i + 1, j + 0] +
        wPPP * value_1[i + 1, j + 1]
    )


@nb.njit
def particle_fields(x, y, xi, grid_steps, grid_step_size, xi_step_size, xi_k,
                    Ex_k_1, Ey_k_1, Ez_k_1, Bx_k_1, By_k_1, Bz_k_1,
                    Ex_k,   Ey_k,   Ez_k,   Bx_k,   By_k,   Bz_k):
    xi_loc = (xi_k - xi) / xi_step_size

    i, j, w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP = weights(
        x, y, xi_loc, grid_steps, grid_step_size
    )

    Ex = interp(Ex_k, Ex_k_1, i, j, 
                w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP)
    Ey = interp(Ey_k, Ey_k_1, i, j,
                w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP)
    Ez = interp(Ez_k, Ez_k_1, i, j,
                w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP)
    Bx = interp(Bx_k, Bx_k_1, i, j,
                w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP)
    By = interp(By_k, By_k_1, i, j,
                w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP)
    Bz = interp(Bz_k, Bz_k_1, i, j,
                w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP)

    return Ex, Ey, Ez, Bx, By, Bz


# Move_particles_kernel helper functions #

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


@nb.njit
def sign(x):
    return -1 if x < 0 else 1 if x > 0 else 0


# Moves one particle as far as possible on current xi layer

@nb.njit(parallel=True)
def move_particles_kernel(grid_steps, grid_step_size, xi_step_size,
                          xi_layer, max_radius,
                          q_m_, dt_, remaining_steps,
                          x, y, xi, px, py, pz, id,
                          Ex_k_1, Ey_k_1, Ez_k_1, Bx_k_1, By_k_1, Bz_k_1,
                          Ex_k,   Ey_k,   Ez_k,   Bx_k,   By_k,   Bz_k):
    """
    Moves one particle as far as possible on current xi layer.
    """
    xi_k = xi_layer * -xi_step_size  # xi_{k}
    xi_k_1 = (xi_layer + 1) * -xi_step_size  # xi_{k+1}
    
    for k in nb.prange(len(id)):
        q_m = q_m_[k]; dt = dt_[k]

        while remaining_steps[k] > 0:
            # Initial impulse and position vectors
            opx, opy, opz = px[k], py[k], pz[k]
            ox, oy, oxi = x[k], y[k], xi[k]

            # Compute approximate position of the particle in the middle of the step
            gamma_m = sqrt((1 / q_m) ** 2 + opx ** 2 + opy ** 2 + opz ** 2)

            x_halfstep  = ox  + dt / 2 * (opx / gamma_m)
            y_halfstep  = oy  + dt / 2 * (opy / gamma_m)
            xi_halfstep = oxi + dt / 2 * (opz / gamma_m - 1)
            # Add time shift correction (dxi = (v_z - c)*dt)

            if not_in_layer(xi_halfstep, xi_k_1):
                # x[k], y[k], xi[k] = x_halfstep, y_halfstep, xi_halfstep
                break

            if is_lost(x_halfstep, y_halfstep, max(0.9 * max_radius, max_radius - 1)):
                # x[k], y[k], xi[k] = x_halfstep, y_halfstep, xi_halfstep
                id[k] *= -1  # Particle hit the wall and is now lost
                remaining_steps[k] = 0
                break

            # Interpolate fields and compute new impulse
            (Ex, Ey, Ez,
            Bx, By, Bz) = particle_fields(x_halfstep, y_halfstep, xi_halfstep,
                                                    grid_steps, grid_step_size,
                                                    xi_step_size, xi_k,
                                                    Ex_k_1, Ey_k_1, Ez_k_1,
                                                    Bx_k_1, By_k_1, Bz_k_1,
                                                    Ex_k, Ey_k, Ez_k,
                                                    Bx_k, By_k, Bz_k)

            # Compute new impulse
            vx, vy, vz = opx / gamma_m, opy / gamma_m, opz / gamma_m
            px_halfstep = (opx + sign(q_m) * dt / 2 * (Ex + vy * Bz - vz * By))
            py_halfstep = (opy + sign(q_m) * dt / 2 * (Ey + vz * Bx - vx * Bz))
            pz_halfstep = (opz + sign(q_m) * dt / 2 * (Ez + vx * By - vy * Bx))

            # Compute final coordinates and impulses
            gamma_m = sqrt((1 / q_m) ** 2
                        + px_halfstep ** 2 + py_halfstep ** 2 + pz_halfstep ** 2)

            x[k]  = ox  + dt * (px_halfstep / gamma_m)      #  x fullstep
            y[k]  = oy  + dt * (py_halfstep / gamma_m)      #  y fullstep
            xi[k] = oxi + dt * (pz_halfstep / gamma_m - 1)  # xi fullstep

            px[k] = 2 * px_halfstep - opx                   # px fullstep
            py[k] = 2 * py_halfstep - opy                   # py fullstep
            pz[k] = 2 * pz_halfstep - opz                   # pz fullstep

            # TODO: Do we need to add it here?
            # if not_in_layer(xi_halfstep, xi_k_1):
            #     continue

            if is_lost(x[k], y[k], max(0.9 * max_radius, max_radius - 1)):
                id[k] *= -1  # Particle hit the wall and is now lost
                remaining_steps[k] = 0
                break

            remaining_steps[k] -= 1


def move_particles(grid_steps, grid_step_size, xi_step_size,
                   idxes, xi_layer, max_radius,
                   beam, fields_k_1, fields_k):
    """
    This is a convenience wrapper around the `move_particles_kernel` CUDA kernel.
    """
    x_new,  y_new,  xi_new = beam.x[idxes],  beam.y[idxes],  beam.xi[idxes]
    px_new, py_new, pz_new = beam.px[idxes], beam.py[idxes], beam.pz[idxes]
    id_new, remaining_steps_new = beam.id[idxes], beam.remaining_steps[idxes]

    move_particles_kernel(grid_steps, grid_step_size, xi_step_size,
                          xi_layer, max_radius,
                          beam.q_m[idxes], beam.dt[idxes],
                          remaining_steps_new,
                          x_new, y_new, xi_new,
                          px_new, py_new, pz_new,
                          id_new,
                          fields_k_1.Ex, fields_k_1.Ey, fields_k_1.Ez,
                          fields_k_1.Bx, fields_k_1.By, fields_k_1.Bz,
                          fields_k.Ex, fields_k.Ey, fields_k.Ez,
                          fields_k.Bx, fields_k.By, fields_k.Bz)

    beam.x[idxes],  beam.y[idxes],  beam.xi[idxes] = x_new,  y_new,  xi_new
    beam.px[idxes], beam.py[idxes], beam.pz[idxes] = px_new, py_new, pz_new
    beam.id[idxes], beam.remaining_steps[idxes] = id_new, remaining_steps_new


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
        arr_to_search = -self.beam.xi[self.layout_count:]
        if len(arr_to_search) != 0:
            end = np.argmax(arr_to_search > -xi_2)
        else:
            end = 0

        return self.layout_count, end + self.layout_count, xi_1, xi_2

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
        if len(idxes) != 0:
            move_particles(self.grid_steps, self.grid_step_size,
                           self.xi_step_size, idxes, self.xi_layer,
                           self.max_radius,
                           self.beam, fields_k_1, fields_k)

        self.stop_moving_layer()
        self.xi_layer += 1

    def move_next_xi_layer(self):
        xi = -self.xi_step_size * self.xi_layer
        xi_1 = -self.xi_step_size * (self.xi_layer + 1)
        
        # Does it find all particles that lay in the layer?
        # Doesn't look like this. Check it.
        arr_to_search = -self.beam.xi[self.touched_count:self.layout_count]
        if len(arr_to_search) != 0:
            end = np.argmax(arr_to_search > -xi_1)
        else:
            end = 0

        return self.touched_count, end + self.touched_count, xi, xi_1

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