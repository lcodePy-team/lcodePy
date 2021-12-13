import numpy as np
import numba as nb

from math import floor, sqrt

# We don't really need this class. It's more convenient
# to have something like GPUArrays from plasma3d_gpu.

class BeamParticles:
    def __init__(self, size):
        """
        Create a new empty array of beam particles. Can be used both as
        a whole beam particles array and as a layer of beam particles.
        """
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
        # An additional parameter to track lost particles.
        self.lost = np.zeros(size, dtype=np.bool8)

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
            self.lost = np.zeros(self.size, dtype=np.bool8)

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

    # Essentials for beam layer calculations #

    def xi_sorted(self):
        """
        Sort beam particles along xi axis.
        """
        sort_idxes = np.argsort(-self.xi)

        self.xi = self.xi[sort_idxes]
        self.x = self.x[sort_idxes]
        self.y = self.y[sort_idxes]
        self.px = self.px[sort_idxes]
        self.py = self.py[sort_idxes]
        self.pz = self.pz[sort_idxes]
        self.q_m = self.q_m[sort_idxes]
        self.q_norm = self.q_norm[sort_idxes]
        self.id = self.id[sort_idxes]
        self.dt = self.dt[sort_idxes]
        self.remaining_steps = self.remaining_steps[sort_idxes]
        self.lost = self.lost[sort_idxes]

    def get_layer(self, begin, end):
        """
        Return a layer with indexes from 'begin' to 'end'.
        """
        # TODO: Find a better method of getting a layer!
        #       Have a look at plasma3d_gpu.data for examples.
        sublayer = BeamParticles(end - begin)

        sublayer.xi = self.xi[begin:end]
        sublayer.x = self.x[begin:end]
        sublayer.y = self.y[begin:end]
        sublayer.px = self.px[begin:end]
        sublayer.py = self.py[begin:end]
        sublayer.pz = self.pz[begin:end]
        sublayer.q_m = self.q_m[begin:end]
        sublayer.q_norm = self.q_norm[begin:end]
        sublayer.id = self.id[begin:end]
        sublayer.dt = self.dt[begin:end]
        sublayer.remaining_steps = self.remaining_steps[begin:end]
        sublayer.lost =   self.lost[begin:end]

        return sublayer
    
    def concatenate(self, other_layer):
        """
        Concatenate two beam particles layers.
        """
        # TODO: The same task as for self.get_sublayer()
        self.size += other_layer.size

        self.xi = np.concatenate(self.xi, other_layer.xi)
        self.x = np.concatenate(self.x, other_layer.x)
        self.y = np.concatenate(self.y, other_layer.y)
        self.px = np.concatenate(self.px, other_layer.px)
        self.py = np.concatenate(self.py, other_layer.py)
        self.pz = np.concatenate(self.pz, other_layer.pz)
        self.q_m = np.concatenate(self.q_m, other_layer.q_m)
        self.q_norm = np.concatenate(self.q_norm, other_layer.q_norm)
        self.id = np.concatenate(self.id, other_layer.id)
        self.dt = np.concatenate(self.dt, other_layer.dt)
        self.remaining_steps = np.concatenate(self.remaining_steps,
                                              other_layer.remaining_steps)
        self.lost = np.concatenate(self.lost, other_layer.lost)

#TODO: The BeamParticles class makes jitting harder. And we don't really need
#      this class. Get rid of it.


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

@nb.njit #(parallel=True)
def move_particles_kernel(grid_steps, grid_step_size, xi_step_size,
                          beam_xi_layer, lost_radius,
                          q_m_, dt_, remaining_steps,
                          x, y, xi, px, py, pz, id, lost,
                          Ex_k_1, Ey_k_1, Ez_k_1, Bx_k_1, By_k_1, Bz_k_1,
                          Ex_k,   Ey_k,   Ez_k,   Bx_k,   By_k,   Bz_k):
    """
    Moves one particle as far as possible on current xi layer.
    """
    xi_k = beam_xi_layer * -xi_step_size  # xi_{k}
    xi_k_1 = (beam_xi_layer + 1) * -xi_step_size  # xi_{k+1}
    
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

            if is_lost(x_halfstep, y_halfstep, lost_radius):
                # x[k], y[k], xi[k] = x_halfstep, y_halfstep, xi_halfstep
                id[k] *= -1  # Particle hit the wall and is now lost
                lost[k] = True
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

            if is_lost(x[k], y[k], lost_radius):
                id[k] *= -1  # Particle hit the wall and is now lost
                lost[k] = True
                remaining_steps[k] = 0
                break

            remaining_steps[k] -= 1


def move_particles(grid_steps, grid_step_size, xi_step_size,
                   idxes, beam_xi_layer, lost_radius,
                   beam, fields_k_1, fields_k):
    """
    This is a convenience wrapper around the `move_particles_kernel` CUDA kernel.
    """
    x_new,  y_new,  xi_new = beam.x[idxes],  beam.y[idxes],  beam.xi[idxes]
    px_new, py_new, pz_new = beam.px[idxes], beam.py[idxes], beam.pz[idxes]
    id_new, remaining_steps_new = beam.id[idxes], beam.remaining_steps[idxes]
    lost_new = beam.lost[idxes]

    move_particles_kernel(grid_steps, grid_step_size, xi_step_size,
                          beam_xi_layer, lost_radius,
                          beam.q_m[idxes], beam.dt[idxes],
                          remaining_steps_new,
                          x_new, y_new, xi_new,
                          px_new, py_new, pz_new,
                          id_new, lost_new,
                          fields_k_1.Ex, fields_k_1.Ey, fields_k_1.Ez,
                          fields_k_1.Bx, fields_k_1.By, fields_k_1.Bz,
                          fields_k.Ex, fields_k.Ey, fields_k.Ez,
                          fields_k.Bx, fields_k.By, fields_k.Bz)

    beam.x[idxes],  beam.y[idxes],  beam.xi[idxes] = x_new,  y_new,  xi_new
    beam.px[idxes], beam.py[idxes], beam.pz[idxes] = px_new, py_new, pz_new
    beam.id[idxes], beam.remaining_steps[idxes] = id_new, remaining_steps_new
    beam.lost[idxes] = lost_new


class BeamCalculator:
    def __init__(self, config):
        # Get main calculation parameters.
        self.xi_step_size = config.getfloat('xi-step')
        self.grid_step_size = config.getfloat('window-width-step-size')
        self.grid_steps = config.getint('window-width-steps')
        self.time_step = config.getfloat('time-step')
        self.substepping_energy = 2 #config.get("beam-substepping-energy")
        
        # Calculate the radius that marks that a particle is lost.
        max_radius = self.grid_step_size * self.grid_steps / 2
        self.lost_radius = max(0.9 * max_radius, max_radius - 1) # or just max_radius?

    # Helper functions for push_solver_3d.py file an inner steps:

    def start_time_step(self):
        # Get a grid for beam rho density
        self.rho_layout = np.zeros((self.grid_steps, self.grid_steps),
                                    dtype=np.float64)

    # Helper functions for depositing beam particles of a layer:

    def layout_beam_layer(self, beam_layer, plasma_layer_idx):
        idxes = np.arange(beam_layer.id.size)
        rho_layout = np.zeros((self.grid_steps, self.grid_steps),
                                    dtype=np.float64)

        if beam_layer.id.size != 0:
            xi_plasma_layer = - self.xi_step_size * plasma_layer_idx

            x, y = beam_layer.x, beam_layer.y
            dxi  = (xi_plasma_layer - beam_layer.xi) / self.xi_step_size
            i, j, w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP = \
                particles_weights(x, y, dxi, self.grid_steps, self.grid_step_size)

            deposit_particles(beam_layer.q_norm,
                              self.rho_layout, rho_layout, i, j,
                              w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP)

        self.rho_layout, rho_layout = rho_layout, self.rho_layout
        rho_layout /= self.grid_step_size ** 2

        return rho_layout

    # Helper functions for moving beam particles of a layer:

    def start_moving_layer(self, beam_layer, idxes):
        # TODO: Do we need to set dt and remaining_steps only for particles
        #       that have dt == 0?
        # mask = beam_layer.id[beam_layer.dt == 0] and idxes -> mask ???
        dt = beam_substepping_step(beam_layer.q_m[idxes], beam_layer.pz[idxes],
                                   self.substepping_energy)
        beam_layer.dt[idxes] = dt * self.time_step
        beam_layer.remaining_steps[idxes] = (1. / dt).astype(np.int_)

    # def stop_moving_layer(self, beam_layer):
    #     idxes = beam_layer.id
    #     if len(idxes) == 0:
    #         return

    #     beam_layer.xi_sorted()
        # s = np.argsort(beam_layer.remaining_steps[idxes])
        # sort_idxes = idxes[s]
        # beam_layer.xi[idxes] =     beam_layer.xi[sort_idxes]
        # beam_layer.x[idxes] =      beam_layer.x[sort_idxes]
        # beam_layer.y[idxes] =      beam_layer.y[sort_idxes]
        # beam_layer.px[idxes] =     beam_layer.px[sort_idxes]
        # beam_layer.py[idxes] =     beam_layer.py[sort_idxes]
        # beam_layer.pz[idxes] =     beam_layer.pz[sort_idxes]
        # beam_layer.q_m[idxes] =    beam_layer.q_m[sort_idxes]
        # beam_layer.q_norm[idxes] = beam_layer.q_norm[sort_idxes]
        # beam_layer.id[idxes] =     beam_layer.id[sort_idxes]
        # beam_layer.dt[idxes] =     beam_layer.dt[sort_idxes]
        # beam_layer.remaining_steps[idxes] = (
        #                            beam_layer.remaining_steps[sort_idxes])

    def move_beam_layer(self, beam_layer, plasma_layer_idx,
                        fields_after_layer, fields_before_layer):
        idxes = beam_layer.id[beam_layer.id > 0]

        if len(idxes) != 0:
            self.start_moving_layer(beam_layer, idxes)

            beam_layer_idx = plasma_layer_idx - 1
            move_particles(self.grid_steps, self.grid_step_size,
                           self.xi_step_size, idxes, beam_layer_idx,
                           self.lost_radius,
                           beam_layer, fields_after_layer, fields_before_layer)

        self.stop_moving_layer(beam_layer)


class BeamSource:
    """
    This class helps to extract a beam layer from beam particles array.
    BeamSource guarantees that all particles in returned layer lie between
    xi_max and xi_min.
    
    """
    def __init__(self, config, beam):
        # From config:
        self.xi_step_size = config.getfloat('xi-step')
        
        # Get the whole beam or a beam layer:
        beam.xi_sorted()
        self.beam = beam

        # Dropped sorted_idxes = argsort(-self.beam.xi)...
        # It needs to be somewhere!

        # Shows how many particles have already deposited:
        self.layout_count = 0 # or _used_count in beam2d

    def get_beam_layer_to_layout(self, plasma_layer_idx):
        # xi_min = - self.xi_step_size * plasma_layer_idx
        xi_max = - self.xi_step_size * (plasma_layer_idx + 1)

        begin = self.layout_count

        # Does it find all particles that lay in the layer?
        # Doesn't look like this. Check it.
        # TODO: Clean this mess with so many minuses!
        #       Isn't that array sorted already? If yes, use searchsorted
        arr_to_search = -self.beam.xi[self.layout_count:]
        if len(arr_to_search) != 0:
            layer_length = np.argmax(arr_to_search > -xi_max)
        else:
            layer_length = 0
        self.layout_count += layer_length

        return self.beam.get_layer(begin, begin + layer_length)
