import cupy as cp
import numba

from math import floor, sqrt

WARP_SIZE = 32

# A new class for BeamParticles that is similar
# to GPUArrays in plasma3d_gpu.data.
# TODO: Use only one type of classes, we don't
#       really need BeamParticles.

class BeamParticles:
    def __init__(self, size):
        """
        Create a new empty array of beam particles. Can be used both as
        a whole beam particles array and as a layer of beam particles.
        """
        self.size = size

        self.xi = cp.zeros(size,     dtype=cp.float64)
        self.x = cp.zeros(size,      dtype=cp.float64)
        self.y = cp.zeros(size,      dtype=cp.float64)
        self.px = cp.zeros(size,     dtype=cp.float64)
        self.py = cp.zeros(size,     dtype=cp.float64)
        self.pz = cp.zeros(size,     dtype=cp.float64)
        self.q_m = cp.zeros(size,    dtype=cp.float64)
        self.q_norm = cp.zeros(size, dtype=cp.float64)
        self.id = cp.zeros(size,     dtype=cp.int64)
        self.dt = cp.zeros(size,     dtype=cp.float64)
        self.remaining_steps = cp.zeros(size,
                                     dtype=cp.int64)

    def load(self, *args, **kwargs):
        with cp.load(*args, **kwargs) as loaded:
            self.size = loaded['xi'].size
            self.xi = loaded['xi']
            self.x = loaded['x']
            self.y = loaded['y']
            self.px = loaded['px']
            self.py = loaded['py']
            self.pz = loaded['pz']
            self.q_m = loaded['q_m']
            self.q_norm = loaded['q_norm']
            self.id = loaded['id']
            self.dt = cp.zeros(self.size, dtype=cp.float64)
            self.remaining_steps = cp.zeros(self.size, dtype=cp.int64)

    def save(self, *args, **kwargs):
        cp.savez_compressed(*args, **kwargs,
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
        sort_idxes = cp.argsort(-self.xi)

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

    def get_layer(self, indexes_arr):
        """
        Return a layer with indexes from indexes_arr.
        """
        # TODO: Find a better method of getting a layer!
        #       Have a look at plasma3d_gpu.data for examples.
        new_beam_layer = BeamParticles(indexes_arr.size)

        new_beam_layer.xi = self.xi[indexes_arr]
        new_beam_layer.x = self.x[indexes_arr]
        new_beam_layer.y = self.y[indexes_arr]
        new_beam_layer.px = self.px[indexes_arr]
        new_beam_layer.py = self.py[indexes_arr]
        new_beam_layer.pz = self.pz[indexes_arr]
        new_beam_layer.q_m = self.q_m[indexes_arr]
        new_beam_layer.q_norm = self.q_norm[indexes_arr]
        new_beam_layer.id = self.id[indexes_arr]
        new_beam_layer.dt = self.dt[indexes_arr]
        new_beam_layer.remaining_steps = self.remaining_steps[indexes_arr]
        # new_beam_layer.lost =   self.lost[indexes_arr]

        return new_beam_layer


def concatenate_beam_layers(b_layer_1: BeamParticles, b_layer_2: BeamParticles):
    """
    Concatenate two beam particles layers.
    """
    new_b_layer = BeamParticles(b_layer_1.size + b_layer_2.size)
    # TODO: The same task as for self.get_sublayer()

    new_b_layer.xi =     cp.concatenate((b_layer_1.xi, b_layer_2.xi))
    new_b_layer.x =      cp.concatenate((b_layer_1.x, b_layer_2.x))
    new_b_layer.y =      cp.concatenate((b_layer_1.y, b_layer_2.y))
    new_b_layer.px =     cp.concatenate((b_layer_1.px, b_layer_2.px))
    new_b_layer.py =     cp.concatenate((b_layer_1.py, b_layer_2.py))
    new_b_layer.pz =     cp.concatenate((b_layer_1.pz, b_layer_2.pz))
    new_b_layer.q_m =    cp.concatenate((b_layer_1.q_m, b_layer_2.q_m))
    new_b_layer.q_norm = cp.concatenate((b_layer_1.q_norm, b_layer_2.q_norm))
    new_b_layer.id =     cp.concatenate((b_layer_1.id, b_layer_2.id))
    new_b_layer.dt =     cp.concatenate((b_layer_1.dt, b_layer_2.dt))
    new_b_layer.remaining_steps = cp.concatenate((b_layer_1.remaining_steps,
                                            b_layer_2.remaining_steps))

    return new_b_layer

#TODO: The BeamParticles class makes jitting harder. And we don't really need
#      this class. Get rid of it.

class BeamSource:
    """
    This class helps to extract a beam layer from beam particles array.
    """
    # Do we really need this class?
    def __init__(self, config, beam: BeamParticles):
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
        xi_min = - self.xi_step_size * plasma_layer_idx
        xi_max = - self.xi_step_size * (plasma_layer_idx + 1)

        begin = self.layout_count

        # Does it find all particles that lay in the layer? Check it.
        arr_to_search = self.beam.xi[begin:]
        if arr_to_search.size != 0:
            layer_length = cp.sum((cp.asarray(xi_max) < arr_to_search) *
                                  (arr_to_search <= cp.asarray(xi_min)))
        else:
            layer_length = 0
        self.layout_count += int(layer_length)

        indexes_arr = cp.arange(begin, begin + layer_length)
        return self.beam.get_layer(indexes_arr)


class BeamDrain:
    def __init__(self):
        self.beam_buffer = BeamParticles(0)
        self.lost_buffer = BeamParticles(0)

    def push_beam_layer(self, beam_layer: BeamParticles):
        if beam_layer.size > 0:
            self.beam_buffer = concatenate_beam_layers(self.beam_buffer,
                                                       beam_layer)

    def push_beam_lost(self, lost_layer: BeamParticles):
        if lost_layer.size > 0:
            self.lost_buffer = concatenate_beam_layers(self.lost_buffer,
                                                       lost_layer)


# Deposition and interpolation helper function #

@numba.njit
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


# Deposition and field interpolation #

@numba.cuda.jit
def deposit_kernel(grid_steps, grid_step_size,
                   x, y, xi_loc, q_norm,
                   rho_layout_0, rho_layout_1):
    """
    Deposit beam particles onto the charge density grids.
    """
    k = numba.cuda.grid(1)
    if k >= q_norm.size:
        return

    # Calculate the weights for a particle
    i, j, w000, w00P, w0P0, w0PP, wP00, wP0P, wPP0, wPPP = weights(
        x[k], y[k], xi_loc[k], grid_steps, grid_step_size
    )
    
    numba.cuda.atomic.add(rho_layout_0, (i + 0, j + 0), q_norm[k] * w000)
    numba.cuda.atomic.add(rho_layout_0, (i + 0, j + 1), q_norm[k] * w00P)
    numba.cuda.atomic.add(rho_layout_0, (i + 1, j + 0), q_norm[k] * w0P0)
    numba.cuda.atomic.add(rho_layout_0, (i + 1, j + 1), q_norm[k] * w0PP)
    numba.cuda.atomic.add(rho_layout_1, (i + 0, j + 0), q_norm[k] * wP00)
    numba.cuda.atomic.add(rho_layout_1, (i + 0, j + 1), q_norm[k] * wP0P)
    numba.cuda.atomic.add(rho_layout_1, (i + 1, j + 0), q_norm[k] * wPP0)
    numba.cuda.atomic.add(rho_layout_1, (i + 1, j + 1), q_norm[k] * wPPP)


def deposit(grid_steps, grid_step_size,
            x, y, xi_loc, q_norm,
            rho_layout_0, rho_layout_1):
    """
    Deposit beam particles onto the charge density grid.
    This is a convenience wrapper around the `deposit_kernel` CUDA kernel.
    """
    cfg = int(cp.ceil(q_norm.size / WARP_SIZE)), WARP_SIZE
    deposit_kernel[cfg](grid_steps, grid_step_size,
                        x.ravel(), y.ravel(),
                        xi_loc.ravel(), q_norm.ravel(),
                        rho_layout_0, rho_layout_1)


@numba.njit
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


@numba.njit
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

def beam_substepping_step(q_m, pz, substepping_energy):
    dt = cp.ones_like(q_m, dtype=cp.float64)
    max_dt = cp.sqrt(cp.sqrt(1 / q_m ** 2 + pz ** 2) / substepping_energy)
    
    a = cp.ceil(cp.log2(dt / max_dt))
    a[a < 0] = 0
    dt /= 2 ** a

    return dt


@numba.njit
def not_in_layer(xi, xi_k_1):
    return xi_k_1 > xi


@numba.njit
def is_lost(x, y, r_max):
    return x ** 2 + y ** 2 >= r_max ** 2
    # return abs(x) >= walls_width or abs(y) >= walls_width


@numba.njit
def sign(x):
    return -1 if x < 0 else 1 if x > 0 else 0


# Move beam particles #

@numba.cuda.jit
def move_particles_kernel(grid_steps, grid_step_size, xi_step_size,
                          beam_xi_layer, lost_radius,
                          q_m_, dt_, remaining_steps,
                          x, y, xi, px, py, pz, id,
                          Ex_k_1, Ey_k_1, Ez_k_1, Bx_k_1, By_k_1, Bz_k_1,
                          Ex_k,   Ey_k,   Ez_k,   Bx_k,   By_k,   Bz_k,
                          lost_idxes, moved_idxes, fell_idxes):
    """
    Moves one particle as far as possible on current xi layer.
    """
    xi_k = beam_xi_layer * -xi_step_size  # xi_{k}
    xi_k_1 = (beam_xi_layer + 1) * -xi_step_size  # xi_{k+1}

    k = numba.cuda.grid(1)
    if k >= id.size:
        return
    
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
            # TODO: Figure out how to tackle this problem!
            x[k], y[k], xi[k] = x_halfstep, y_halfstep, xi_halfstep
            fell_idxes[k] = True
            break

        if is_lost(x_halfstep, y_halfstep, lost_radius):
            x[k], y[k], xi[k] = x_halfstep, y_halfstep, xi_halfstep
            id[k] *= -1  # Particle hit the wall and is now lost
            lost_idxes[k] = True
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
        if not_in_layer(xi_halfstep, xi_k_1):
            fell_idxes[k] = True
            break

        if is_lost(x[k], y[k], lost_radius):
            id[k] *= -1  # Particle hit the wall and is now lost
            lost_idxes[k] = True
            remaining_steps[k] = 0
            break

        remaining_steps[k] -= 1
    
    if fell_idxes[k] == False and lost_idxes[k] == False:
        moved_idxes[k] = True


def move_particles(grid_steps, grid_step_size, xi_step_size,
                   idxes, beam_xi_layer, lost_radius,
                   beam, fields_k_1, fields_k,
                   lost_idxes, moved_idxes, fell_idxes):
    """
    This is a convenience wrapper around the `move_particles_kernel` CUDA kernel.
    """
    cfg = int(cp.ceil(idxes.size / WARP_SIZE)), WARP_SIZE

    x_new,  y_new,  xi_new = beam.x[idxes],  beam.y[idxes],  beam.xi[idxes]
    px_new, py_new, pz_new = beam.px[idxes], beam.py[idxes], beam.pz[idxes]
    id_new, remaining_steps_new = beam.id[idxes], beam.remaining_steps[idxes]

    move_particles_kernel[cfg](grid_steps, grid_step_size, xi_step_size,
                               beam_xi_layer, lost_radius,
                               beam.q_m[idxes], beam.dt[idxes],
                               remaining_steps_new,
                               x_new, y_new, xi_new,
                               px_new, py_new, pz_new,
                               id_new,
                               fields_k_1.Ex, fields_k_1.Ey, fields_k_1.Ez,
                               fields_k_1.Bx, fields_k_1.By, fields_k_1.Bz,
                               fields_k.Ex, fields_k.Ey, fields_k.Ez,
                               fields_k.Bx, fields_k.By, fields_k.Bz,
                               lost_idxes, moved_idxes, fell_idxes)

    beam.x[idxes],  beam.y[idxes],  beam.xi[idxes] = x_new,  y_new,  xi_new
    beam.px[idxes], beam.py[idxes], beam.pz[idxes] = px_new, py_new, pz_new
    beam.id[idxes], beam.remaining_steps[idxes] = id_new, remaining_steps_new
    
    return lost_idxes, moved_idxes, fell_idxes


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

    # Helper functions for one time step cicle:

    def start_time_step(self):
        """
        Perform necessary operations before starting the time step.
        """
        # Get a grid for beam rho density
        self.rho_layout = cp.zeros((self.grid_steps, self.grid_steps),
                                    dtype=cp.float64)

    # Helper functions for depositing beam particles of a layer:

    def layout_beam_layer(self, beam_layer, plasma_layer_idx):
        rho_layout = cp.zeros((self.grid_steps, self.grid_steps),
                                    dtype=cp.float64)

        if beam_layer.id.size != 0:
            xi_plasma_layer = - self.xi_step_size * plasma_layer_idx

            dxi = (xi_plasma_layer - beam_layer.xi) / self.xi_step_size
            deposit(self.grid_steps, self.grid_step_size,
                    beam_layer.x, beam_layer.y, dxi,
                    beam_layer.q_norm,
                    self.rho_layout, rho_layout)

        self.rho_layout, rho_layout = rho_layout, self.rho_layout
        rho_layout /= self.grid_step_size ** 2

        return rho_layout

    # Helper functions for moving beam particles of a layer:

    def start_moving_layer(self, beam_layer, idxes):
        """
        Perform necessary operations before moving a beam layer.
        """
        # TODO: Do we need to set dt and remaining_steps only for particles
        #       that have dt == 0?
        # mask = beam_layer.id[beam_layer.dt == 0] and idxes -> mask ???
        dt = beam_substepping_step(beam_layer.q_m[idxes], beam_layer.pz[idxes],
                                   self.substepping_energy)
        beam_layer.dt[idxes] = dt * self.time_step
        beam_layer.remaining_steps[idxes] = (1. / dt).astype(cp.int_)

    def move_beam_layer(self, beam_layer: BeamParticles, fell_size,
                        pl_layer_idx, fields_after_layer, fields_before_layer):
        idxes_1 = cp.arange(beam_layer.id.size - fell_size)
        idxes_2 = cp.arange(beam_layer.id.size)

        size = idxes_2.size
        lost_idxes  = cp.zeros(size, dtype=cp.bool8)
        moved_idxes = cp.zeros(size, dtype=cp.bool8)
        fell_idxes  = cp.zeros(size, dtype=cp.bool8)

        if size != 0:
            self.start_moving_layer(beam_layer, idxes_1)
            beam_layer_to_move_idx = pl_layer_idx - 1

            lost_idxes, moved_idxes, fell_idxes = move_particles(
                self.grid_steps, self.grid_step_size,
                self.xi_step_size, idxes_2, beam_layer_to_move_idx,
                self.lost_radius,
                beam_layer, fields_after_layer, fields_before_layer,
                lost_idxes, moved_idxes, fell_idxes)

        lost  = beam_layer.get_layer(idxes_2[lost_idxes])
        moved = beam_layer.get_layer(idxes_2[moved_idxes])
        fell  = beam_layer.get_layer(idxes_2[fell_idxes])

        return lost, moved, fell
