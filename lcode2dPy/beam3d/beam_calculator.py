import numpy as np
import numba as nb

from math import floor, sqrt

from lcode2dPy.config.config import Config
from lcode2dPy.beam3d.data import BeamParticles


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
    Calculates the position and the weights of a beam particleon a 3d cartesian
    grid.
    """
    x_h, y_h = x / grid_step_size + .5, y / grid_step_size + .5
    i, j = int(floor(x_h) + grid_steps // 2), int(floor(y_h) + grid_steps // 2)
    x_loc, y_loc = x_h - floor(x_h) - .5, y_h - floor(y_h) - .5
    # xi_loc = dxi = (xi_prev - xi) / D_XIP

    # First order core along xi axis
    wxi0 = xi_loc
    wxiP = (1 - xi_loc)

    # Second order core along x and y axes
    wx0, wy0 = .75 - x_loc**2, .75 - y_loc**2
    wxP, wyP = (.5 + x_loc)**2 / 2, (.5 + y_loc)**2 / 2
    wxM, wyM = (.5 - x_loc)**2 / 2, (.5 - y_loc)**2 / 2

    w0MP, w00P, w0PP = wxi0 * wxM * wyP, wxi0 * wx0 * wyP, wxi0 * wxP * wyP
    w0M0, w000, w0P0 = wxi0 * wxM * wy0, wxi0 * wx0 * wy0, wxi0 * wxP * wy0
    w0MM, w00M, w0PM = wxi0 * wxM * wyM, wxi0 * wx0 * wyM, wxi0 * wxP * wyM

    wPMP, wP0P, wPPP = wxiP * wxM * wyP, wxiP * wx0 * wyP, wxiP * wxP * wyP
    wPM0, wP00, wPP0 = wxiP * wxM * wy0, wxiP * wx0 * wy0, wxiP * wxP * wy0
    wPMM, wP0M, wPPM = wxiP * wxM * wyM, wxiP * wx0 * wyM, wxiP * wxP * wyM

    return (i, j,
            w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
            wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM
    )

# TODO: we have similar functions for GPU in lcode3d code


# Deposition and field interpolation #

@nb.njit #(parallel=True)
def deposit_kernel(grid_steps, grid_step_size,
                   x, y, xi_loc, q_norm,
                   rho_layout_0, rho_layout_1):
    """
    Deposit beam particles onto the charge density grids.
    """
    for k in nb.prange(len(q_norm)):

        # Calculate the weights for a particle
        (i, j,
        w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
        wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM
        ) = weights(
            x[k], y[k], xi_loc[k], grid_steps, grid_step_size
        )

        rho_layout_0[i - 1, j + 1] += q_norm[k] * w0MP
        rho_layout_0[i + 0, j + 1] += q_norm[k] * w00P
        rho_layout_0[i + 1, j + 1] += q_norm[k] * w0PP
        rho_layout_0[i - 1, j + 0] += q_norm[k] * w0M0
        rho_layout_0[i + 0, j + 0] += q_norm[k] * w000
        rho_layout_0[i + 1, j + 0] += q_norm[k] * w0P0
        rho_layout_0[i - 1, j - 1] += q_norm[k] * w0MM
        rho_layout_0[i + 0, j - 1] += q_norm[k] * w00M
        rho_layout_0[i + 1, j - 1] += q_norm[k] * w0PM

        rho_layout_1[i - 1, j + 1] += q_norm[k] * wPMP
        rho_layout_1[i + 0, j + 1] += q_norm[k] * wP0P
        rho_layout_1[i + 1, j + 1] += q_norm[k] * wPPP
        rho_layout_1[i - 1, j + 0] += q_norm[k] * wPM0
        rho_layout_1[i + 0, j + 0] += q_norm[k] * wP00
        rho_layout_1[i + 1, j + 0] += q_norm[k] * wPP0
        rho_layout_1[i - 1, j - 1] += q_norm[k] * wPMM
        rho_layout_1[i + 0, j - 1] += q_norm[k] * wP0M
        rho_layout_1[i + 1, j - 1] += q_norm[k] * wPPM

def deposit(grid_steps, grid_step_size,
            x, y, xi_loc, q_norm,
            rho_layout_0, rho_layout_1):
    """
    Deposit beam particles onto the charge density grid.
    This is a convenience wrapper around the `deposit_kernel` CUDA kernel.
    """
    deposit_kernel(grid_steps, grid_step_size,
                        x.ravel(), y.ravel(),
                        xi_loc.ravel(), q_norm.ravel(),
                        rho_layout_0, rho_layout_1)


@nb.njit
def interp(value_0, value_1, i, j,
           w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
           wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM):
    """
    Collect value from a cell and surrounding cells (using `weights` output).
    """
    return (
        value_0[i - 1, j + 1] * w0MP +
        value_0[i + 0, j + 1] * w00P +
        value_0[i + 1, j + 1] * w0PP +
        value_0[i - 1, j + 0] * w0M0 +
        value_0[i + 0, j + 0] * w000 +
        value_0[i + 1, j + 0] * w0P0 +
        value_0[i - 1, j - 1] * w0MM +
        value_0[i + 0, j - 1] * w00M +
        value_0[i + 1, j - 1] * w0PM +
    
        value_1[i - 1, j + 1] * wPMP +
        value_1[i + 0, j + 1] * wP0P +
        value_1[i + 1, j + 1] * wPPP +
        value_1[i - 1, j + 0] * wPM0 +
        value_1[i + 0, j + 0] * wP00 +
        value_1[i + 1, j + 0] * wPP0 +
        value_1[i - 1, j - 1] * wPMM +
        value_1[i + 0, j - 1] * wP0M +
        value_1[i + 1, j - 1] * wPPM
    )


@nb.njit
def particle_fields(x, y, xi, grid_steps, grid_step_size, xi_step_size, xi_k,
                    Ex_k_1, Ey_k_1, Ez_k_1, Bx_k_1, By_k_1, Bz_k_1,
                    Ex_k,   Ey_k,   Ez_k,   Bx_k,   By_k,   Bz_k):
    xi_loc = (xi_k - xi) / xi_step_size

    (i, j,
    w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
    wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM
    ) = weights(
        x, y, xi_loc, grid_steps, grid_step_size
    )

    Ex = interp(Ex_k, Ex_k_1, i, j,
                w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
                wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM)
    Ey = interp(Ey_k, Ey_k_1, i, j,
                w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
                wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM)
    Ez = interp(Ez_k, Ez_k_1, i, j,
                w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
                wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM)
    Bx = interp(Bx_k, Bx_k_1, i, j,
                w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
                wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM)
    By = interp(By_k, By_k_1, i, j,
                w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
                wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM)
    Bz = interp(Bz_k, Bz_k_1, i, j,
                w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
                wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM)

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
                          x, y, xi, px, py, pz, id,
                          Ex_k_1, Ey_k_1, Ez_k_1, Bx_k_1, By_k_1, Bz_k_1,
                          Ex_k,   Ey_k,   Ez_k,   Bx_k,   By_k,   Bz_k,
                          lost_idxes, moved_idxes, fell_idxes):
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

            if is_lost(x[k], y[k], lost_radius):
                id[k] *= -1  # Particle hit the wall and is now lost
                lost_idxes[k] = True
                remaining_steps[k] = 0
                break

            remaining_steps[k] -= 1
        
        # TODO: Do we need to add it here? (Yes, write why)
        if remaining_steps[k] == 0 and not_in_layer(xi_halfstep, xi_k_1):
            fell_idxes[k] = True

        if fell_idxes[k] == False and lost_idxes[k] == False:
            moved_idxes[k] = True


def move_particles(grid_steps, grid_step_size, xi_step_size,
                   idxes, beam_xi_layer, lost_radius,
                   beam: BeamParticles, fields_k_1, fields_k,
                   lost_idxes, moved_idxes, fell_idxes):
    """
    This is a convenience wrapper around the `move_particles_kernel` CUDA kernel.
    """
    x_new,  y_new,  xi_new = beam.x[idxes],  beam.y[idxes],  beam.xi[idxes]
    px_new, py_new, pz_new = beam.px[idxes], beam.py[idxes], beam.pz[idxes]
    id_new, remaining_steps_new = beam.id[idxes], beam.remaining_steps[idxes]

    move_particles_kernel(grid_steps, grid_step_size, xi_step_size,
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
    def __init__(self, config: Config):
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
        self.rho_layout = np.zeros((self.grid_steps, self.grid_steps),
                                    dtype=np.float64)

    # Helper functions for depositing beam particles of a layer:

    def layout_beam_layer(self, beam_layer: BeamParticles, plasma_layer_idx):
        rho_layout = np.zeros((self.grid_steps, self.grid_steps),
                                    dtype=np.float64)

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

    def start_moving_layer(self, beam_layer: BeamParticles, idxes):
        """
        Perform necessary operations before moving a beam layer.
        """
        # TODO: Do we need to set dt and remaining_steps only for particles
        #       that have dt == 0?
        # mask = beam_layer.id[beam_layer.dt == 0] and idxes -> mask ???
        dt = beam_substepping_step(beam_layer.q_m[idxes], beam_layer.pz[idxes],
                                   self.substepping_energy)
        beam_layer.dt[idxes] = dt * self.time_step
        beam_layer.remaining_steps[idxes] = (1. / dt).astype(np.int_)

    def move_beam_layer(self, beam_layer: BeamParticles, fell_size,
                        pl_layer_idx, fields_after_layer, fields_before_layer):
        idxes_1 = np.arange(beam_layer.id.size - fell_size)
        idxes_2 = np.arange(beam_layer.id.size)

        size = idxes_2.size
        lost_idxes  = np.zeros(size, dtype=np.bool8)
        moved_idxes = np.zeros(size, dtype=np.bool8)
        fell_idxes  = np.zeros(size, dtype=np.bool8)

        if len(idxes_2) != 0:
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
