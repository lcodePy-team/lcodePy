import cupy as cp
import numba
import numba.cuda

from math import floor, sqrt

from ..config.config import Config
from .data import BeamParticles
from .weights import get_deposit_beam

WARP_SIZE = 32


# Deposition and interpolation helper function #

@numba.njit
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


# Deposition and field interpolation #

@numba.njit
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


@numba.njit
def particle_fields(x, y, xi, grid_steps, grid_step_size, xi_step_size, xi_k,
                    Ex_k_1, Ey_k_1, Ez_k_1, Bx_k_1, By_k_1, Bz_k_1,
                    Ex_k,   Ey_k,   Ez_k,   Bx_k,   By_k,   Bz_k):
    xi_loc = (xi - xi_k) / xi_step_size

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
    Moves one particle as far as possible on current xi layer. Based on
    Higuera-Cary method (https://doi.org/10.1063/1.4979989)
    """
    xi_k = beam_xi_layer * -xi_step_size  # xi_{k}
    xi_k_1 = (beam_xi_layer + 1) * -xi_step_size  # xi_{k+1}

    k = numba.cuda.grid(1)
    if k >= id.size:
        return
    
    q_m = q_m_[k]; dt = dt_[k]
    # TODO: We should use q, not q_m. Or use u, not p!
    q = sign(q_m)

    while remaining_steps[k] > 0:
        # 1. We have initial momentum and an initial position vector:
        opx, opy, opz = px[k], py[k], pz[k]
        ox, oy, oxi = x[k], y[k], xi[k]

        # 2. We calculate the posistion vector at half time step:
        m_gamma = sqrt((1 / q_m)**2 + opx**2 + opy**2 + opz**2)

        x_half  = ox  + dt / 2 * (opx / m_gamma)
        y_half  = oy  + dt / 2 * (opy / m_gamma)
        xi_half = oxi + dt / 2 * (opz / m_gamma - 1)

        if not_in_layer(xi_half, xi_k_1):
            # If the particle fells to the next layer, we quit this loop, but 
            # don't save any values. The particle will move to a new layer 
            # afterwards. Does it break depositing? Think about it.
            fell_idxes[k] = True
            break

        if is_lost(x_half, y_half, lost_radius):
            x[k], y[k], xi[k] = x_half, y_half, xi_half
            id[k] *= -1  # Particle hit the wall and is now lost
            lost_idxes[k] = True
            remaining_steps[k] = 0
            break

        # TODO: What if the particles is lost or not in layer?

        # 3. Iterpolate fiels on particle's position:
        Ex, Ey, Ez, Bx, By, Bz = particle_fields(
            x_half, y_half, xi_half, grid_steps, grid_step_size,
            xi_step_size, xi_k, Ex_k_1, Ey_k_1, Ez_k_1, Bx_k_1, By_k_1, Bz_k_1,
            Ex_k, Ey_k, Ez_k, Bx_k, By_k, Bz_k
        )

        # 4. Calculate the relativistic factor at half time step:
        px_m, bx = opx + q * dt / 2 * Ex, q * dt / 2 * Bx
        py_m, by = opy + q * dt / 2 * Ey, q * dt / 2 * By
        pz_m, bz = opz + q * dt / 2 * Ez, q * dt / 2 * Bz

        m_gamma_m = sqrt((1 / q_m)**2 + px_m**2 + py_m**2 + pz_m**2)

        b_sqr = bx**2 + by**2 + bz**2
        m_gamma_half = sqrt(
            (m_gamma_m**2 - b_sqr + sqrt(
                (m_gamma_m**2 - b_sqr)**2 + 4 * b_sqr +
                4 * (bx * px_m + by * py_m + bz * pz_m)**2
            )) / 2
        )

        # 5. Calculate auxiliary values:
        tx, ty, tz = bx / m_gamma_half, by / m_gamma_half, bz / m_gamma_half
        t_sqr = tx**2 + ty**2 + tz**2
        sx, sy, sz = 2*tx / (1 + t_sqr), 2*ty / (1 + t_sqr), 2*tz / (1 + t_sqr) 
        s_dot_p_m = sx * px_m + sy * py_m + sz * pz_m

        # 6. Compute a new momentum at full time step:
        px[k] = (   # px fullstep
            tx * s_dot_p_m + px_m * (1 - t_sqr) / (1 + t_sqr) +
            py_m * sz - pz_m * sy + q * dt / 2 * Ex
        )
        py[k] = (   # py fullstep
            ty * s_dot_p_m + py_m * (1 - t_sqr) / (1 + t_sqr) +
            pz_m * sx - px_m * sz + q * dt / 2 * Ey
        )
        pz[k] = (   # pz fullstep
            tz * s_dot_p_m + pz_m * (1 - t_sqr) / (1 + t_sqr) +
            px_m * sy - py_m * sx + q * dt / 2 * Ez
        )

        # 7. Calculate a new position vector at full time step:
        m_gamma_full = sqrt((1 / q_m)**2 + px[k]**2 + py[k]**2 + pz[k]**2)
        x[k]  = x_half  + dt / 2 * (px[k] / m_gamma_full)     #  x fullstep
        y[k]  = y_half  + dt / 2 * (py[k] / m_gamma_full)     #  y fullstep
        xi[k] = xi_half + dt / 2 * (pz[k] / m_gamma_full - 1) # xi fullstep

        if is_lost(x[k], y[k], lost_radius):
            id[k] *= -1  # Particle hit the wall and is now lost
            lost_idxes[k] = True
            remaining_steps[k] = 0
            break

        remaining_steps[k] -= 1
    
    # TODO: Do we need to add it here? (Yes, write why)
    if remaining_steps[k] == 0 and not_in_layer(xi[k], xi_k_1):
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

        self.deposit = get_deposit_beam(
            self.grid_steps, self.grid_step_size, self.xi_step_size)

    # Helper functions for one time step cicle:

    def start_time_step(self):
        """
        Perform necessary operations before starting the time step.
        """
        # Get a grid for beam rho density
        self.rho_layout = cp.zeros((self.grid_steps, self.grid_steps),
                                    dtype=cp.float64)

    # Helper functions for depositing beam particles of a layer:

    def layout_beam_layer(self, beam_layer: BeamParticles, plasma_layer_idx):
        rho_layout = cp.zeros((self.grid_steps, self.grid_steps),
                                    dtype=cp.float64)

        if beam_layer.id.size != 0:
            self.deposit(
                plasma_layer_idx, beam_layer.x, beam_layer.y, beam_layer.xi,
                beam_layer.q_norm, self.rho_layout, rho_layout)

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
