"""Module for weights interpolation and movement routines."""
import numba as nb

from math import sqrt, floor

from ..config.config import Config
from ..beam3d.data import BeamParticles
from ..beam3d.weights import weight1, weight4, weight1_cupy, weight4_cupy


# Beam particles mover auxiliary functions, for CPU #

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
def move_beam_particles_kernel_numba(
    xi_step_size, lost_radius, beam_layer_idx, grid_step_size, grid_steps,
    q_m, dt,
    Ex_k,  Ey_k,  Ez_k,  Bx_k,  By_k,  Bz_k,
    Ex_k1, Ey_k1, Ez_k1, Bx_k1, By_k1, Bz_k1,
    remaining_steps, id, x, y, xi, px, py, pz,
    lost_idxes, moved_idxes, fell_idxes, size):
    """
    Moves one particle as far as possible on current xi layer.
    """
    xi_k = beam_layer_idx * -xi_step_size  # xi_{k}
    xi_k_1 = (beam_layer_idx + 1) * -xi_step_size  # xi_{k+1}

    for k in nb.prange(size):
        while remaining_steps[k] > 0:
            # Compute approximate position of the particle in the middle of the step
            gamma_m = sqrt(
                (1 / q_m[k]) ** 2 + px[k] ** 2 + py[k]** 2 + pz[k] ** 2)

            x_halfstep  = x[k]  + dt[k] / 2 * (px[k] / gamma_m)
            y_halfstep  = y[k]  + dt[k] / 2 * (py[k] / gamma_m)
            xi_halfstep = xi[k] + dt[k] / 2 * (pz[k] / gamma_m - 1)
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

            # Interpolate fields
            x_h = x_halfstep / grid_step_size + .5
            y_h = y_halfstep / grid_step_size + .5
            x_loc = x_h - floor(x_h) - .5
            y_loc = y_h - floor(y_h) - .5
            xi_loc = (xi_halfstep - xi_k) / xi_step_size
            ix = int(floor(x_h) + grid_steps // 2)
            iy = int(floor(y_h) + grid_steps // 2)

            Ex, Ey, Ez, Bx, By, Bz = 0, 0, 0, 0, 0, 0
            for kx in range(-2, 3):
                wx = weight4(x_loc, kx)
                for ky in range(-2, 3):
                    w = wx * weight4(y_loc, ky)
                    w0 = w * weight1(xi_loc, 0)
                    w1 = w * weight1(xi_loc, 1)
                    idx_x, idx_y = ix + kx, iy + ky

                    # Collect value from a cell and 8 surrounding cells.
                    Ex += Ex_k[idx_x, idx_y] * w0 + Ex_k1[idx_x, idx_y] * w1
                    Ey += Ey_k[idx_x, idx_y] * w0 + Ey_k1[idx_x, idx_y] * w1
                    Ez += Ez_k[idx_x, idx_y] * w0 + Ez_k1[idx_x, idx_y] * w1
                    Bx += Bx_k[idx_x, idx_y] * w0 + Bx_k1[idx_x, idx_y] * w1
                    By += By_k[idx_x, idx_y] * w0 + By_k1[idx_x, idx_y] * w1
                    Bz += Bz_k[idx_x, idx_y] * w0 + Bz_k1[idx_x, idx_y] * w1

            # Compute new impulse
            vx, vy, vz = px[k] / gamma_m, py[k] / gamma_m, pz[k] / gamma_m
            px_halfstep = (
                px[k] + sign(q_m[k]) * dt[k] / 2 * (Ex + vy * Bz - vz * By))
            py_halfstep = (
                py[k] + sign(q_m[k]) * dt[k] / 2 * (Ey + vz * Bx - vx * Bz))
            pz_halfstep = (
                pz[k] + sign(q_m[k]) * dt[k] / 2 * (Ez + vx * By - vy * Bx))

            # Compute final coordinates and impulses
            gamma_m = sqrt((1 / q_m[k]) ** 2
                        + px_halfstep ** 2 + py_halfstep ** 2 + pz_halfstep ** 2)

            x[k]  += dt[k] * (px_halfstep / gamma_m)      #  x fullstep
            y[k]  += dt[k] * (py_halfstep / gamma_m)      #  y fullstep
            xi[k] += dt[k] * (pz_halfstep / gamma_m - 1)  # xi fullstep

            px[k] = 2 * px_halfstep - px[k]               # px fullstep
            py[k] = 2 * py_halfstep - py[k]               # py fullstep
            pz[k] = 2 * pz_halfstep - pz[k]               # pz fullstep

            if is_lost(x[k], y[k], lost_radius):
                id[k] *= -1  # Particle hit the wall and is now lost
                lost_idxes[k] = True
                remaining_steps[k] = 0
                break

            remaining_steps[k] -= 1

        # TODO: Do we need to add it here? (Yes, write why)
        if remaining_steps[k] == 0 and not_in_layer(x[k], xi_k_1):
            fell_idxes[k] = True

        if fell_idxes[k] == False and lost_idxes[k] == False:
            moved_idxes[k] = True


# Beam particles mover, for GPU #

def get_move_beam_particles_kernel_cupy():
    """
    Moves beam particles as far as possible on current xi layer. Based on
    Higuera-Cary method (https://doi.org/10.1063/1.4979989)
    """
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, float64 r_max, float64 beam_layer_idx,
        float64 grid_step_size, float64 grid_steps,
        raw T q_m, raw T dt,
        raw T Ex_k,  raw T Ey_k,  raw T Ez_k,
        raw T Bx_k,  raw T By_k,  raw T Bz_k,
        raw T Ex_k1, raw T Ey_k1, raw T Ez_k1,
        raw T Bx_k1, raw T By_k1, raw T Bz_k1
        """,
        out_params="""
        raw int64 out_remaining_steps, raw int64 out_id,
        raw T out_x, raw T out_y, raw T out_xi,
        raw T out_px, raw T out_py, raw T out_pz,
        raw bool lost_idxes, raw bool moved_idxes, raw bool fell_idxes
        """,
        operation="""
        const double xi_k   = beam_layer_idx       * (-xi_step_size);
        const double xi_k_1 = (beam_layer_idx + 1) * (-xi_step_size);

        double q = copysign(1.0, q_m[i]);

        while (out_remaining_steps[i] > 0) {
            // 1. We have an initial momentum and an initial position vector:

            // 2. Calculate the position vector at half time step:
            T gamma_m = sqrt((1. / q_m[i])*(1. / q_m[i]) +
                out_px[i]*out_px[i] + out_py[i]*out_py[i] + out_pz[i]*out_pz[i]
            );

            T x_half  = out_x[i]  + dt[i] / 2. * (out_px[i] / gamma_m);
            T y_half  = out_y[i]  + dt[i] / 2. * (out_py[i] / gamma_m);
            T xi_half = out_xi[i] + dt[i] / 2. * (out_pz[i] / gamma_m - 1);

            if (xi_half < xi_k_1) {
                // If the particle fells to the next layer, we quit this loop,
                // but don't save any values. The particle will move to a new
                // layer afterwards. Does it break depositing? Think about it.
                fell_idxes[i] = true;
                break;
            }

            if (x_half*x_half + y_half*y_half >= r_max*r_max) {
                // Particle hit the wall and is now lost
                out_x[i] = x_half, out_y[i] = y_half, out_xi[i] = xi_half;
                out_id[i] *= -1;
                lost_idxes[i] = true;
                out_remaining_steps[i] = 0;
                break;
            }

            // 3. Interpolate fields from the grid on the particle position:
            T x_h = x_half / (T) grid_step_size + 0.5;
            T y_h = y_half / (T) grid_step_size + 0.5;
            T x_loc = x_h - floor(x_h) - 0.5;
            T y_loc = y_h - floor(y_h) - 0.5;
            T xi_loc = ((T) xi_half - xi_k) / (T) xi_step_size;
            int ix = floor(x_h) + floor(grid_steps / 2);
            int iy = floor(y_h) + floor(grid_steps / 2);

            T Ex = 0, Ey = 0, Ez = 0, Bx = 0, By = 0, Bz = 0;
            for (int kx = -2; kx <= 2; kx++) {
                const T wx = weight4(x_loc, kx);
                for (int ky = -2; ky <= 2; ky++) {
                    const T w  = wx * weight4(y_loc,  ky);
                    const T w0 = w  * weight1(xi_loc, 0);
                    const T w1 = w  * weight1(xi_loc, 1);
                    const int idx = (iy + ky) + (int) grid_steps * (ix + kx);

                    Ex += Ex_k[idx] * w0 + Ex_k1[idx] * w1;
                    Bx += Bx_k[idx] * w0 + Bx_k1[idx] * w1;
                    Ey += Ey_k[idx] * w0 + Ey_k1[idx] * w1;
                    By += By_k[idx] * w0 + By_k1[idx] * w1;
                    Ez += Ez_k[idx] * w0 + Ez_k1[idx] * w1;
                    Bz += Bz_k[idx] * w0 + Bz_k1[idx] * w1;
                }
            }

            // 4. Calculate the relativistic factor at half time step:
            T px_m = out_px[i] + q * dt[i] / 2. * Ex, bx = q * dt[i] / 2. * Bx;
            T py_m = out_py[i] + q * dt[i] / 2. * Ey, by = q * dt[i] / 2. * By;
            T pz_m = out_pz[i] + q * dt[i] / 2. * Ez, bz = q * dt[i] / 2. * Bz;
            gamma_m = sqrt((1. / q_m[i])*(1. / q_m[i]) +
                            px_m*px_m + py_m*py_m + pz_m*pz_m);

            T b_sq = bx*bx + by*by + bz*bz;
            gamma_m = sqrt((gamma_m*gamma_m - b_sq + sqrt(
                (gamma_m*gamma_m - b_sq)*(gamma_m*gamma_m - b_sq) + 4 * b_sq +
                4 * (bx * px_m + by * py_m + bz * pz_m)*
                    (bx * px_m + by * py_m + bz * pz_m)
            )) / 2.);

            // 5. Calculate auxiliary values:
            T tx = bx / gamma_m, ty = by / gamma_m, tz = bz / gamma_m;
            T t_sq_pl = 1. + tx*tx + ty*ty + tz*tz;
            T t_sq_mi = 1. - tx*tx - ty*ty - tz*tz;
            T sx = 2.*tx / t_sq_pl, sy = 2.*ty / t_sq_pl, sz = 2.*tz / t_sq_pl;
            T s_dot_p_m = sx * px_m + sy * py_m + sz * pz_m;

            // 6. Compute a new momentum at full time step:
            out_px[i] = (tx * s_dot_p_m + px_m * t_sq_mi / t_sq_pl + 
                         py_m * sz - pz_m * sy + q * dt[i] / 2. * Ex);          
            out_py[i] = (ty * s_dot_p_m + py_m * t_sq_mi / t_sq_pl + 
                         pz_m * sx - px_m * sz + q * dt[i] / 2. * Ey);
            out_pz[i] = (tz * s_dot_p_m + pz_m * t_sq_mi / t_sq_pl + 
                         px_m * sy - py_m * sx + q * dt[i] / 2. * Ez);
            
            // 7. Calculate a new position vector at full time step:
            gamma_m = sqrt((1. / q_m[i])*(1. / q_m[i]) +
                out_px[i]*out_px[i] + out_py[i]*out_py[i] + out_pz[i]*out_pz[i]
            );
            out_x[i]  = x_half  + dt[i] / 2. * (out_px[i] / gamma_m);
            out_y[i]  = y_half  + dt[i] / 2. * (out_py[i] / gamma_m);
            out_xi[i] = xi_half + dt[i] / 2. * (out_pz[i] / gamma_m - 1) ;

            if (out_x[i]*out_x[i] + out_y[i]*out_y[i] >= r_max*r_max) {
                // Particle hit the wall and is now lost
                out_id[i] *= -1;
                lost_idxes[i] = true;
                out_remaining_steps[i] = 0;
                break;
            }

            out_remaining_steps[i] -= 1;
        }

        // TODO: Do we need to add it here? (Yes, write why)
        if (out_remaining_steps[i] == 0 && out_xi[i] < xi_k_1) {
            fell_idxes[i] = true;
        }

        if (fell_idxes[i] == false && lost_idxes[i] == false) {
            moved_idxes[i] = true;
        }
        """,
        name='move_beam_cupy', preamble=weight1_cupy+weight4_cupy,
        no_return=True
    )


def get_move_beam_particles(config: Config):
    xi_step_size = config.getfloat('xi-step')
    grid_step_size = config.getfloat('window-width-step-size')
    grid_steps = config.getint('window-width-steps')

    # Calculate the radius that marks that a particle is lost.
    max_radius = grid_step_size * grid_steps / 2
    lost_radius = max(0.9 * max_radius, max_radius - 1) # or just max_radius?

    pu_type = config.get('processing-unit-type').lower()
    if pu_type == 'cpu':
        move_beam_particles_kernel = move_beam_particles_kernel_numba
    if pu_type == 'gpu':
        move_beam_particles_kernel = get_move_beam_particles_kernel_cupy()

    def move_beam_particles(beam_layer_idx, beam_layer: BeamParticles, fields_k,
                            fields_k_1, lost_idxes, moved_idxes, fell_idxes):
        move_beam_particles_kernel(
            xi_step_size, lost_radius, beam_layer_idx, grid_step_size,
            grid_steps, beam_layer.q_m, beam_layer.dt,
            fields_k.Ex, fields_k.Ey, fields_k.Ez,
            fields_k.Bx, fields_k.By, fields_k.Bz,
            fields_k_1.Ex, fields_k_1.Ey, fields_k_1.Ez,
            fields_k_1.Bx, fields_k_1.By, fields_k_1.Bz,

            beam_layer.remaining_steps, beam_layer.id,
            beam_layer.x, beam_layer.y, beam_layer.xi,
            beam_layer.px, beam_layer.py, beam_layer.pz,
            lost_idxes, moved_idxes, fell_idxes,

            size=beam_layer.id.size)

    return move_beam_particles
