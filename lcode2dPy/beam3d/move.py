"""Module for weights interpolation and movement routines."""
import numpy as np
import numba as nb
from math import sqrt

from ..config.config import Config
from ..beam3d.weights import weights, weight1_cupy, weight4_cupy


# Beam particles mover, for CPU #

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
                   beam, fields_k_1, fields_k,
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


# Beam particles mover, for GPU #

def get_move_beam_cupy():
    import cupy as cp
    
    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, float64 r_max, float64 beam_xi_layer,
        float64 grid_step_size, float64 grid_steps,
        raw T q_m, raw T dt, raw int64 remaining_steps, raw int64 id,
        raw T x, raw T y, raw T xi, raw T px, raw T py, raw T pz,
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
        const double xi_k   = beam_xi_layer       * (-xi_step_size);
        const double xi_k_1 = (beam_xi_layer + 1) * (-xi_step_size);

        double q = copysign(1.0, q_m[i]);

        out_remaining_steps[i] = remaining_steps[i], out_id[i] = id[i];
        out_px[i] = px[i], out_py[i] = py[i], out_pz[i] = pz[i];
        out_x[i]  = x[i],  out_y[i]  = y[i],  out_xi[i] = xi[i];

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

                    Ex += Ex_k[i] * w0 + Ex_k1[i] * w1;
                    Bx += Bx_k[i] * w0 + Bx_k1[i] * w1;
                    Ey += Ey_k[i] * w0 + Ey_k1[i] * w1;
                    By += By_k[i] * w0 + By_k1[i] * w1;
                    Ez += Ez_k[i] * w0 + Ez_k1[i] * w1;
                    Bz += Bz_k[i] * w0 + Bz_k1[i] * w1;
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
            T t_sq_mi = 1. - tx*tx + ty*ty + tz*tz;
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
        if (remaining_steps[i] == 0 && out_xi[i] < xi_k_1) {
            fell_idxes[i] = true;
        }

        if (fell_idxes[i] == false && lost_idxes[i] == false) {
            moved_idxes[i] = true;
        }
        

        """,
        name='move_beam_cupy', preamble=weight1_cupy+weight4_cupy
    )


def get_move_beam(config: Config):
    xi_step_size = config.getfloat('xi-step')
    grid_step_size = config.getfloat('window-width-step-size')
    grid_steps = config.getint('window-width-steps')
    pu_type = config.get('processing-unit-type').lower()

    # Calculate the radius that marks that a particle is lost.
    max_radius = grid_step_size * grid_steps / 2
    lost_radius = max(0.9 * max_radius, max_radius - 1) # or just max_radius?

    if pu_type == 'cpu':
        def move_beam(idxes, beam_layer_idx, beam,
                      fields_k, fields_k_1, lost_idxes, moved_idxes, fell_idxes):
            return move_particles(grid_steps, grid_step_size, xi_step_size,
                                  idxes, beam_layer_idx, lost_radius,
                                  beam, fields_k_1, fields_k,
                                  lost_idxes, moved_idxes, fell_idxes)

    if pu_type == 'gpu':
        import cupy as cp

        move_beam_cupy = get_move_beam_cupy()

        def move_beam(idxes, beam_layer_idx, beam,
                      fields_k, fields_k_1, lost_idxes, moved_idxes, fell_idxes):
            """
            Moves beam particles as far as possible on current xi layer. Based on
            Higuera-Cary method (https://doi.org/10.1063/1.4979989)
            """
            # TODO: This is horribly unoptimized!
            rem_steps_new = cp.zeros_like(beam.remaining_steps[idxes])
            id_new = cp.zeros_like(beam.id[idxes])
            x_new  = cp.zeros_like(beam.x[idxes])
            y_new  = cp.zeros_like(beam.y[idxes])
            xi_new = cp.zeros_like(beam.xi[idxes])
            px_new = cp.zeros_like(beam.px[idxes])
            py_new = cp.zeros_like(beam.py[idxes])
            pz_new = cp.zeros_like(beam.pz[idxes])

            (rem_steps_new, id_new, x_new, y_new, xi_new, px_new, py_new, pz_new,
            lost_idxes, moved_idxes, fell_idxes) = move_beam_cupy(
                xi_step_size, lost_radius, beam_layer_idx, grid_step_size,
                grid_steps, beam.q_m[idxes], beam.dt[idxes],
                beam.remaining_steps[idxes], beam.id[idxes],
                beam.x[idxes],  beam.y[idxes],  beam.xi[idxes],
                beam.px[idxes], beam.py[idxes], beam.pz[idxes],
                fields_k.Ex, fields_k.Ey, fields_k.Ez,
                fields_k.Bx, fields_k.By, fields_k.Bz,
                fields_k_1.Ex, fields_k_1.Ey, fields_k_1.Ez,
                fields_k_1.Bx, fields_k_1.By, fields_k_1.Bz,

                rem_steps_new, id_new,
                x_new, y_new, xi_new, px_new, py_new, pz_new,
                lost_idxes, moved_idxes, fell_idxes,

                size=idxes.size
            )

            beam.remaining_steps[idxes], beam.id[idxes] = rem_steps_new, id_new
            beam.x[idxes],  beam.y[idxes],  beam.xi[idxes] = x_new,  y_new,  xi_new
            beam.px[idxes], beam.py[idxes], beam.pz[idxes] = px_new, py_new, pz_new

            return lost_idxes, moved_idxes, fell_idxes

    return move_beam
