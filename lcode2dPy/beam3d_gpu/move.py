"""Module for weights interpolation and movement routines."""
from .data import BeamParticles
from .weights import weight1_cupy, weight4_cupy

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


def get_move_beam(grid_steps, grid_step_size, xi_step_size):
    import cupy as cp

    # Calculate the radius that marks that a particle is lost.
    max_radius = grid_step_size * grid_steps / 2
    lost_radius = max(0.9 * max_radius, max_radius - 1) # or just max_radius?

    move_beam_cupy = get_move_beam_cupy()

    def move_beam(idxes, beam_layer_idx, beam: BeamParticles,
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
