"""Module for calculation of particles movement and interpolation of fields on particles positions."""
import numba as nb
import numpy as np

from math import sqrt, floor

from ..config.config import Config
from .weights import weight4, weight4_cupy
from .data import Arrays

# Field interpolation and particle movement (fused), for CPU #
        #    move_smart_kernel(
        #        xi_step_size, boundary, grid_step_size, grid_steps,
        #        
        #        fields.Ex, fields.Ey, fields.Ez,
        #        fields.Bx, fields.By, fields.Bz,

        #        particles_prev[sort].x_init, particles_prev[sort].y_init,

        #        particles_prev[sort].x_offt, particles_prev[sort].y_offt,
        #        particles_prev[sort].px, particles_prev[sort].py, 
        #        particles_prev[sort].pz,

        #        
        #        particles_full[sort].m, particles_full[sort].q, 
        #        particles_full[sort].x_offt, particles_full[sort].y_offt,

@nb.njit(inline='always')
def _correct_boundary_particles(x_init, y_init, x_offt, y_offt, 
                                px, py, pz, boundary): 
    x = x_init + x_offt
    y = y_init + y_offt
    if x > +boundary:
        x = +2 * boundary - x
        x_offt = x - x_init
        px = py = pz = 0
    if x < -boundary:
        x = -2 * boundary - x
        x_offt = x - x_init
        px = py = pz = 0
    if y > +boundary:
        y = +2 * boundary - y
        y_offt = y - y_init
        px = py = pz = 0
    if y < -boundary:
        y = -2 * boundary - y
        y_offt = y - y_init
        px = py = pz = 0
    return x, y, x_offt, y_offt, px, py, pz


@nb.njit(parallel=True, cache=True)
def move_smart_kernel_numba(
    xi_step_size, boundary, grid_step_size, grid_steps,
    Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg,
    x_init, y_init, 
    x_offt_prev, y_offt_prev, px_prev, py_prev, pz_prev,
    ms, qs,
    x_offt_full, y_offt_full, px_full, py_full, pz_full,
    size):
    """
    Update plasma particle coordinates and momenta according to the field
    values interpolated halfway between the previous plasma particle location
    and the the best estimation of its next location currently available to us.
    Also absorb the particles at `+-boundary`.
    """
    ms, qs = ms.ravel(), qs.ravel()
    x_init, y_init = x_init.ravel(), y_init.ravel()
    x_offt_prev, y_offt_prev = x_offt_prev.ravel(), y_offt_prev.ravel()
    x_offt_full, y_offt_full = x_offt_full.ravel(), y_offt_full.ravel()
    px_prev, py_prev, pz_prev = px_prev.ravel(), py_prev.ravel(), pz_prev.ravel()
    px_full, py_full, pz_full = px_full.ravel(), py_full.ravel(), pz_full.ravel()

    for k in nb.prange(size):
        if qs[k] == 0:
            continue
        m, q = ms[k], qs[k]

        opx, opy, opz = px_prev[k], py_prev[k], pz_prev[k]
        px, py, pz = opx, opy, opz
        x_offt, y_offt = x_offt_prev[k], y_offt_prev[k]
        # Calculate midstep positions and fields in them.
        x_offt_aver = (x_offt_prev[k] + x_offt_full[k]) / 2
        y_offt_aver = (y_offt_prev[k] + y_offt_full[k]) / 2
        x_halfstep, y_halfstep, _, _, px, py, pz = \
            _correct_boundary_particles(x_init[k], y_init[k], 
                                        x_offt_aver, y_offt_aver, 
                                        px, py, pz, boundary)

        x_h = x_halfstep / grid_step_size + .5
        y_h = y_halfstep / grid_step_size + .5
        x_loc = x_h - floor(x_h) - .5
        y_loc = y_h - floor(y_h) - .5
        ix = int(floor(x_h) + grid_steps // 2)
        iy = int(floor(y_h) + grid_steps // 2)

        Ex, Ey, Ez, Bx, By, Bz = 0, 0, 0, 0, 0, 0
        for kx in range(-2, 3):
            wx = weight4(x_loc, kx)
            for ky in range(-2, 3):
                w = wx * weight4(y_loc, ky)
                idx_x, idx_y = ix + kx, iy + ky

                # Collect value from a cell and 8 surrounding cells.
                Ex += Ex_avg[idx_x, idx_y] * w; Bx += Bx_avg[idx_x, idx_y] * w
                Ey += Ey_avg[idx_x, idx_y] * w; By += By_avg[idx_x, idx_y] * w
                Ez += Ez_avg[idx_x, idx_y] * w; Bz += Bz_avg[idx_x, idx_y] * w

        # Move the particles according the the fields
        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)
        vx, vy, vz = px / gamma_m, py / gamma_m, pz / gamma_m
        factor_1 = q * xi_step_size / (1 - pz / gamma_m)
        dpx = factor_1 * (Ex + vy * Bz - vz * By)
        dpy = factor_1 * (Ey - vx * Bz + vz * Bx)
        dpz = factor_1 * (Ez + vx * By - vy * Bx)
        px, py, pz = opx + dpx / 2, opy + dpy / 2, opz + dpz / 2

        # Move the particles according the the fields again using updated momenta
        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)
        vx, vy, vz = px / gamma_m, py / gamma_m, pz / gamma_m
        factor_1 = q * xi_step_size / (1 - pz / gamma_m)
        dpx = factor_1 * (Ex + vy * Bz - vz * By)
        dpy = factor_1 * (Ey - vx * Bz + vz * Bx)
        dpz = factor_1 * (Ez + vx * By - vy * Bx)
        px, py, pz = opx + dpx / 2, opy + dpy / 2, opz + dpz / 2

        # Apply the coordinate and momenta increments.
        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)

        x_offt += px / (gamma_m - pz) * xi_step_size  # no mixing with x_init
        y_offt += py / (gamma_m - pz) * xi_step_size  # no mixing with y_init

        px, py, pz = opx + dpx, opy + dpy, opz + dpz
        
        _, _, x_offt, y_offt, px, py, pz = \
            _correct_boundary_particles(x_init[k], y_init[k], x_offt, y_offt, 
                                        px, py, pz, boundary)

        # Protection from non-physical particles.
        x = x_init[k] + x_offt
        y = y_init[k] + y_offt
        if x >= +boundary or x <= -boundary or y >= +boundary or y <= -boundary:
            x_offt = y_offt = px = py = pz = 0
            qs[k] = 0

        # Save the results into the output arrays  # TODO: get rid of that
        x_offt_full[k], y_offt_full[k] = x_offt, y_offt
        px_full[k], py_full[k], pz_full[k] = px, py, pz


@nb.njit(parallel=True)
def move_estimate_wo_fields_numba(xi_step, m, x_offt, y_offt, px, py, pz, size):
    """
    Move coarse plasma particles as if there were no fields.
    """
    m = m.ravel()
    px, py, pz = px.ravel(), py.ravel(), pz.ravel()
    x_offt, y_offt = x_offt.ravel(), y_offt.ravel()
    for k in nb.prange(size):
        if m[k] == 0:
            continue
        gamma_m = np.sqrt(m[k]**2 + pz[k]**2 + px[k]**2 + py[k]**2)

        x_offt[k] += px[k] / (gamma_m - pz[k]) * xi_step
        y_offt[k] += py[k] / (gamma_m - pz[k]) * xi_step

    
# TODO: Not very smart to get a kernel this way.
def get_move_smart_kernel_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, float64 boundary,
        float64 grid_step_size, float64 grid_steps,
        raw T Ex_avg, raw T Ey_avg, raw T Ez_avg,
        raw T Bx_avg, raw T By_avg, raw T Bz_avg,
        raw T x_init, raw T y_init,
        raw T x_offt_prev, raw T y_offt_prev,
        raw T px_prev, raw T py_prev, raw T pz_prev,
        raw T m
        """,
        out_params="""
        raw T q, raw T x_offt_full, raw T y_offt_full,
        raw T px_full, raw T py_full, raw T pz_full
        """,
        operation="""
        if (q[i] == 0){
            continue;
        }
        T px = px_prev[i], py = py_prev[i], pz = pz_prev[i];
        T x_halfstep = x_init[i] + (x_offt_prev[i] + x_offt_full[i]) / 2;
        T y_halfstep = y_init[i] + (y_offt_prev[i] + y_offt_full[i]) / 2;
        if (x_halfstep > boundary) {
            x_halfstep =  2 * boundary  - x_halfstep;
            px = py = pz = 0;
        }
        if (x_halfstep < -boundary) {
            x_halfstep = -2 * boundary - x_halfstep;
            px = py = pz = 0;
        }
        if (y_halfstep > boundary) {
            y_halfstep = 2 * boundary  - y_halfstep;
            px = py = pz = 0;
        }
        if (y_halfstep < -boundary) {
            y_halfstep = -2 * boundary - y_halfstep;
            px = py = pz = 0;
        }
        
        const T x_h = x_halfstep / (T) grid_step_size + 0.5;
        const T y_h = y_halfstep / (T) grid_step_size + 0.5;
        const T x_loc = x_h - floor(x_h) - 0.5;
        const T y_loc = y_h - floor(y_h) - 0.5;
        const int ix = floor(x_h) + floor(grid_steps / 2);
        const int iy = floor(y_h) + floor(grid_steps / 2);

        T Ex = 0, Ey = 0, Ez = 0, Bx = 0, By = 0, Bz = 0;
        for (int kx = -2; kx <= 2; kx++) {
            const double wx = weight4(x_loc, kx);
            for (int ky = -2; ky <= 2; ky++) {
                const double w = wx * weight4(y_loc, ky);
                const int idx = (iy + ky) + (int) grid_steps * (ix + kx);

                Ex += Ex_avg[idx] * w; Bx += Bx_avg[idx] * w;
                Ey += Ey_avg[idx] * w; By += By_avg[idx] * w;
                Ez += Ez_avg[idx] * w; Bz += Bz_avg[idx] * w;
            }
        }

        const T opx = px_prev[i], opy = py_prev[i], opz = pz_prev[i];
        T x_offt = x_offt_prev[i], y_offt = y_offt_prev[i];

        T gamma_m = sqrt(m[i]*m[i] + px*px + py*py + pz*pz);
        T vx = px / gamma_m, vy = py / gamma_m, vz = pz / gamma_m;
        T factor = q[i] * (T) xi_step_size / (1 - pz / gamma_m);
        T dpx = factor * (Ex + vy * Bz - vz * By);
        T dpy = factor * (Ey - vx * Bz + vz * Bx);
        T dpz = factor * (Ez + vx * By - vy * Bx);
        px = opx + dpx / 2; py = opy + dpy / 2; pz = opz + dpz / 2;

        gamma_m = sqrt(m[i]*m[i] + px*px + py*py + pz*pz);
        vx = px / gamma_m, vy = py / gamma_m, vz = pz / gamma_m;
        factor = q[i] * (T) xi_step_size / (1 - pz / gamma_m);
        dpx = factor * (Ex + vy * Bz - vz * By);
        dpy = factor * (Ey - vx * Bz + vz * Bx);
        dpz = factor * (Ez + vx * By - vy * Bx);
        px = opx + dpx / 2; py = opy + dpy / 2; pz = opz + dpz / 2;

        gamma_m = sqrt(m[i]*m[i] + px*px + py*py + pz*pz);
        x_offt += px / (gamma_m - pz) * xi_step_size;
        y_offt += py / (gamma_m - pz) * xi_step_size;
        px = opx + dpx; py = opy + dpy; pz = opz + dpz;

        T x = x_init[i] + x_offt, y = y_init[i] + y_offt;
        if (x > boundary) {
            x =  2 * boundary  - x;
            x_offt = x - x_init[i];
            px = py = pz = 0;
        }
        if (x < -boundary) {
            x = -2 * boundary - x;
            x_offt = x - x_init[i];
            px = py = pz = 0;
        }
        if (y > boundary) {
            y = 2 * boundary  - y;
            y_offt = y - y_init[i];
            px = py = pz = 0;
        }
        if (y < -boundary) {
            y = -2 * boundary - y;
            y_offt = y - y_init[i];
            px = py = pz = 0;
        }
        
        if (x >= +boundary || x <= -boundary 
            || y >= +boundary || y <= -boundary) {
            x_offt = y_offt = 0;
            px = py = pz = 0;
            q[i] = 0; 
        }

        x_offt_full[i] = x_offt; y_offt_full[i] = y_offt;
        px_full[i] = px; py_full[i] = py; pz_full[i] = pz;

        """,
        name='move_smart_cupy', preamble=weight4_cupy, no_return=True
    )


def get_move_wo_fields_kernel_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, raw T m
        """,
        out_params="""
        raw T x_offt, raw T y_offt, raw T px, raw T py, raw T pz
        """,
        operation="""
        if (m[i] == 0){
            continue;
        }
        const T gamma_m = sqrt(
            m[i]*m[i] + px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i]);

        x_offt[i] += px[i] / (gamma_m - pz[i]) * xi_step_size;
        y_offt[i] += py[i] / (gamma_m - pz[i]) * xi_step_size;
        """,
        name='move_wo_fields_cupy', no_return=True
    )


def get_plasma_particles_mover(config: Config):
    xi_step_size     = config.getfloat('xi-step')
    grid_step_size   = config.getfloat('transverse-step')
    grid_steps       = config.getint('window-width-steps')
    bound_padding_steps = config.getint('bound-padding-steps')
    boundary = grid_step_size * (grid_steps / 2 - bound_padding_steps)

    pu_type = config.get('processing-unit-type').lower()
    if pu_type == 'cpu':
        move_wo_fields_kernel = move_estimate_wo_fields_numba
        move_smart_kernel = move_smart_kernel_numba
    elif pu_type == 'gpu':
        move_wo_fields_kernel = get_move_wo_fields_kernel_cupy()
        move_smart_kernel = get_move_smart_kernel_cupy()

    def move_particles_wo_fields(particles: dict, const_arrays: Arrays):
        """
        Move coarse plasma particles as if there were no fields.
        """
        # NOTE: We need to copy particles to discriminate
        #       particles_full and particles_prev.
        particles_full = {}
        for sort in const_arrays.sorts:
            particles_full[sort] = particles[sort].copy()

        for sort in const_arrays.sorts:
            move_wo_fields_kernel(
                xi_step_size, particles_full[sort].m,
                particles_full[sort].x_offt, particles_full[sort].y_offt,
                particles_full[sort].px, particles_full[sort].py, 
                particles_full[sort].pz, 
                size = particles_full[sort].m.size)

        return particles_full

    def move_particles_smart(
        fields: Arrays, particles_prev: dict, particles_full: dict,
        const_arrays: Arrays):
        """
        Update plasma particle coordinates and momenta according to the
        field values interpolated halfway between the previous plasma
        particle location and the the best estimation of its next location
        currently available to us.
        """
        for sort in const_arrays.sorts:
            move_smart_kernel(
                xi_step_size, boundary, grid_step_size, grid_steps,
                
                fields.Ex, fields.Ey, fields.Ez,
                fields.Bx, fields.By, fields.Bz,

                particles_prev[sort].x_init, particles_prev[sort].y_init,

                particles_prev[sort].x_offt, particles_prev[sort].y_offt,
                particles_prev[sort].px, particles_prev[sort].py, 
                particles_prev[sort].pz,

                
                particles_full[sort].m, particles_full[sort].q, 
                particles_full[sort].x_offt, particles_full[sort].y_offt,
                particles_full[sort].px, particles_full[sort].py, 
                particles_full[sort].pz,

                size=particles_prev[sort].m.size)

        return particles_full

    return move_particles_wo_fields, move_particles_smart
