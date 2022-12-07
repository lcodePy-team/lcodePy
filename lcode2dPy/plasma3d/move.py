"""Module for calculation of particles movement and interpolation of fields on particles positions."""
import numba as nb
import numpy as np

from math import floor, sin, sqrt, pi

from ..config.config import Config
from .weights import weights, weight4_cupy
from .data import Arrays

# Field interpolation and particle movement (fused), for CPU #

@nb.njit
def interp25(a, i, j,
             w2M2P, w1M2P, w02P, w1P2P, w2P2P,
             w2M1P, w1M1P, w01P, w1P1P, w2P1P,
             w2M0,  w1M0,  w00,  w1P0,  w2P0,
             w2M1M, w1M1M, w01M, w1P1M, w2P1M,
             w2M2M, w1M2M, w02M, w1P2M, w2P2M):
    """
    Collect value from a cell and 8 surrounding cells (using `weights` output).
    """
    return (
        a[i - 2, j + 2] * w2M2P + a[i - 1, j + 2] * w1M2P +
        a[i + 0, j + 2] * w02P  + a[i + 1, j + 2] * w1P2P +
        a[i + 2, j + 2] * w2P2P + a[i - 2, j + 1] * w2M1P +
        a[i - 1, j + 1] * w1M1P + a[i + 0, j + 1] * w01P  +
        a[i + 1, j + 1] * w1P1P + a[i + 2, j + 1] * w2P1P +
        a[i - 2, j + 0] * w2M0  + a[i - 1, j + 0] * w1M0  +
        a[i + 0, j + 0] * w00   + a[i + 1, j + 0] * w1P0  +
        a[i + 2, j + 0] * w2P0  + a[i - 2, j - 1] * w2M1M +
        a[i - 1, j - 1] * w1M1M + a[i + 0, j - 1] * w01M  +
        a[i + 1, j - 1] * w1P1M + a[i + 2, j - 1] * w2P1M +
        a[i - 2, j - 2] * w2M2M + a[i - 1, j - 2] * w1M2M +
        a[i + 0, j - 2] * w02M  + a[i + 1, j - 2] * w1P2M +
        a[i + 2, j - 2] * w2P2M
    )


@nb.njit
def noise_reductor(ro, i, j, Ex, Ey, x_half, y_half, grid,
                   grid_step_size, noise_reductor_ampl):
    # Density-based noise reductor
    Ax = (+ (ro[i + 1, j - 1] + ro[i - 1, j - 1] - 2 * ro[i, j - 1]) / 4
          + (ro[i + 1, j + 0] + ro[i - 1, j + 0] - 2 * ro[i, j + 0]) / 4
          + (ro[i + 1, j + 1] + ro[i - 1, j + 1] - 2 * ro[i, j + 1]) / 4) / 3
    x_i = grid[i]
    dEx = Ax * grid_step_size * sin(pi * (x_half - x_i) / grid_step_size) / pi
    Ex_new = Ex + dEx * noise_reductor_ampl

    Ay = (+ (ro[i - 1, j + 1] + ro[i - 1, j - 1] - 2 * ro[i - 1, j]) / 4
          + (ro[i + 0, j + 1] + ro[i + 0, j - 1] - 2 * ro[i + 0, j]) / 4
          + (ro[i + 1, j + 1] + ro[i + 1, j - 1] - 2 * ro[i + 1, j]) / 4) / 3
    y_j = grid[j]
    dEy = Ay * grid_step_size * sin(pi * (y_half - y_j) / grid_step_size) / pi
    Ey_new = Ey + dEy * noise_reductor_ampl

    return Ex_new, Ey_new


@nb.njit(parallel=True)
def move_smart_kernel(xi_step_size, reflect_boundary,
                      grid_step_size, grid_steps,
                      ms, qs,
                      x_init, y_init,
                      prev_x_offt, prev_y_offt,
                      estimated_x_offt, estimated_y_offt,
                      prev_px, prev_py, prev_pz,
                      Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg,
                      new_x_offt, new_y_offt, new_px, new_py, new_pz,
                      grid, ro_noisy, noise_reductor_ampl):
    """
    Update plasma particle coordinates and momenta according to the field
    values interpolated halfway between the previous plasma particle location
    and the the best estimation of its next location currently available to us.
    Also reflect the particles from `+-reflect_boundary`.
    """
    for k in nb.prange(ms.size):
        m, q = ms[k], qs[k]

        opx, opy, opz = prev_px[k], prev_py[k], prev_pz[k]
        px, py, pz = opx, opy, opz
        x_offt, y_offt = prev_x_offt[k], prev_y_offt[k]

        # Calculate midstep positions and fields in them.
        x_halfstep = x_init[k] + (prev_x_offt[k] + estimated_x_offt[k]) / 2
        y_halfstep = y_init[k] + (prev_y_offt[k] + estimated_y_offt[k]) / 2

        (i, j, 
         w2M2P, w1M2P, w02P, w1P2P, w2P2P,
         w2M1P, w1M1P, w01P, w1P1P, w2P1P,
         w2M0,  w1M0,  w00,  w1P0,  w2P0,
         w2M1M, w1M1M, w01M, w1P1M, w2P1M,
         w2M2M, w1M2M, w02M, w1P2M, w2P2M
        ) = weights(
            x_halfstep, y_halfstep, grid_steps, grid_step_size
        )

        Ex = interp25(Ex_avg, i, j,
                      w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                      w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                      w2M0,  w1M0,  w00,  w1P0,  w2P0,
                      w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                      w2M2M, w1M2M, w02M, w1P2M, w2P2M)
        Ey = interp25(Ey_avg, i, j,
                      w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                      w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                      w2M0,  w1M0,  w00,  w1P0,  w2P0,
                      w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                      w2M2M, w1M2M, w02M, w1P2M, w2P2M)
        Ez = interp25(Ez_avg, i, j,
                      w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                      w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                      w2M0,  w1M0,  w00,  w1P0,  w2P0,
                      w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                      w2M2M, w1M2M, w02M, w1P2M, w2P2M)
        Bx = interp25(Bx_avg, i, j,
                      w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                      w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                      w2M0,  w1M0,  w00,  w1P0,  w2P0,
                      w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                      w2M2M, w1M2M, w02M, w1P2M, w2P2M)
        By = interp25(By_avg, i, j,
                      w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                      w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                      w2M0,  w1M0,  w00,  w1P0,  w2P0,
                      w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                      w2M2M, w1M2M, w02M, w1P2M, w2P2M)
        Bz = interp25(Bz_avg, i, j,
                      w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                      w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                      w2M0,  w1M0,  w00,  w1P0,  w2P0,
                      w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                      w2M2M, w1M2M, w02M, w1P2M, w2P2M)

        Ex, Ey = noise_reductor(ro_noisy, i, j, Ex, Ey, x_halfstep, y_halfstep,
                                grid, grid_step_size, noise_reductor_ampl)

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

        # Apply the coordinate and momenta increments
        gamma_m = sqrt(m**2 + pz**2 + px**2 + py**2)

        x_offt += px / (gamma_m - pz) * xi_step_size  # no mixing with x_init
        y_offt += py / (gamma_m - pz) * xi_step_size  # no mixing with y_init

        px, py, pz = opx + dpx, opy + dpy, opz + dpz

        # Reflect the particles from `+-reflect_boundary`.
        # TODO: avoid branching?
        x = x_init[k] + x_offt
        y = y_init[k] + y_offt
        if x > +reflect_boundary:
            x = +2 * reflect_boundary - x
            x_offt = x - x_init[k]
            px = -px
        if x < -reflect_boundary:
            x = -2 * reflect_boundary - x
            x_offt = x - x_init[k]
            px = -px
        if y > +reflect_boundary:
            y = +2 * reflect_boundary - y
            y_offt = y - y_init[k]
            py = -py
        if y < -reflect_boundary:
            y = -2 * reflect_boundary - y
            y_offt = y - y_init[k]
            py = -py

        # Save the results into the output arrays  # TODO: get rid of that
        new_x_offt[k], new_y_offt[k] = x_offt, y_offt
        new_px[k], new_py[k], new_pz[k] = px, py, pz


def move_smart(xi_step, reflect_boundary, grid_step_size, grid_steps,
               particles: Arrays, estimated_particles: Arrays,
               fields: Arrays, const_arrays: Arrays, ro_noisy: np.ndarray,
               noise_reductor_ampl):
    """
    Update plasma particle coordinates and momenta according to the field
    values interpolated halfway between the previous plasma particle location
    and the the best estimation of its next location currently available to us.
    This is a convenience wrapper around the `move_smart_kernel` CUDA kernel.
    """

    m = particles.m
    q = particles.q

    x_init = particles.x_init
    y_init = particles.y_init

    x_prev_offt = particles.x_offt
    y_prev_offt = particles.y_offt

    px_prev = particles.px
    py_prev = particles.py
    pz_prev = particles.pz

    estimated_x_offt = estimated_particles.x_offt
    estimated_y_offt = estimated_particles.y_offt
    
    x_offt_new = np.zeros_like(x_prev_offt)
    y_offt_new = np.zeros_like(y_prev_offt)
    px_new = np.zeros_like(px_prev)
    py_new = np.zeros_like(py_prev)
    pz_new = np.zeros_like(pz_prev)

    move_smart_kernel(xi_step, reflect_boundary,
                      grid_step_size, grid_steps,
                      m.ravel(), q.ravel(),
                      x_init.ravel(), y_init.ravel(),
                      x_prev_offt.ravel(), y_prev_offt.ravel(),
                      estimated_x_offt.ravel(), estimated_y_offt.ravel(),
                      px_prev.ravel(), py_prev.ravel(), pz_prev.ravel(),
                      fields.Ex, fields.Ey, fields.Ez,
                      fields.Bx, fields.By, fields.Bz,
                      x_offt_new.ravel(), y_offt_new.ravel(),
                      px_new.ravel(), py_new.ravel(), pz_new.ravel(),
                      const_arrays.grid, ro_noisy, noise_reductor_ampl)

    return Arrays(fields.xp, x_init=x_init, y_init=y_init,
                  x_offt=x_offt_new, y_offt=y_offt_new,
                  px=px_new, py=py_new, pz=pz_new, q=q, m=m)
    # I don't like how it looks. TODO: write a new method for Particles class
    # or somehow use Particles.copy().


def move_estimate_wo_fields(xi_step, reflect_boundary, particles: Arrays):
    """
    Move coarse plasma particles as if there were no fields.
    Also reflect the particles from `+-reflect_boundary`.
    """
    m, q = particles.m, particles.q
    x_init, y_init = particles.x_init, particles.y_init
    prev_x_offt, prev_y_offt = particles.x_offt, particles.y_offt
    px, py, pz = particles.px, particles.py, particles.pz

    x, y = x_init + prev_x_offt, y_init + prev_y_offt
    gamma_m = np.sqrt(m**2 + pz**2 + px**2 + py**2)

    x += px / (gamma_m - pz) * xi_step
    y += py / (gamma_m - pz) * xi_step

    reflect = reflect_boundary
    x[x >= +reflect] = +2 * reflect - x[x >= +reflect]
    x[x <= -reflect] = -2 * reflect - x[x <= -reflect]
    y[y >= +reflect] = +2 * reflect - y[y >= +reflect]
    y[y <= -reflect] = -2 * reflect - y[y <= -reflect]

    x_offt, y_offt = x - x_init, y - y_init

    return Arrays(particles.xp, x_init=x_init, y_init=y_init,
                  x_offt=x_offt, y_offt=y_offt,
                  px=px, py=py, pz=pz, q=q, m=m)


# Field interpolation and particle movement (fused), for GPU #

# TODO: Not very smart to get a kernel this way.
def get_move_smart_kernel_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, float64 reflect_boundary,
        float64 grid_step_size, float64 grid_steps,
        raw T m, raw T q, raw T x_init, raw T y_init,
        raw T prev_x_offt, raw T prev_y_offt,
        raw T estim_x_offt, raw T estim_y_offt,
        raw T prev_px, raw T prev_py, raw T prev_pz,
        raw T Ex_avg, raw T Ey_avg, raw T Ez_avg,
        raw T Bx_avg, raw T By_avg, raw T Bz_avg
        """,
        out_params="""
        raw T out_x_offt, raw T out_y_offt,
        raw T out_px, raw T out_py, raw T out_pz
        """,
        operation="""
        const T x_halfstep = x_init[i] + (prev_x_offt[i] + estim_x_offt[i]) / 2;
        const T y_halfstep = y_init[i] + (prev_y_offt[i] + estim_y_offt[i]) / 2;
        
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

        T px = prev_px[i], py = prev_py[i], pz = prev_pz[i];
        const T opx = prev_px[i], opy = prev_py[i], opz = prev_pz[i];
        T x_offt = prev_x_offt[i], y_offt = prev_y_offt[i];

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
        if (x > reflect_boundary) {
            x =  2 * reflect_boundary  - x;
            x_offt = x - x_init[i];
            px = -px;
        }
        if (x < -reflect_boundary) {
            x = -2 * reflect_boundary - x;
            x_offt = x - x_init[i];
            px = -px;
        }
        if (y > reflect_boundary) {
            y = 2 * reflect_boundary  - y;
            y_offt = y - y_init[i];
            py = -py;
        }
        if (y < -reflect_boundary) {
            y = -2 * reflect_boundary - y;
            y_offt = y - y_init[i];
            py = -py;
        }

        out_x_offt[i] = x_offt; out_y_offt[i] = y_offt;
        out_px[i] = px; out_py[i] = py; out_pz[i] = pz;

        """,
        name='move_smart_cupy', preamble=weight4_cupy
    )


def get_move_wo_fields_kernel_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, float64 reflect_boundary,
        raw T m, raw T q, raw T x_init, raw T y_init,
        raw T prev_x_offt, raw T prev_y_offt,
        raw T prev_px, raw T prev_py, raw T prev_pz
        """,
        out_params="""
        raw T out_x_offt, raw T out_y_offt,
        raw T out_px, raw T out_py, raw T out_pz
        """,
        operation="""
        T x_offt = prev_x_offt[i], y_offt = prev_y_offt[i];
        T px = prev_px[i], py = prev_py[i], pz = prev_pz[i];
        const T gamma_m = sqrt(m[i]*m[i] + px*px + py*py + pz*pz);

        x_offt += px / (gamma_m - pz) * xi_step_size;
        y_offt += py / (gamma_m - pz) * xi_step_size;

        T x = x_init[i] + x_offt, y = y_init[i] + y_offt;
        if (x > reflect_boundary) {
            x =  2 * reflect_boundary  - x;
            x_offt = x - x_init[i];
            px = -px;
        }
        if (x < -reflect_boundary) {
            x = -2 * reflect_boundary - x;
            x_offt = x - x_init[i];
            px = -px;
        }
        if (y > reflect_boundary) {
            y = 2 * reflect_boundary  - y;
            y_offt = y - y_init[i];
            py = -py;
        }
        if (y < -reflect_boundary) {
            y = -2 * reflect_boundary - y;
            y_offt = y - y_init[i];
            py = -py;
        }

        out_x_offt[i] = x_offt; out_y_offt[i] = y_offt;
        out_px[i] = px; out_py[i] = py; out_pz[i] = pz;

        """,
        name='move_wo_fields_cupy'
    )


def get_plasma_particles_mover(config: Config):
    xi_step_size     = config.getfloat('xi-step')
    grid_step_size   = config.getfloat('window-width-step-size')
    grid_steps       = config.getint('window-width-steps')
    reflect_padding_steps = config.getint('reflect-padding-steps')
    reflect_boundary = grid_step_size * (grid_steps / 2 - reflect_padding_steps)
    pu_type = config.get('processing-unit-type').lower()

    noise_reductor_ampl = config.getfloat('noise-reductor-amplitude')

    if pu_type == 'cpu':
        def move_particles_smart(fields, particles, estimated_particles,
                                 const_arrays, ro_noisy):
            return move_smart(
                xi_step_size, reflect_boundary, grid_step_size, grid_steps,
                particles, estimated_particles, fields, const_arrays, ro_noisy,
                noise_reductor_ampl)

        def move_particles_wo_fields(particles):
            return move_estimate_wo_fields(
                xi_step_size, reflect_boundary, particles)

    elif pu_type == 'gpu':
        move_smart_kernel = get_move_smart_kernel_cupy()

        def move_particles_smart(
            fields: Arrays, particles: Arrays, estimated_particles: Arrays,
            const_arrays, ro_noisy):
            """
            Update plasma particle coordinates and momenta according to the
            field values interpolated halfway between the previous plasma
            particle location and the the best estimation of its next location
            currently available to us.
            """
            xp = particles.xp

            x_offt_new = xp.zeros_like(particles.x_offt)
            y_offt_new = xp.zeros_like(particles.y_offt)
            px_new = xp.zeros_like(particles.px)
            py_new = xp.zeros_like(particles.py)
            pz_new = xp.zeros_like(particles.pz)

            x_offt_new, y_offt_new, px_new, py_new, pz_new = move_smart_kernel(
                xi_step_size, reflect_boundary, grid_step_size, grid_steps,
                particles.m, particles.q, particles.x_init, particles.y_init,
                particles.x_offt, particles.y_offt,
                estimated_particles.x_offt,
                estimated_particles.y_offt,
                particles.px, particles.py, particles.pz,
                fields.Ex, fields.Ey, fields.Ez,
                fields.Bx, fields.By, fields.Bz,
                x_offt_new, y_offt_new, px_new, py_new, pz_new,
                size=(particles.m).size
            )

            return Arrays(xp,
                          x_init=particles.x_init, y_init=particles.y_init,
                          x_offt=x_offt_new, y_offt=y_offt_new,
                          px=px_new, py=py_new, pz=pz_new,
                          q=particles.q, m=particles.m)

        move_wo_fields_kernel_cupy = get_move_wo_fields_kernel_cupy()

        def move_particles_wo_fields(particles: Arrays):
            """
            Move coarse plasma particles as if there were no fields.
            Also reflect the particles from `+-reflect_boundary`.
            """
            xp = particles.xp

            x_offt_new = xp.zeros_like(particles.x_offt)
            y_offt_new = xp.zeros_like(particles.y_offt)
            px_new = xp.zeros_like(particles.px)
            py_new = xp.zeros_like(particles.py)
            pz_new = xp.zeros_like(particles.pz)

            x_offt_new, y_offt_new, px_new, py_new, pz_new =\
                move_wo_fields_kernel_cupy(
                    xi_step_size, reflect_boundary, particles.m, particles.q,
                    particles.x_init, particles.y_init,
                    particles.x_offt, particles.y_offt,
                    particles.px, particles.py, particles.pz,
                    x_offt_new, y_offt_new, px_new, py_new, pz_new,
                    size=(particles.m).size)

            return Arrays(xp,
                          x_init=particles.x_init, y_init=particles.y_init,
                          x_offt=x_offt_new, y_offt=y_offt_new,
                          px=px_new, py=py_new, pz=pz_new,
                          q=particles.q, m=particles.m)

    return move_particles_smart, move_particles_wo_fields
