"""Module for calculation of particles movement and interpolation of fields on particles positions."""
import numba as nb
import numpy as np

from math import sqrt

from lcode2dPy.plasma3d.weights import weights
from lcode2dPy.plasma3d.data import Particles

# Field interpolation and particle movement (fused) #

@nb.njit(cache=True)
def interp9(a, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM):
    """
    Collect value from a cell and 8 surrounding cells (using `weights` output).
    """
    return (
        a[i - 1, j + 1] * wMP + a[i + 0, j + 1] * w0P + a[i + 1, j + 1] * wPP +
        a[i - 1, j + 0] * wM0 + a[i + 0, j + 0] * w00 + a[i + 1, j + 0] * wP0 +
        a[i - 1, j - 1] * wMM + a[i + 0, j - 1] * w0M + a[i + 1, j - 1] * wPM
    )


@nb.njit
def move_smart_kernel(xi_step_size, reflect_boundary,
                      grid_step_size, grid_steps,
                      ms, qs,
                      x_init, y_init,
                      prev_x_offt, prev_y_offt,
                      estimated_x_offt, estimated_y_offt,
                      prev_px, prev_py, prev_pz,
                      Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg,
                      new_x_offt, new_y_offt, new_px, new_py, new_pz):
    """
    Update plasma particle coordinates and momenta according to the field
    values interpolated halfway between the previous plasma particle location
    and the the best estimation of its next location currently available to us.
    Also reflect the particles from `+-reflect_boundary`.
    """
    for k in range(ms.size):
        m, q = ms[k], qs[k]

        opx, opy, opz = prev_px[k], prev_py[k], prev_pz[k]
        px, py, pz = opx, opy, opz
        x_offt, y_offt = prev_x_offt[k], prev_y_offt[k]

        # Calculate midstep positions and fields in them.
        x_halfstep = x_init[k] + (prev_x_offt[k] + estimated_x_offt[k]) / 2
        y_halfstep = y_init[k] + (prev_y_offt[k] + estimated_y_offt[k]) / 2
        i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM = weights(
            x_halfstep, y_halfstep, grid_steps, grid_step_size
        )
        Ex = interp9(Ex_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        Ey = interp9(Ey_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        Ez = interp9(Ez_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        Bx = interp9(Bx_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        By = interp9(By_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        Bz = interp9(Bz_avg, i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)

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
               particles, estimated_particles, fields):
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
                      px_new.ravel(), py_new.ravel(), pz_new.ravel())

    return Particles(x_init, y_init, x_offt_new, y_offt_new,
                     px_new, py_new, pz_new, q, m)
    # I don't like how it looks. TODO: write a new method for Particles class
    # or somehow use Particles.copy().


class ParticleMover:
    def __init__(self, config):
        self.xi_step          = config.getfloat('xi-step')
        self.reflect_boundary = config.getfloat('reflect-boundary')
        self.grid_step_size   = config.getfloat('window-xy-step-size')
        self.grid_steps       = config.getint('window-xy-steps')

    def move_particles(self, fields, particles, estimated_particles):
        return move_smart(self.xi_step, self.reflect_boundary,
                          self.grid_step_size, self.grid_steps,
                          particles, estimated_particles, fields)