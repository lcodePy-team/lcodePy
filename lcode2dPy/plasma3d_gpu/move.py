"""Module for calculation of particles movement and interpolation of fields on particles positions."""
import numba as numba
import numba.cuda

import cupy as cp

from math import sqrt

from lcode2dPy.config.config import Config
from lcode2dPy.plasma3d_gpu.weights import weights
from lcode2dPy.plasma3d_gpu.data import GPUArrays

WARP_SIZE = 32


# Field interpolation and particle movement (fused) #

@numba.njit
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


@numba.cuda.jit
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
    # Do nothing if our thread does not have a particle to move.
    k = numba.cuda.grid(1)
    if k >= ms.size:
        return

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
    
    x_offt_new = cp.zeros_like(x_prev_offt)
    y_offt_new = cp.zeros_like(y_prev_offt)
    px_new = cp.zeros_like(px_prev)
    py_new = cp.zeros_like(py_prev)
    pz_new = cp.zeros_like(pz_prev)

    cfg = int(cp.ceil(x_init.size / WARP_SIZE)), WARP_SIZE
    move_smart_kernel[cfg](xi_step, reflect_boundary,
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

    return GPUArrays(x_init=x_init, y_init=y_init,
                     x_offt=x_offt_new, y_offt=y_offt_new,
                     px=px_new, py=py_new, pz=pz_new, q=q, m=m)
    # I don't like how it looks. TODO: write a new method for Particles class
    # or somehow use Particles.copy().


def move_estimate_wo_fields(xi_step, reflect_boundary, particles):
    """
    Move coarse plasma particles as if there were no fields.
    Also reflect the particles from `+-reflect_boundary`.
    """
    m, q = particles.m, particles.q
    x_init, y_init = particles.x_init, particles.y_init
    prev_x_offt, prev_y_offt = particles.x_offt, particles.y_offt
    px, py, pz = particles.px, particles.py, particles.pz

    x, y = x_init + prev_x_offt, y_init + prev_y_offt
    gamma_m = cp.sqrt(m**2 + pz**2 + px**2 + py**2)

    x += px / (gamma_m - pz) * xi_step
    y += py / (gamma_m - pz) * xi_step

    reflect = reflect_boundary
    x[x >= +reflect] = +2 * reflect - x[x >= +reflect]
    x[x <= -reflect] = -2 * reflect - x[x <= -reflect]
    y[y >= +reflect] = +2 * reflect - y[y >= +reflect]
    y[y <= -reflect] = -2 * reflect - y[y <= -reflect]

    x_offt, y_offt = x - x_init, y - y_init

    return GPUArrays(x_init=x_init, y_init=y_init,
                     x_offt=x_offt, y_offt=y_offt,
                     px=px, py=py, pz=pz, q=q, m=m)


class ParticleMover:
    def __init__(self, config: Config):
        self.xi_step          = config.getfloat('xi-step')
        reflect_padding_steps = config.getint('reflect-padding-steps')
        self.grid_step_size   = config.getfloat('window-width-step-size')
        self.grid_steps       = config.getint('window-width-steps')
        self.reflect_boundary = self.grid_step_size * (
                                self.grid_steps / 2 - reflect_padding_steps)

    def move_particles(self, fields, particles, estimated_particles):
        return move_smart(self.xi_step, self.reflect_boundary,
                          self.grid_step_size, self.grid_steps,
                          particles, estimated_particles, fields)
    
    def move_particles_wo_fields(self, particles):
        return move_estimate_wo_fields(self.xi_step, self.reflect_boundary,
                                       particles)