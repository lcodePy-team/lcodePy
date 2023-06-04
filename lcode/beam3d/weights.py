"""Module for weights calculation and deposition routines."""
import numpy as np
import numba as nb
from math import floor

from ..config.config import Config


@nb.njit
def weight1(x, place):
    if place == 0:
        return x
    elif place == 1:
        return 1 - x


@nb.njit
def weight4(x, place):
    """
    Calculate the indices of a cell corresponding to the coordinates,
    and the coefficients of interpolation and deposition for this cell
    and 24 surrounding cells.
    The weights correspond to ...
    """
    # TODO: Change to switch statement (match and case) when Python 3.10 is
    #       supported by Anaconda.
    if place == -2:
        return (1 / 2 - x) ** 4 / 24
    elif place == -1:
        return 19/96 - 11/24 * x + x ** 2 / 4 + x ** 3 / 6 - x ** 4 / 6
    elif place == 0:
        return 115/192 - x ** 2 * 5/8 + x ** 4 / 4
    elif place == 1:
        return 19/96 + 11/24 * x + x ** 2 / 4 - x ** 3 / 6 - x ** 4 / 6
    elif place == 2:
        return (1 / 2 + x) ** 4 / 24


# Deposition and field interpolation #
@nb.njit(parallel=True)
def deposit_beam_numba(grid_steps, x_h, y_h, xi_loc, q_norm,
                       out_ro0, out_ro1, size):
    """
    Deposit beam particles onto the charge density grids.
    """
    x_h, y_h = x_h.ravel(), y_h.ravel()
    xi_loc, q_norm = xi_loc.ravel(), q_norm.ravel()
    num_threads = nb.get_num_threads()
    out_ro0_ = np.zeros(shape = (num_threads, grid_steps, grid_steps))
    out_ro1_ = np.zeros(shape = (num_threads, grid_steps, grid_steps))

    for k in nb.prange(size):
        thread = nb.get_thread_id()
        x_loc = x_h[k] - floor(x_h[k]) - .5
        y_loc = y_h[k] - floor(y_h[k]) - .5
        ix = int(floor(x_h[k]) + grid_steps // 2)
        iy = int(floor(y_h[k]) + grid_steps // 2)

        for kx in range(-2, 3):
            wx = weight4(x_loc, kx)
            for ky in range(-2, 3):
                w = wx * weight4(y_loc, ky)
                w0 = w * weight1(xi_loc[k], 0)
                w1 = w * weight1(xi_loc[k], 1)
                index_x, index_y = ix + kx, iy + ky

                out_ro0_[thread, index_x, index_y] += q_norm[k] * w0
                out_ro1_[thread, index_x, index_y] += q_norm[k] * w1
    
    for index_x in nb.prange(out_ro0.shape[0]): 
        for index_y in range(out_ro0.shape[1]): 
            c1 = c2 = 0
            for i in range(num_threads):
                c1 += out_ro0_[i, index_x, index_y] 
                c2 += out_ro1_[i, index_x, index_y] 
            out_ro0[index_x, index_y] += c1
            out_ro1[index_x, index_y] += c2


# Calculates the weights for a particle, for GPU #

weight1_cupy = """
__device__ inline T weight1(T x, int place) {
    if (place == 0)
        return x;
    else if (place == 1)
        return (1 - x);
    else
        return 0.;
}
"""


weight4_cupy = """
__device__ inline T weight4(T x, int place) {
    if (place == -2)
        return (1 / 2. - x) * (1 / 2. - x) * (1 / 2. - x) * (1 / 2. - x) / 24.;
    else if (place == -1)
        return (
            19 / 96. - 11 / 24. * x + x * x / 4. + x * x * x / 6. -
            x * x * x * x / 6.
        );
    else if (place == 0)
        return 115 / 192. - 5 / 8. * x * x + x * x * x * x / 4.;
    else if (place == 1)
        return (
            19 / 96. + 11 / 24. * x + x * x / 4. - x * x * x / 6. -
            x * x * x * x / 6.
        );
    else if (place == 2)
        return (1 / 2. + x) * (1 / 2. + x) * (1 / 2. + x) * (1 / 2. + x) / 24.;
    else
        return 0.;
}
"""

# A function that performs gridding of beam particles onto
# the charge density grid, for GPU #

def get_deposit_beam_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 grid_steps, raw T x_h, raw T y_h, raw T xi_loc,
        raw T q
        """,
        out_params='raw T out_ro0, raw T out_ro1',
        operation="""
        const T x_loc = x_h[i] - floor(x_h[i]) - 0.5;
        const T y_loc = y_h[i] - floor(y_h[i]) - 0.5;
        const int ix = floor(x_h[i]) + floor(grid_steps / 2);
        const int iy = floor(y_h[i]) + floor(grid_steps / 2);

        for (int kx = -2; kx <= 2; kx++) {
            const T wx = weight4(x_loc, kx);
            for (int ky = -2; ky <= 2; ky++) {
                const T w  = wx * weight4(y_loc,  ky);
                const T w0 = w  * weight1(xi_loc[i], 0);
                const T w1 = w  * weight1(xi_loc[i], 1);
                const int idx = (iy + ky) + (int) grid_steps * (ix + kx);

                atomicAdd(&out_ro0[idx], q[i] * w0);
                atomicAdd(&out_ro1[idx], q[i] * w1);
            }
        }
        """,
        name='deposit_beam_cupy', preamble=weight1_cupy+weight4_cupy,
        no_return=True
    )


def get_deposit_beam(config: Config):
    xi_step_size = config.getfloat('xi-step')
    grid_step_size = config.getfloat('window-width-step-size')
    grid_steps = config.getint('window-width-steps')

    pu_type = config.get('processing-unit-type').lower()
    if pu_type == 'cpu':
        deposit_beam_kernel = deposit_beam_numba
    elif pu_type == 'gpu':
        deposit_beam_kernel = get_deposit_beam_cupy()

    def deposit_beam(plasma_layer_idx, x, y, xi, q_norm,
                     out_ro0, out_ro1):
        """
        Deposit beam particles onto the charge density grid.
        """
        xi_plasma_layer = - xi_step_size * plasma_layer_idx
        xi_loc = (xi_plasma_layer - xi) / xi_step_size
        x_h, y_h = x / grid_step_size + 0.5, y / grid_step_size + 0.5

        deposit_beam_kernel(
            grid_steps, x_h, y_h, xi_loc, q_norm, out_ro0, out_ro1,
            size=q_norm.size)

    return deposit_beam
