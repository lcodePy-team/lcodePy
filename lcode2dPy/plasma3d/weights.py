"""Module for weights calculation, interpolation and deposition routines."""
import numba as nb
import numpy as np
from math import sqrt, floor

from ..config.config import Config
from .data import Arrays

# Deposition and interpolation functions, for CPU #

@nb.njit
def weight4(x, place):
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


@nb.njit
def weights(x_h, y_h, grid_steps):
    """
    Calculate the indices of a cell corresponding to the coordinates,
    and the coefficients of interpolation and deposition for this cell
    and 24 surrounding cells.
    The weights correspond to ...
    """
    i, j = int(floor(x_h) + grid_steps // 2), int(floor(y_h) + grid_steps // 2)
    x_loc, y_loc = x_h - floor(x_h) - .5, y_h - floor(y_h) - .5
    # centered to -.5 to 5, not 0 to 1, as formulas use offset from cell center
    # TODO: get rid of this deoffsetting/reoffsetting festival

    wx0 = 115/192 - x_loc**2 * 5/8 + x_loc**4 / 4
    wy0 = 115/192 - y_loc**2 * 5/8 + y_loc**4 / 4
    wx1P = 19/96 + 11/24 * x_loc + x_loc**2 / 4 - x_loc**3 / 6 - x_loc**4 / 6
    wy1P = 19/96 + 11/24 * y_loc + y_loc**2 / 4 - y_loc**3 / 6 - y_loc**4 / 6
    wx1M = 19/96 - 11/24 * x_loc + x_loc**2 / 4 + x_loc**3 / 6 - x_loc**4 / 6
    wy1M = 19/96 - 11/24 * y_loc + y_loc**2 / 4 + y_loc**3 / 6 - y_loc**4 / 6
    wx2P = (1 / 2 + x_loc)**4 / 24
    wy2P = (1 / 2 + y_loc)**4 / 24
    wx2M = (1 / 2 - x_loc)**4 / 24
    wy2M = (1 / 2 - y_loc)**4 / 24

    w2M2P, w1M2P, w02P, w1P2P, w2P2P = (wx2M * wy2P, wx1M * wy2P, wx0 * wy2P,
                                        wx1P * wy2P, wx2P * wy2P)
    w2M1P, w1M1P, w01P, w1P1P, w2P1P = (wx2M * wy1P, wx1M * wy1P, wx0 * wy1P,
                                        wx1P * wy1P, wx2P * wy1P)
    w2M0,  w1M0,  w00,  w1P0,  w2P0  = (wx2M * wy0 , wx1M * wy0 , wx0 * wy0 ,
                                        wx1P * wy0 , wx2P * wy0 ) 
    w2M1M, w1M1M, w01M, w1P1M, w2P1M = (wx2M * wy1M, wx1M * wy1M, wx0 * wy1M,
                                        wx1P * wy1M, wx2P * wy1M)
    w2M2M, w1M2M, w02M, w1P2M, w2P2M = (wx2M * wy2M, wx1M * wy2M, wx0 * wy2M,
                                        wx1P * wy2M, wx2P * wy2M)

    return (
        i, j,
        w2M2P, w1M2P, w02P, w1P2P, w2P2P,
        w2M1P, w1M1P, w01P, w1P1P, w2P1P,
        w2M0,  w1M0,  w00,  w1P0,  w2P0,
        w2M1M, w1M1M, w01M, w1P1M, w2P1M,
        w2M2M, w1M2M, w02M, w1P2M, w2P2M
    )


# Deposition #

@nb.njit(parallel=True)
def deposit_kernel(grid_steps, x_h, y_h, m, q, px, py, pz,
                   out_ro, out_jx, out_jy, out_jz):
    """
    Deposit plasma particles onto the charge density and current grids.
    """
    for k in nb.prange(m.size):    
        # Deposit the resulting fine particle on ro/j grids.
        gamma_m = sqrt(m[k]**2 + px[k]**2 + py[k]**2 + pz[k]**2)
        dro = q[k] / (1 - pz[k] / gamma_m)
        djx = px[k] * (dro / gamma_m)
        djy = py[k] * (dro / gamma_m)
        djz = pz[k] * (dro / gamma_m)

        x_loc = x_h[k] - floor(x_h[k]) - .5
        y_loc = y_h[k] - floor(y_h[k]) - .5
        ix = int(floor(x_h[k]) + grid_steps // 2)
        iy = int(floor(y_h[k]) + grid_steps // 2)

        for kx in range(-2, 3):
            wx = weight4(x_loc, kx)
            for ky in range(-2, 3):
                w = wx * weight4(y_loc, ky)
                index_x, index_y = ix + kx, iy + ky

                out_ro[index_x, index_y] += dro * w
                out_jx[index_x, index_y] += djx * w
                out_jy[index_x, index_y] += djy * w
                out_jz[index_x, index_y] += djz * w


# Helper function for dual plasma approach, for GPU #

def get_coarse_to_fine_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 virtplasma_smallness_factor,
        float64 fine_grid_size, float64 coarse_grid_size,
        raw T c_x_offt, raw T c_y_offt, raw T c_px, raw T c_py, raw T c_pz,
        raw T c_q, raw T c_m, raw T fine_grid,
        raw T influence_prev, raw T influence_next,
        raw int64 inds_prev, raw int64 inds_next
        """,
        out_params="""
        raw T x, raw T y, raw T px, raw T py, raw T pz, raw T q, raw T m""",
        operation="""
        const int fi = floor(i / fine_grid_size);
        const int fj = i % (int) fine_grid_size;

        const T wpp = influence_prev[fi] * influence_prev[fj];
        const T wpn = influence_prev[fi] * influence_next[fj];
        const T wnp = influence_next[fi] * influence_prev[fj];
        const T wnn = influence_next[fi] * influence_next[fj];

        const int ind1 = (int) coarse_grid_size * inds_prev[fi] + inds_prev[fj];
        const int ind2 = (int) coarse_grid_size * inds_prev[fi] + inds_next[fj];
        const int ind3 = (int) coarse_grid_size * inds_next[fi] + inds_prev[fj];
        const int ind4 = (int) coarse_grid_size * inds_next[fi] + inds_next[fj];
        
        x[i] = fine_grid[fi] + 
            (wpp * c_x_offt[ind1] + wpn * c_x_offt[ind2] +
             wnp * c_x_offt[ind3] + wnn * c_x_offt[ind4]);
        y[i] = fine_grid[fj] +
            (wpp * c_y_offt[ind1] + wpn * c_y_offt[ind2] +
             wnp * c_y_offt[ind3] + wnn * c_y_offt[ind4]);
        
        px[i] = virtplasma_smallness_factor *
            (wpp * c_px[ind1] + wpn * c_px[ind2] +
             wnp * c_px[ind3] + wnn * c_px[ind4]);
        py[i] = virtplasma_smallness_factor *
            (wpp * c_py[ind1] + wpn * c_py[ind2] +
             wnp * c_py[ind3] + wnn * c_py[ind4]);
        pz[i] = virtplasma_smallness_factor *
            (wpp * c_pz[ind1] + wpn * c_pz[ind2] +
             wnp * c_pz[ind3] + wnn * c_pz[ind4]);
        
        q[i] = virtplasma_smallness_factor *
            (wpp * c_q[ind1] + wpn * c_q[ind2] +
             wnp * c_q[ind3] + wnn * c_q[ind4]);
        m[i] = virtplasma_smallness_factor *
            (wpp * c_m[ind1] + wpn * c_m[ind2] +
             wnp * c_m[ind3] + wnn * c_m[ind4]);
        """,
        name='coarse_to_fine'
    )


# Deposition and interpolation helper function, for GPU #

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


def get_deposit_plasma_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 grid_steps, raw T x_h, raw T y_h,
        raw T px, raw T py, raw T pz, raw T q, raw T m
        """,
        out_params='raw T out_ro, raw T out_jx, raw T out_jy, raw T out_jz',
        operation="""
        const T gamma_m = sqrt(
            m[i]*m[i] + px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i]);
        const T dro = q[i] / (1 - pz[i] / gamma_m);
        const T djx = px[i] * (dro / gamma_m);
        const T djy = py[i] * (dro / gamma_m);
        const T djz = pz[i] * (dro / gamma_m);

        const T x_loc = x_h[i] - floor(x_h[i]) - 0.5;
        const T y_loc = y_h[i] - floor(y_h[i]) - 0.5;
        const int ix = floor(x_h[i]) + floor(grid_steps / 2);
        const int iy = floor(y_h[i]) + floor(grid_steps / 2);

        for (int kx = -2; kx <= 2; kx++) {
            const T wx = weight4(x_loc, kx);
            for (int ky = -2; ky <= 2; ky++) {
                const T w = wx * weight4(y_loc, ky);
                const int idx = (iy + ky) + (int) grid_steps * (ix + kx);

                atomicAdd(&out_ro[idx], dro * w);
                atomicAdd(&out_jx[idx], djx * w);
                atomicAdd(&out_jy[idx], djy * w);
                atomicAdd(&out_jz[idx], djz * w);
            }
        }
        """,
        name='deposit_plasma_cupy', preamble=weight4_cupy
    )


def get_deposit_plasma(config: Config):
    """
    Check if a user set dual-plasma-approach to True or False and return
    the deposition function for the set approach.
    """
    grid_step_size = config.getfloat('window-width-step-size')
    grid_steps = config.getint('window-width-steps')
    plasma_fineness = config.getint('plasma-fineness')
    dual_plasma_approach = config.getbool('dual-plasma-approach')

    pu_type = config.get('processing-unit-type').lower()
    if pu_type == 'cpu':
        def deposit_single(x_init, y_init, x_offt, y_offt,
                           px, py, pz, q, m, const_arrays):
            """
            Deposit plasma particles onto the charge density and current
            grids.
            """
            xp = const_arrays.xp

            ro = xp.zeros((grid_steps, grid_steps), dtype=xp.float64)
            jx = xp.zeros((grid_steps, grid_steps), dtype=xp.float64)
            jy = xp.zeros((grid_steps, grid_steps), dtype=xp.float64)
            jz = xp.zeros((grid_steps, grid_steps), dtype=xp.float64)

            x_h = (x_init + x_offt) / grid_step_size + 0.5
            y_h = (y_init + y_offt) / grid_step_size + 0.5

            deposit_kernel(grid_steps,
                           x_h.ravel(), y_h.ravel(),
                           m.ravel(), q.ravel(),
                           px.ravel(), py.ravel(), pz.ravel(),
                           ro, jx, jy, jz)

            return ro, jx, jy, jz

        return deposit_single

    elif pu_type == 'gpu':
        deposit_plasma_cupy = get_deposit_plasma_cupy()

        if dual_plasma_approach:
            plasma_coarseness = config.getint('plasma-coarseness')

            coarse_to_fine = get_coarse_to_fine_cupy()

            def deposit_dual(x_init, y_init, x_offt, y_offt, px, py, pz, q, m,
                             const_arrays: Arrays):
                """
                Interpolate coarse plasma into fine plasma and deposit it on the
                charge density and current grids. This is a convenience wrapper
                around the `deposit_kernel` CUDA kernel.
                """
                xp = const_arrays.xp

                virtplasma_smallness_factor = 1 / (
                    plasma_coarseness * plasma_fineness)**2
                fine_grid_size = const_arrays.fine_grid.size
                coarse_grid_size = len(m)

                x_fine  = xp.zeros((fine_grid_size, fine_grid_size))
                y_fine  = xp.zeros((fine_grid_size, fine_grid_size))
                px_fine = xp.zeros((fine_grid_size, fine_grid_size))
                py_fine = xp.zeros((fine_grid_size, fine_grid_size))
                pz_fine = xp.zeros((fine_grid_size, fine_grid_size))
                q_fine  = xp.zeros((fine_grid_size, fine_grid_size))
                m_fine  = xp.zeros((fine_grid_size, fine_grid_size))

                x_fine, y_fine, px_fine, py_fine, pz_fine, q_fine, m_fine =\
                    coarse_to_fine(
                        virtplasma_smallness_factor,
                        fine_grid_size, coarse_grid_size,
                        x_offt, y_offt, px, py, pz, q, m,
                        const_arrays.fine_grid,
                        const_arrays.influence_prev, const_arrays.influence_next,
                        const_arrays.indices_prev, const_arrays.indices_next,
                        x_fine, y_fine, px_fine, py_fine, pz_fine, q_fine, m_fine,
                        size=fine_grid_size**2)

                ro = xp.zeros((grid_steps, grid_steps))
                jx = xp.zeros((grid_steps, grid_steps))
                jy = xp.zeros((grid_steps, grid_steps))
                jz = xp.zeros((grid_steps, grid_steps))

                x_h = x_fine / grid_step_size + 0.5
                y_h = y_fine / grid_step_size + 0.5

                return deposit_plasma_cupy(
                    grid_steps, x_h, y_h, px_fine, py_fine, pz_fine,
                    q_fine, m_fine, ro, jx, jy, jz, size=fine_grid_size**2
                )

            return deposit_dual

        else:
            def deposit_single(x_init, y_init, x_offt, y_offt, px, py, pz, q, m,
                               const_arrays: Arrays):
                """
                Deposit plasma particles onto the charge density and current
                grids.
                """
                xp = const_arrays.xp

                ro = xp.zeros((grid_steps, grid_steps), dtype=xp.float64)
                jx = xp.zeros((grid_steps, grid_steps), dtype=xp.float64)
                jy = xp.zeros((grid_steps, grid_steps), dtype=xp.float64)
                jz = xp.zeros((grid_steps, grid_steps), dtype=xp.float64)

                x_h = (x_init + x_offt) / grid_step_size + 0.5
                y_h = (y_init + y_offt) / grid_step_size + 0.5

                return deposit_plasma_cupy(
                    grid_steps, x_h, y_h, px, py, pz, q, m, ro, jx, jy, jz,
                    size=m.size
                )

            return deposit_single
