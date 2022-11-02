"""Module for weights calculation, interpolation and deposition routines."""
import numba as numba
import numba.cuda

from math import sqrt, floor

WARP_SIZE = 32


# Deposition and interpolation helper functions #

@numba.njit
def weights(x, y, grid_steps, grid_step_size):
    """
    Calculate the indices of a cell corresponding to the coordinates,
    and the coefficients of interpolation and deposition for this cell
    and 24 surrounding cells.
    The weights correspond to ...
    """
    x_h, y_h = x / grid_step_size + .5, y / grid_step_size + .5
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


@numba.jit
def deposit25(a, i, j, val, 
        w2M2P, w1M2P, w02P, w1P2P, w2P2P,
        w2M1P, w1M1P, w01P, w1P1P, w2P1P,
        w2M0,  w1M0,  w00,  w1P0,  w2P0,
        w2M1M, w1M1M, w01M, w1P1M, w2P1M,
        w2M2M, w1M2M, w02M, w1P2M, w2P2M):
    """
    Deposit value into a cell and 8 surrounding cells (using `weights` output).
    """
    numba.cuda.atomic.add(a, (i - 2, j + 2), val * w2M2P)
    numba.cuda.atomic.add(a, (i - 1, j + 2), val * w1M2P)
    numba.cuda.atomic.add(a, (i + 0, j + 2), val * w02P)
    numba.cuda.atomic.add(a, (i + 1, j + 2), val * w1P2P)
    numba.cuda.atomic.add(a, (i + 2, j + 2), val * w2P2P)
    numba.cuda.atomic.add(a, (i - 2, j + 1), val * w2M1P)
    numba.cuda.atomic.add(a, (i - 1, j + 1), val * w1M1P)
    numba.cuda.atomic.add(a, (i + 0, j + 1), val * w01P)
    numba.cuda.atomic.add(a, (i + 1, j + 1), val * w1P1P)
    numba.cuda.atomic.add(a, (i + 2, j + 1), val * w2P1P)
    numba.cuda.atomic.add(a, (i - 2, j + 0), val * w2M0)
    numba.cuda.atomic.add(a, (i - 1, j + 0), val * w1M0)
    numba.cuda.atomic.add(a, (i + 0, j + 0), val * w00)
    numba.cuda.atomic.add(a, (i + 1, j + 0), val * w1P0)
    numba.cuda.atomic.add(a, (i + 2, j + 0), val * w2P0)
    numba.cuda.atomic.add(a, (i - 2, j - 1), val * w2M1M)
    numba.cuda.atomic.add(a, (i - 1, j - 1), val * w1M1M)
    numba.cuda.atomic.add(a, (i + 0, j - 1), val * w01M)
    numba.cuda.atomic.add(a, (i + 1, j - 1), val * w1P1M)
    numba.cuda.atomic.add(a, (i + 2, j - 1), val * w2P1M)
    numba.cuda.atomic.add(a, (i - 2, j - 2), val * w2M2M)
    numba.cuda.atomic.add(a, (i - 1, j - 2), val * w1M2M)
    numba.cuda.atomic.add(a, (i + 0, j - 2), val * w02M)
    numba.cuda.atomic.add(a, (i + 1, j - 2), val * w1P2M)
    numba.cuda.atomic.add(a, (i + 2, j - 2), val * w2P2M)


# Helper functions for dual plasma approach #

# TODO: This doesn't work, figure it out!
@numba.njit(inline='always')
def mix(coarse, A, B, C, D, pi, ni, pj, nj):
    """
    Bilinearly interpolate fine plasma properties from four
    historically-neighbouring plasma particle property values.
     B    D  #  y ^         A - bottom-left  neighbour, indices: pi, pj
        .    #    |         B - top-left     neighbour, indices: pi, nj
             #    +---->    C - bottom-right neighbour, indices: ni, pj
     A    C  #         x    D - top-right    neighbour, indices: ni, nj
    See the rest of the deposition and plasma creation for more info.
    """
    return (A * coarse[pi, pj] + B * coarse[pi, nj] +
            C * coarse[ni, pj] + D * coarse[ni, nj])


@numba.njit(inline='always')
def coarse_to_fine(
    fi, fj, c_x_offt, c_y_offt, c_m, c_q, c_px, c_py, c_pz,
    virtplasma_smallness_factor, fine_grid,
    influence_prev, influence_next, indices_prev, indices_next
):
    """
    Bilinearly interpolate fine plasma properties from four
    historically-neighbouring plasma particle property values.
    """
    # Calculate the weights of the historically-neighbouring coarse particles
    A = influence_prev[fi] * influence_prev[fj]
    B = influence_prev[fi] * influence_next[fj]
    C = influence_next[fi] * influence_prev[fj]
    D = influence_next[fi] * influence_next[fj]
    # and retrieve their indices.
    pi, ni = indices_prev[fi], indices_next[fi]
    pj, nj = indices_prev[fj], indices_next[fj]

    # Now we're ready to mix the fine particle characteristics
    # x_offt = (
    #     A * c_x_offt[pi, pj] + B * c_x_offt[pi, nj] +
    #     C * c_x_offt[ni, pj] + D * c_x_offt[ni, nj]
    # )
    # y_offt = (
    #     A * c_y_offt[pi, pj] + B * c_y_offt[pi, nj] +
    #     C * c_y_offt[ni, pj] + D * c_y_offt[ni, nj]
    # )
    x_offt = mix(c_x_offt, A, B, C, D, pi, ni, pj, nj)
    y_offt = mix(c_y_offt, A, B, C, D, pi, ni, pj, nj)
    x = fine_grid[fi] + x_offt  # x_fine_init
    y = fine_grid[fj] + y_offt  # y_fine_init

    # TODO: const m and q
    # m_mix = (
    #     A * c_m[pi, pj] + B * c_m[pi, nj] + C * c_m[ni, pj] + D * c_m[ni, nj]
    # )
    # q_mix = (
    #     A * c_q[pi, pj] + B * c_q[pi, nj] + C * c_q[ni, pj] + D * c_q[ni, nj]
    # )
    m = virtplasma_smallness_factor * mix(c_m, A, B, C, D, pi, ni, pj, nj)
    q = virtplasma_smallness_factor * mix(c_q, A, B, C, D, pi, ni, pj, nj)

    # px_mix = (
    #     A * c_px[pi, pj] + B * c_px[pi, nj] + C * c_px[ni, pj] + D * c_px[ni, nj]
    # )
    # py_mix = (
    #     A * c_py[pi, pj] + B * c_py[pi, nj] + C * c_py[ni, pj] + D * c_py[ni, nj]
    # )
    # pz_mix = (
    #     A * c_pz[pi, pj] + B * c_pz[pi, nj] + C * c_pz[ni, pj] + D * c_pz[ni, nj]
    # )
    px = virtplasma_smallness_factor * mix(c_px, A, B, C, D, pi, ni, pj, nj)
    py = virtplasma_smallness_factor * mix(c_py, A, B, C, D, pi, ni, pj, nj)
    pz = virtplasma_smallness_factor * mix(c_pz, A, B, C, D, pi, ni, pj, nj)
    return x, y, m, q, px, py, pz


# Deposition in case of the single plasma approach #

weight_cupy = """
__device__ inline T weight(T x, int place) {
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


# TODO: Doesn't look very smart to get a kernel this way.
def get_deposit_single_kernel_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 grid_steps, float64 grid_step_size,
        raw T x_init, raw T y_init, raw T x_offt, raw T y_offt,
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

        const T x_h = (x_init[i] + x_offt[i]) / (T) grid_step_size + 0.5;
        const T y_h = (y_init[i] + y_offt[i]) / (T) grid_step_size + 0.5;
        const T x_loc = x_h - floor(x_h) - 0.5;
        const T y_loc = y_h - floor(y_h) - 0.5;
        const int ix = floor(x_h) + floor(grid_steps / 2);
        const int iy = floor(y_h) + floor(grid_steps / 2);

        for (int kx = -2; kx <= 2; kx++) {
            const T wx = weight(x_loc, kx);
            for (int ky = -2; ky <= 2; ky++) {
                const T w = wx * weight(y_loc, ky);
                const int idx = (iy + ky) + (int) grid_steps * (ix + kx);

                atomicAdd(&out_ro[idx], dro * w);
                atomicAdd(&out_jx[idx], djx * w);
                atomicAdd(&out_jy[idx], djy * w);
                atomicAdd(&out_jz[idx], djz * w);
            }
        }
        """,
        name='deposit_cupy', preamble=weight_cupy
    )


# Deposition in case of the dual plasma approach #

@numba.cuda.jit
def deposit_dual_kernel(
        grid_steps, grid_step_size, virtplasma_smallness_factor,
        c_x_offt, c_y_offt, c_m, c_q, c_px, c_py, c_pz, # coarse
        fine_grid,
        influence_prev, influence_next, indices_prev, indices_next,
        out_ro, out_jx, out_jy, out_jz):
    """
    Interpolate coarse plasma into fine plasma and deposit it on the
    charge density and current grids.
    """
    # Do nothing if our thread does not have a fine particle to deposit.
    fk = numba.cuda.grid(1)
    if fk >= fine_grid.size**2:
        return
    fi, fj = fk // fine_grid.size, fk % fine_grid.size

    # Interpolate fine plasma particle from coarse particle characteristics
    x, y, m, q, px, py, pz = coarse_to_fine(fi, fj, c_x_offt, c_y_offt,
                                            c_m, c_q, c_px, c_py, c_pz,
                                            virtplasma_smallness_factor,
                                            fine_grid,
                                            influence_prev, influence_next,
                                            indices_prev, indices_next)

    # Deposit the resulting fine particle on ro/j grids.
    gamma_m = sqrt(m**2 + px**2 + py**2 + pz**2)
    dro = q / (1 - pz / gamma_m)
    djx = px * (dro / gamma_m)
    djy = py * (dro / gamma_m)
    djz = pz * (dro / gamma_m)

    (i, j, 
         w2M2P, w1M2P, w02P, w1P2P, w2P2P,
         w2M1P, w1M1P, w01P, w1P1P, w2P1P,
         w2M0,  w1M0,  w00,  w1P0,  w2P0,
         w2M1M, w1M1M, w01M, w1P1M, w2P1M,
         w2M2M, w1M2M, w02M, w1P2M, w2P2M
    ) = weights(
        x, y, grid_steps, grid_step_size
    )
    
    deposit25(out_ro, i, j, dro,
                w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                w2M0,  w1M0,  w00,  w1P0,  w2P0,
                w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                w2M2M, w1M2M, w02M, w1P2M, w2P2M)
    deposit25(out_jx, i, j, djx,
                w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                w2M0,  w1M0,  w00,  w1P0,  w2P0,
                w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                w2M2M, w1M2M, w02M, w1P2M, w2P2M)
    deposit25(out_jy, i, j, djy,
                w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                w2M0,  w1M0,  w00,  w1P0,  w2P0,
                w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                w2M2M, w1M2M, w02M, w1P2M, w2P2M)
    deposit25(out_jz, i, j, djz, 
                w2M2P, w1M2P, w02P, w1P2P, w2P2P,
                w2M1P, w1M1P, w01P, w1P1P, w2P1P,
                w2M0,  w1M0,  w00,  w1P0,  w2P0,
                w2M1M, w1M1M, w01M, w1P1M, w2P1M,
                w2M2M, w1M2M, w02M, w1P2M, w2P2M)


def get_deposit_func(dual_plasma_approach, grid_steps, grid_step_size,
                     plasma_coarseness, plasma_fineness):
    """
    Check if a user set dual-plasma-approach to True or False and return
    the deposition function for the set approach.
    """
    import cupy as cp
    
    if dual_plasma_approach:
        def deposit_dual(x_init, y_init, x_offt, y_offt, px, py, pz, q, m,
                         const_arrays):
            """
            Interpolate coarse plasma into fine plasma and deposit it on the
            charge density and current grids. This is a convenience wrapper
            around the `deposit_kernel` CUDA kernel.
            """
            virtplasma_smallness_factor = 1 / (
                plasma_coarseness * plasma_fineness)**2
            ro = cp.zeros((grid_steps, grid_steps))
            jx = cp.zeros((grid_steps, grid_steps))
            jy = cp.zeros((grid_steps, grid_steps))
            jz = cp.zeros((grid_steps, grid_steps))

            cfg = (int(cp.ceil(const_arrays.fine_grid.size**2 / WARP_SIZE)),
                   WARP_SIZE)
            deposit_dual_kernel[cfg](
                grid_steps, grid_step_size, virtplasma_smallness_factor,
                x_offt, y_offt, m, q, px, py, pz, const_arrays.fine_grid,
                const_arrays.influence_prev, const_arrays.influence_next,
                const_arrays.indices_prev, const_arrays.indices_next,
                ro, jx, jy, jz)

            return ro, jx, jy, jz
        
        return deposit_dual

    else:
        deposit_kernel_cupy = get_deposit_single_kernel_cupy()

        def deposit_single(x_init, y_init, x_offt, y_offt, px, py, pz, q, m,
                           const_arrays):
            """
            Deposit plasma particles onto the charge density and current grids.
            This is a convenience wrapper around the `deposit_kernel` CUDA kernel.
            """   
            ro = cp.zeros((grid_steps, grid_steps), dtype=cp.float64)
            jx = cp.zeros((grid_steps, grid_steps), dtype=cp.float64)
            jy = cp.zeros((grid_steps, grid_steps), dtype=cp.float64)
            jz = cp.zeros((grid_steps, grid_steps), dtype=cp.float64)

            return deposit_kernel_cupy(
                grid_steps, grid_step_size,
                x_init, y_init, x_offt, y_offt,
                px, py, pz, q, m, ro, jx, jy, jz,
                size=m.size
            )

        return deposit_single
