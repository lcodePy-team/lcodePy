"""Module for weights calculation, interpolation and deposition routines."""

# Helper function for dual plasma approach #

def get_coarse_to_fine_func():
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


# Deposition and interpolation helper function #

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


def get_deposit_single_kernel_cupy():
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


def get_deposit_func(dual_plasma_approach, grid_steps, grid_step_size,
                     plasma_coarseness, plasma_fineness):
    """
    Check if a user set dual-plasma-approach to True or False and return
    the deposition function for the set approach.
    """
    import cupy as cp

    deposit_kernel_cupy = get_deposit_single_kernel_cupy()
    
    if dual_plasma_approach:
        coarse_to_fine = get_coarse_to_fine_func()

        def deposit_dual(x_init, y_init, x_offt, y_offt, px, py, pz, q, m,
                         const_arrays):
            """
            Interpolate coarse plasma into fine plasma and deposit it on the
            charge density and current grids. This is a convenience wrapper
            around the `deposit_kernel` CUDA kernel.
            """
            virtplasma_smallness_factor = 1 / (
                plasma_coarseness * plasma_fineness)**2
            fine_grid_size = const_arrays.fine_grid.size
            coarse_grid_size = len(m)

            x_fine  = cp.zeros((fine_grid_size, fine_grid_size))
            y_fine  = cp.zeros((fine_grid_size, fine_grid_size))
            px_fine = cp.zeros((fine_grid_size, fine_grid_size))
            py_fine = cp.zeros((fine_grid_size, fine_grid_size))
            pz_fine = cp.zeros((fine_grid_size, fine_grid_size))
            q_fine  = cp.zeros((fine_grid_size, fine_grid_size))
            m_fine  = cp.zeros((fine_grid_size, fine_grid_size))

            x_fine, y_fine, px_fine, py_fine, pz_fine, q_fine, m_fine =\
                coarse_to_fine(
                    virtplasma_smallness_factor,
                    fine_grid_size, coarse_grid_size,
                    x_offt, y_offt, px, py, pz, q, m, const_arrays.fine_grid,
                    const_arrays.influence_prev, const_arrays.influence_next,
                    const_arrays.indices_prev, const_arrays.indices_next,
                    x_fine, y_fine, px_fine, py_fine, pz_fine, q_fine, m_fine,
                    size=fine_grid_size**2)
            
            ro = cp.zeros((grid_steps, grid_steps))
            jx = cp.zeros((grid_steps, grid_steps))
            jy = cp.zeros((grid_steps, grid_steps))
            jz = cp.zeros((grid_steps, grid_steps))

            x_h = x_fine / grid_step_size + 0.5
            y_h = y_fine / grid_step_size + 0.5

            return deposit_kernel_cupy(
                grid_steps, x_h, y_h, px_fine, py_fine, pz_fine,
                q_fine, m_fine, ro, jx, jy, jz, size=fine_grid_size**2
            )
        
        return deposit_dual

    else:
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

            x_h = (x_init + x_offt) / grid_step_size + 0.5
            y_h = (y_init + y_offt) / grid_step_size + 0.5

            return deposit_kernel_cupy(
                grid_steps, x_h, y_h, px, py, pz, q, m, ro, jx, jy, jz, size=m.size
            )

        return deposit_single
