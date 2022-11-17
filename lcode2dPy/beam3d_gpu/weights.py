"""Module for weights calculation, interpolation and deposition routines."""

# Calculate the weights for a particle

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

# A function that performs gridding of beam particles onto the charge density grid.

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
        name='deposit_beam_cupy', preamble=weight1_cupy+weight4_cupy
    )


def get_deposit_beam(grid_steps, grid_step_size, xi_step_size):
    deposit_beam_cupy = get_deposit_beam_cupy()

    def deposit_beam(plasma_layer_idx, x, y, xi, q_norm, out_ro0, out_ro1):
        """
        Deposit beam particles onto the charge density grids.
        """
        xi_plasma_layer = - xi_step_size * plasma_layer_idx
        xi_loc = (xi_plasma_layer - xi) / xi_step_size
        x_h, y_h = x / grid_step_size + 0.5, y / grid_step_size + 0.5

        return deposit_beam_cupy(
            grid_steps, x_h, y_h, xi_loc, q_norm, out_ro0, out_ro1,
            size=q_norm.size
        )
    
    return deposit_beam
