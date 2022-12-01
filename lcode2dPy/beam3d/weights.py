"""Module for weights calculation and deposition routines."""
import numba as nb
from math import floor

from ..config.config import Config


@nb.njit
def weights(x, y, xi_loc, grid_steps, grid_step_size):
    """
    Calculates the position and the weights of a beam particleon a 3d cartesian
    grid.
    """
    x_h, y_h = x / grid_step_size + .5, y / grid_step_size + .5
    i, j = int(floor(x_h) + grid_steps // 2), int(floor(y_h) + grid_steps // 2)
    x_loc, y_loc = x_h - floor(x_h) - .5, y_h - floor(y_h) - .5
    # xi_loc = dxi = (xi_prev - xi) / D_XIP

    # First order core along xi axis
    wxi0 = xi_loc
    wxiP = (1 - xi_loc)

    # Second order core along x and y axes
    wx0, wy0 = .75 - x_loc**2, .75 - y_loc**2
    wxP, wyP = (.5 + x_loc)**2 / 2, (.5 + y_loc)**2 / 2
    wxM, wyM = (.5 - x_loc)**2 / 2, (.5 - y_loc)**2 / 2

    w0MP, w00P, w0PP = wxi0 * wxM * wyP, wxi0 * wx0 * wyP, wxi0 * wxP * wyP
    w0M0, w000, w0P0 = wxi0 * wxM * wy0, wxi0 * wx0 * wy0, wxi0 * wxP * wy0
    w0MM, w00M, w0PM = wxi0 * wxM * wyM, wxi0 * wx0 * wyM, wxi0 * wxP * wyM

    wPMP, wP0P, wPPP = wxiP * wxM * wyP, wxiP * wx0 * wyP, wxiP * wxP * wyP
    wPM0, wP00, wPP0 = wxiP * wxM * wy0, wxiP * wx0 * wy0, wxiP * wxP * wy0
    wPMM, wP0M, wPPM = wxiP * wxM * wyM, wxiP * wx0 * wyM, wxiP * wxP * wyM

    return (i, j,
            w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
            wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM
    )


# Deposition and field interpolation #

@nb.njit(parallel=True)
def deposit_beam_cpu(grid_steps, grid_step_size, x, y, xi_loc, q_norm,
                     rho_layout_0, rho_layout_1):
    """
    Deposit beam particles onto the charge density grids.
    """
    for k in nb.prange(len(q_norm)):

        # Calculate the weights for a particle
        (i, j,
        w0MP, w00P, w0PP, w0M0, w000, w0P0, w0MM, w00M, w0PM,
        wPMP, wP0P, wPPP, wPM0, wP00, wPP0, wPMM, wP0M, wPPM
        ) = weights(
            x[k], y[k], xi_loc[k], grid_steps, grid_step_size
        )

        rho_layout_0[i - 1, j + 1] += q_norm[k] * w0MP
        rho_layout_0[i + 0, j + 1] += q_norm[k] * w00P
        rho_layout_0[i + 1, j + 1] += q_norm[k] * w0PP
        rho_layout_0[i - 1, j + 0] += q_norm[k] * w0M0
        rho_layout_0[i + 0, j + 0] += q_norm[k] * w000
        rho_layout_0[i + 1, j + 0] += q_norm[k] * w0P0
        rho_layout_0[i - 1, j - 1] += q_norm[k] * w0MM
        rho_layout_0[i + 0, j - 1] += q_norm[k] * w00M
        rho_layout_0[i + 1, j - 1] += q_norm[k] * w0PM

        rho_layout_1[i - 1, j + 1] += q_norm[k] * wPMP
        rho_layout_1[i + 0, j + 1] += q_norm[k] * wP0P
        rho_layout_1[i + 1, j + 1] += q_norm[k] * wPPP
        rho_layout_1[i - 1, j + 0] += q_norm[k] * wPM0
        rho_layout_1[i + 0, j + 0] += q_norm[k] * wP00
        rho_layout_1[i + 1, j + 0] += q_norm[k] * wPP0
        rho_layout_1[i - 1, j - 1] += q_norm[k] * wPMM
        rho_layout_1[i + 0, j - 1] += q_norm[k] * wP0M
        rho_layout_1[i + 1, j - 1] += q_norm[k] * wPPM


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
        name='deposit_beam_cupy', preamble=weight1_cupy+weight4_cupy
    )


def get_deposit_beam(config: Config):
    xi_step_size = config.getfloat('xi-step')
    grid_step_size = config.getfloat('window-width-step-size')
    grid_steps = config.getint('window-width-steps')
    pu_type = config.get('processing-unit-type').lower()

    if pu_type == 'cpu':
        def deposit_beam(plasma_layer_idx, x, y, xi, q_norm,
                         rho_layout_0, rho_layout_1):
            """
            Deposit beam particles onto the charge density grid.
            This is a convenience wrapper around the `deposit_kernel` CUDA kernel.
            """
            xi_plasma_layer = - xi_step_size * plasma_layer_idx
            xi_loc = (xi_plasma_layer - xi) / xi_step_size

            return deposit_beam_cpu(
                grid_steps, grid_step_size, x.ravel(), y.ravel(),
                xi_loc.ravel(), q_norm.ravel(), rho_layout_0, rho_layout_1)

    elif pu_type == 'gpu':
        deposit_beam_cupy = get_deposit_beam_cupy()

        def deposit_beam(plasma_layer_idx, x, y, xi, q_norm,
                         out_ro0, out_ro1):
            """
            Deposit beam particles onto the charge density grids.
            """
            xi_plasma_layer = - xi_step_size * plasma_layer_idx
            xi_loc = (xi_plasma_layer - xi) / xi_step_size
            x_h, y_h = x / grid_step_size + 0.5, y / grid_step_size + 0.5

            return deposit_beam_cupy(
                grid_steps, x_h, y_h, xi_loc, q_norm, out_ro0, out_ro1,
                size=q_norm.size)

    return deposit_beam
