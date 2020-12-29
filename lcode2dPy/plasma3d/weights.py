"""Module for weights calculation, interpolation and deposition routines."""
import numba as nb
import numpy as np

from math import sqrt, floor

# Deposition and interpolation helper functions #

@nb.njit(parallel=True)
def weights(x, y, grid_steps, grid_step_size):
    """
    Calculate the indices of a cell corresponding to the coordinates,
    and the coefficients of interpolation and deposition for this cell
    and 8 surrounding cells.
    The weights correspond to 2D triangluar shaped cloud (TSC2D).
    """
    x_h, y_h = x / grid_step_size + .5, y / grid_step_size + .5
    i, j = int(floor(x_h) + grid_steps // 2), int(floor(y_h) + grid_steps // 2)
    x_loc, y_loc = x_h - floor(x_h) - .5, y_h - floor(y_h) - .5
    # centered to -.5 to 5, not 0 to 1, as formulas use offset from cell center
    # TODO: get rid of this deoffsetting/reoffsetting festival

    wx0, wy0 = .75 - x_loc**2, .75 - y_loc**2  # fx1, fy1
    wxP, wyP = (.5 + x_loc)**2 / 2, (.5 + y_loc)**2 / 2  # fx2**2/2, fy2**2/2
    wxM, wyM = (.5 - x_loc)**2 / 2, (.5 - y_loc)**2 / 2  # fx3**2/2, fy3**2/2

    wMP, w0P, wPP = wxM * wyP, wx0 * wyP, wxP * wyP
    wM0, w00, wP0 = wxM * wy0, wx0 * wy0, wxP * wy0
    wMM, w0M, wPM = wxM * wyM, wx0 * wyM, wxP * wyM

    return i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM


@nb.njit(cache=True)
def deposit9(a, i, j, val, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM):
    """
    Deposit value into a cell and 8 surrounding cells (using `weights` output).
    """
    a[i - 1, j + 1] += val * wMP
    a[i + 0, j + 1] += val * w0P
    a[i + 1, j + 1] += val * wPP
    a[i - 1, j + 0] += val * wM0
    a[i + 0, j + 0] += val * w00
    a[i + 1, j + 0] += val * wP0
    a[i - 1, j - 1] += val * wMM
    a[i + 0, j - 1] += val * w0M
    a[i + 1, j - 1] += val * wPM


# Deposition #

@nb.njit(parallel=True)
def deposit_kernel(grid_steps, grid_step_size,
                   x_init, y_init, x_offt, y_offt, m, q, px, py, pz,
                   out_ro, out_jx, out_jy, out_jz):
    """
    Deposit plasma particles onto the charge density and current grids.
    """
    for k in range(m.size):    
        # Deposit the resulting fine particle on ro/j grids.
        gamma_m = sqrt(m[k]**2 + px[k]**2 + py[k]**2 + pz[k]**2)
        dro = q[k] / (1 - pz[k] / gamma_m)
        djx = px[k] * (dro / gamma_m)
        djy = py[k] * (dro / gamma_m)
        djz = pz[k] * (dro / gamma_m)

        x, y = x_init[k] + x_offt[k], y_init[k] + y_offt[k]
        i, j, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM = weights(
            x, y, grid_steps, grid_step_size
        )
        deposit9(out_ro, i, j, dro, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        deposit9(out_jx, i, j, djx, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        deposit9(out_jy, i, j, djy, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)
        deposit9(out_jz, i, j, djz, wMP, w0P, wPP, wM0, w00, wP0, wMM, w0M, wPM)


@nb.njit
def deposit(grid_steps, grid_step_size,
            x_init, y_init, x_offt, y_offt, px, py, pz, q, m):
    """
    Deposit plasma particles onto the charge density and current grids.
    This is a convenience wrapper around the `deposit_kernel` CUDA kernel.
    """   
    ro = np.zeros((grid_steps, grid_steps))
    jx = np.zeros((grid_steps, grid_steps))
    jy = np.zeros((grid_steps, grid_steps))
    jz = np.zeros((grid_steps, grid_steps))

    deposit_kernel(grid_steps, grid_step_size,
                   x_init.ravel(), y_init.ravel(),
                   x_offt.ravel(), y_offt.ravel(),
                   m.ravel(), q.ravel(),
                   px.ravel(), py.ravel(), pz.ravel(),
                   ro, jx, jy, jz)

    return ro, jx, jy, jz


def initial_deposition(grid_steps, grid_step_size,
                       x_init, y_init, x_offt, y_offt, px, py, pz, m, q):
    """
    Determine the background ion charge density by depositing the electrons
    with their initial parameters and negating the result.
    """
    ro_electrons_initial, _, _, _ = deposit(grid_steps, grid_step_size,
                                            x_init, y_init, x_offt, y_offt,
                                            px, py, pz, q, m)
    return -ro_electrons_initial