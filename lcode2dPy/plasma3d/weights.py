"""Module for weights calculation, interpolation and deposition routines."""
import numba as nb
import numpy as np

from math import sqrt, floor

# Deposition and interpolation helper functions #

@nb.njit#(parallel=True)
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


@nb.njit(cache=True)
def deposit25(a, i, j, val, 
        w2M2P, w1M2P, w02P, w1P2P, w2P2P,
        w2M1P, w1M1P, w01P, w1P1P, w2P1P,
        w2M0,  w1M0,  w00,  w1P0,  w2P0,
        w2M1M, w1M1M, w01M, w1P1M, w2P1M,
        w2M2M, w1M2M, w02M, w1P2M, w2P2M):
    """
    Deposit value into a cell and 24 surrounding cells (using `weights` output).
    """
    a[i - 2, j + 2] += val * w2M2P
    a[i - 1, j + 2] += val * w1M2P
    a[i + 0, j + 2] += val * w02P
    a[i + 1, j + 2] += val * w1P2P
    a[i + 2, j + 2] += val * w2P2P
    a[i - 2, j + 1] += val * w2M1P
    a[i - 1, j + 1] += val * w1M1P
    a[i + 0, j + 1] += val * w01P
    a[i + 1, j + 1] += val * w1P1P
    a[i + 2, j + 1] += val * w2P1P
    a[i - 2, j + 0] += val * w2M0
    a[i - 1, j + 0] += val * w1M0
    a[i + 0, j + 0] += val * w00
    a[i + 1, j + 0] += val * w1P0
    a[i + 2, j + 0] += val * w2P0
    a[i - 2, j - 1] += val * w2M1M
    a[i - 1, j - 1] += val * w1M1M
    a[i + 0, j - 1] += val * w01M
    a[i + 1, j - 1] += val * w1P1M
    a[i + 2, j - 1] += val * w2P1M
    a[i - 2, j - 2] += val * w2M2M
    a[i - 1, j - 2] += val * w1M2M
    a[i + 0, j - 2] += val * w02M
    a[i + 1, j - 2] += val * w1P2M
    a[i + 2, j - 2] += val * w2P2M


# Deposition #

@nb.njit# (parallel=True)
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