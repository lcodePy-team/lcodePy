"""Routines to solve ODEs."""

import numpy as np
from numba import njit

@njit(cache=True)
def cumtrapz_numba(y, dx, out, mode):  # noqa: WPS111
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : ndarray
        1d array with values to integrate.
    dx : float
        Spacing between elements of y. Must be positive.
    out : ndarray
        A location into which the result is stored. Must have the same length
        as `y`.
    mode : {'forward', 'backward'}
        An array will be integrated from the beginning ('forward') or from
        the end ('backward') of the array.Initial value of the result array
        is 0.
    """
    length = out.size
    if mode == 'forward':
        out[0] = 0.0
        for idx in np.arange(0, length - 1):
            y_average = (y[idx + 1] + y[idx]) / 2
            out[idx + 1] = out[idx] + dx * y_average
    elif mode == 'backward':
        out[length - 1] = 0.0
        for idx in np.arange(length - 2, -1, -1):
            y_average = (y[idx + 1] + y[idx]) / 2
            out[idx] = out[idx + 1] - dx * y_average


# Solve ode using tridiagonal matrix method
# d/dx((d/dx(x * y))/x) - k(x) * y = f(x) - k(x) * y_approximation
@njit(cache=True)
def tridiagonal_solve(right_part, previous_factor, r_step, boundaries):
    nr = right_part.size - 1
    alpha = np.zeros_like(right_part)
    beta = np.zeros_like(right_part)
    for i in range(1, nr):
        a = 1 + 0.5 / i
        b = 2 + previous_factor[i] * r_step ** 2 + 1 / i ** 2
        c = 1 - 0.5 / i
        d = r_step ** 2 * right_part[i]
        denom = 1 / (b - c * alpha[i - 1])
        alpha[i] = a * denom
        beta[i] = (c * beta[i - 1] - d) * denom
    c = boundaries[0]
    b = boundaries[1]
    d = r_step * right_part[nr]
    beta[nr] = (c * beta[nr - 1] - d) / (b - c * alpha[nr - 1])
    for i in range(nr - 1, -1, -1):
        beta[i] += beta[i + 1] * alpha[i]
    return beta


@njit(cache=True)
def tridiagonal_solve_neumann_like(right_part, previous_factor, r_step):
    nr = right_part.size - 1
    boundaries = np.array((-(nr - 1) / (nr - 1 / 2), -nr / (nr - 1 / 2)))
    return tridiagonal_solve(right_part, previous_factor, r_step, boundaries)


@njit(cache=True)
def tridiagonal_solve_dirichlet(right_part, previous_factor, r_step):
    boundaries = np.array((0, -1))
    return tridiagonal_solve(right_part, previous_factor, r_step, boundaries)
