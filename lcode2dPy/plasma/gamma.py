"""Routine for computing gamma_mass."""
from math import sqrt

from numba import float64, vectorize


@vectorize([float64(float64, float64, float64, float64)], cache=True)
def gamma_mass_nb(mass, p_x, p_y, p_z):
    """Compute particle mass multiplied by gamma-factor."""
    return sqrt(mass ** 2 + p_x ** 2 + p_y ** 2 + p_z ** 2)
