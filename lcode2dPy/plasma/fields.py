"""Functions to find fields on the next step of plasma evolution."""
import numpy as np
from numba import njit

from lcode2dPy.plasma.data import Fields
from lcode2dPy.plasma.ode import (
    cumtrapz_numba,
    tridiagonal_solve_neumann_like,
    tridiagonal_solve_dirichlet,
)


#NJIT@nb.njit @njit
def compute_e_r(total_rho, j_r, j_r_previous, e_r_previous, previous_factor,
                r_step, xi_step_p):
    """
    Compute :math:`E_r` field on the next xi step.

    Parameters
    ----------
    total_rho : np.ndarray
        Predicted total (plasma and beam) charge density on the next step.
    j_r : np.ndarray
        Predicted radial plasma current density on the next step.
    j_r_previous : np.ndarray
        Radial plasma current density from the previous step.
    e_r_previous : np.ndarray
        :math:`E_r` field from the previous step.
    previous_factor : float or np.ndarray
        Factor of usage of the previous value.
    r_step : float
        Radial grid step size.
    xi_step_p : float
        Absolute value of longitudinal grid step size.

    Returns
    -------
    np.ndarray
        :math:`E_r` field on the next step.

    """
    total_rho_deriv_r = (total_rho[2:] - total_rho[:-2]) / 2 / r_step

    j_r_deriv_xi = (j_r - j_r_previous) / xi_step_p

    right_part = j_r_deriv_xi - previous_factor * e_r_previous
    right_part[1:-1] += total_rho_deriv_r
    # Boundary conditions
    right_part[0] = 0
    right_part[-1] = (total_rho[-2] + total_rho[-1]) / 2

    return tridiagonal_solve_neumann_like(right_part, previous_factor, r_step)


#NJIT@nb.njit @njit
def compute_e_phi(j_phi, j_phi_previous, e_phi_previous, previous_factor,
                  r_step, xi_step_p):
    r"""
    Compute :math:`E_\phi` field on the next xi step.

    Parameters
    ----------
    j_phi : np.ndarray
        Predicted radial plasma current density on the next step.
    j_phi_previous : np.ndarray
        Second transverse plasma current density from the previous step.
    e_phi_previous : np.ndarray
        :math:`E_\phi` field from the previous step.
    previous_factor : float or np.ndarray
        Factor of usage of the previous field value.
    r_step : float
        Radial grid step size.
    xi_step_p : float
        Absolute value of longitudinal grid step size.

    Returns
    -------
    np.ndarray
        :math:`E_\phi` field on the next step.

    """
    j_f_deriv_xi = (j_phi - j_phi_previous) / xi_step_p
    right_part = j_f_deriv_xi
    right_part -= previous_factor * e_phi_previous
    # Boundary conditions
    right_part[0] = 0
    right_part[-1] = 0

    return tridiagonal_solve_dirichlet(right_part, previous_factor, r_step)


#NJIT@nb.njit @njit
def compute_e_z(j_r, r_step):
    """
    Compute :math:`E_z` field on the next xi step.

    Parameters
    ----------
    j_r : np.ndarray
        Predicted radial plasma current density on the next step.
    r_step : float
        Radial grid step size.

    Returns
    -------
    np.ndarray
        :math:`E_z` field on the next step.

    """
    new_e_z_field = np.zeros_like(j_r)
    cumtrapz_numba(j_r, r_step, new_e_z_field, mode='backward')
    return new_e_z_field


#NJIT@nb.njit @njit
def compute_b_phi(rho, j_z, e_r, r_step):
    r"""
    Compute :math:`B_\phi` field on the next xi step.

    Parameters
    ----------
    rho : np.ndarray
        Predicted plasma charge density on the next step.
    j_z : np.ndarray
        Predicted longitudinal plasma current density on the next step.
    e_r : np.ndarray
        Predicted :math:`E_r` field on the next step.
    r_step : float
        Radial grid step size.

    Returns
    -------
    np.ndarray
        :math:`B_\phi` field on the next step.

    """
    temporary_b_phi = np.zeros_like(e_r)
    cumtrapz_numba(
        np.arange(rho.size) * (rho - j_z),
        r_step,
        temporary_b_phi,
        mode='forward',
    )
    temporary_b_phi[1:] /= np.arange(1, rho.size)
    return e_r - temporary_b_phi


#NJIT@nb.njit @njit
def compute_b_z(j_phi, r_step):
    """
    Compute :math:`B_z` field on the next xi step.

    Parameters
    ----------
    j_phi : np.ndarray
        Predicted radial plasma current density on the next step.
    r_step : float
        Radial grid step size.

    Returns
    -------
    np.ndarray
        :math:`B_z` field on the next step.

    """
    new_b_z_field = np.zeros_like(j_phi)
    cumtrapz_numba(
        -j_phi,
        r_step,
        new_b_z_field,
        mode='backward',
    )
    magnetic_flux = np.sum((2 * np.arange(j_phi.size) + 1) * new_b_z_field)
    new_b_z_field -= magnetic_flux / (j_phi.size - 1) ** 2
    return new_b_z_field


_prev_factor_multiplier = 3
_prev_factor_threshold = 5


#NJIT@nb.njit @njit
def _compute_e_r_previous_factor(rho, j_r, j_phi, j_z):
    prev_factor = np.zeros_like(rho)
    electron_density = rho - 1.0  # TODO: Use correct electron density
    for j in np.arange(rho.size):
        if electron_density[j] > -0.1:
            continue
        invgamma_sq = 1 - (j_r[j] ** 2 + j_phi[j] ** 2 + j_z[j] ** 2) / electron_density[j] ** 2
        if invgamma_sq <= 0:
            continue
        prev_factor[j] = _prev_factor_multiplier * (1 / np.sqrt(invgamma_sq) - 1) - _prev_factor_threshold
    prev_factor[prev_factor < 0] = 0
    return 1 + prev_factor


class FieldComputer(object):
    """
    Class to store some parameters for fields computing.

    Parameters
    ----------
    config : lcode2dPy.config.Config

    Attributes
    ----------
    r_step : float
        Radial grid step size.

    """

    def __init__(self, config):
        self.r_step = float(config.get('window-width-step-size'))

    def compute_fields(self, fields, rho_beam, currents_previous, currents,
                       xi_step_p):
        """
        Compute fields on the next xi step.

        Parameters
        ----------
        fields : lcode2dPy.plasma.data.Fields
            Fields from the previous xi step
        rho_beam : np.ndarray
            Beam charge density on the next step.
        currents_previous : lcode2dPy.plasma.data.Currents
            Current densities from the previous step
        currents : lcode2dPy.plasma.data.Currents
            Predicted current densities on the next step
        xi_step_p : float
            Absolute value of longitudinal grid step size.

        Returns
        -------
        lcode2dPy.plasma.data.Fields
            Fields on the next xi step

        """
        previous_factor = _compute_e_r_previous_factor(
            currents.rho, currents.j_r, currents.j_f, currents.j_z,
        )
        new_fields = Fields(fields.E_r.size)
        total_rho = currents.rho + rho_beam
        new_fields.E_r = compute_e_r(total_rho, currents.j_r,
                                     currents_previous.j_r, fields.E_r,
                                     previous_factor, self.r_step, xi_step_p)
        previous_factor = np.ones_like(fields.E_f)
        new_fields.E_f = compute_e_phi(currents.j_f, currents_previous.j_f,
                                       fields.E_f, previous_factor,
                                       self.r_step, xi_step_p)
        new_fields.E_z = compute_e_z(currents.j_r, self.r_step)
        new_fields.B_f = compute_b_phi(currents.rho, currents.j_z, new_fields.E_r,
                                       self.r_step)
        new_fields.B_z = compute_b_z(currents.j_f, self.r_step)

        return new_fields
