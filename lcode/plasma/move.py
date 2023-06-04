import numpy as np
from numba import njit, prange

from ..config.config import Config


@njit(cache=True, inline='always')
def _interpolate_fields(cell_idx, local_coord, E_r, E_f, E_z, B_f, B_z):
    """Interpolates fields from grid to particles positions."""
    a_1 = local_coord - 0.5
    a_1 *= 0.5 * a_1
    a_2 = 0.75 - local_coord ** 2
    a_3 = local_coord + 0.5
    a_3 *= 0.5 * a_3

    jc = cell_idx
    jm = jc - 1
    jp = jc + 1
    if jc == E_r.size - 1:
        jp = jm
    elif jc == 0:
        jm = jp
    elif jc >= E_r.size:
        return 0, 0, 0, 0, 0
    if jc == 0:
        e_x = (a_3 - a_1) * E_r[jm] + a_2 * E_r[jc]
        e_y = (a_3 - a_1) * E_f[jm] + a_2 * E_f[jc]
        b_y = (a_3 - a_1) * B_f[jm] + a_2 * B_f[jc]
    else:
        e_x = a_1 * E_r[jm] + a_2 * E_r[jc] + a_3 * E_r[jp]
        e_y = a_1 * E_f[jm] + a_2 * E_f[jc] + a_3 * E_f[jp]
        b_y = a_1 * B_f[jm] + a_2 * B_f[jc] + a_3 * B_f[jp]
    e_z = a_1 * E_z[jm] + a_2 * E_z[jc] + a_3 * E_z[jp]
    b_z = a_1 * B_z[jm] + a_2 * B_z[jc] + a_3 * B_z[jp]
    return e_x, e_y, e_z, b_y, b_z


@njit(cache=True, inline='always')
def _interpolate_noisereductor(
    noise_amplitude, cell_idx, local_coord, r_step,
):

    if cell_idx >= noise_amplitude.size:
        return 0
    """Calculates noise correction value to be added to e_x."""
    if cell_idx == 0 or noise_amplitude[cell_idx] == 0:
        return 0

    

    noise_func = np.pi * np.sin(np.pi * local_coord)
    noise_func += np.cos(np.pi * local_coord) / cell_idx
    noise_func /= (np.pi ** 2 + 1 / cell_idx ** 2)
    result = noise_amplitude[cell_idx] * r_step * noise_func
    return result


@njit(cache=True, inline='always')
def _move_one_particle(E_r, E_f, E_z, B_f, B_z,
                       r, p_r, p_f, p_z, q, m,
                       noise_amplitude, r_step, xi_step_p, max_radius):
    """Move one particle in fields.

    Returns
    -------
    r : float
    p_r : float
    p_f : float
    p_z : float
    q : float
        New parameters of the particle
    path : float
        Maximum distance between points of the particle trajectory
    age : float
        Effective age of the particle (xi_step_p / (1 - v_z))
    """
    p_xo = p_r
    p_yo = p_f
    p_zo = p_z
    gammam = np.sqrt(m ** 2 + p_xo ** 2 + p_yo ** 2 + p_zo ** 2)
    if gammam == p_z or not np.isfinite(gammam):
        return max_radius / 2, 0, 0, 0, 0, 0, 0
    dl_t = xi_step_p / (gammam - p_zo)

    # Predict middle point of particle trajectory
    r_pred = r_k_1_2 = r + p_xo * dl_t / 2

    # Correct particle position if it reaches boundary or crosses the axis
    if r_k_1_2 < 0:
        r_k_1_2 *= -1
        p_xo *= -1
    elif r_k_1_2 > max_radius:
        r_k_1_2 = 2 * max_radius - r_k_1_2
        p_xo = 0
        p_yo = 0
        p_zo = 0
        dl_t = xi_step_p / m
        gammam = m
    # xi_step_p was too big, particle is out of window after correction
    if r_k_1_2 < 0 or r_k_1_2 > max_radius:
        return max_radius / 2, 0, 0, 0, 0, np.abs(r - r_pred), gammam * dl_t

    cell_idx = int(np.floor(r_k_1_2 / r_step + 0.5))
    local_coord = r_k_1_2 / r_step - cell_idx

    e_x, e_y, e_z, b_y, b_z = _interpolate_fields(
        cell_idx, local_coord, E_r, E_f, E_z, B_f, B_z)
    e_x += _interpolate_noisereductor(noise_amplitude, cell_idx, local_coord, r_step)

    # Apply Lorentz force
    d_px = q * dl_t * (gammam * e_x + p_yo * b_z - p_zo * b_y)
    d_py = q * dl_t * (gammam * e_y - p_zo * e_y - p_xo * b_z)
    d_pz = q * dl_t * (gammam * e_z + p_xo * b_y + p_yo * e_y)

    # Predict half-step momenta
    p_x = p_xo + d_px / 2
    p_y = p_yo + d_py / 2
    p_z = p_zo + d_pz / 2

    # Correct gamma and step length
    gammam = np.sqrt(m ** 2 + p_x ** 2 + p_y ** 2 + p_z ** 2)
    if gammam == p_z or not np.isfinite(gammam):
        return max_radius / 2, 0, 0, 0, 0, 0, 0
    dl_t = xi_step_p / (gammam - p_z)

    # Apply Lorentz force
    d_px = q * dl_t * (gammam * e_x + p_y * b_z - p_z * b_y)
    d_py = q * dl_t * (gammam * e_y - p_z * e_y - p_x * b_z)
    d_pz = q * dl_t * (gammam * e_z + p_x * b_y + p_y * e_y)

    p_x = p_xo + d_px / 2
    p_y = p_yo + d_py / 2
    dx = p_x * dl_t
    dy = p_y * dl_t

    p_x = p_xo + d_px
    p_y = p_yo + d_py
    p_z = p_zo + d_pz

    # Transform back to cylindrical geometry
    x_new = r + dx
    r_new = np.sqrt(x_new ** 2 + dy ** 2)
    path = max(np.abs(r - r_new), np.abs(r - r_pred), np.abs(r_new - r_pred))
    if r_new != 0:
        p_r = (p_x * x_new + p_y * dy) / r_new
        p_f = (p_y * x_new - p_x * dy) / r_new
    else:
        p_r = p_x
        p_f = 0

    # Correct particle position if it reaches boundary or crosses the axis
    if r_new < 0:
        r_new *= -1
        p_r *= -1
    elif r_new > max_radius:
        r_new = 2 * max_radius - r_new
        p_r = p_f = p_z = 0

    if not np.isfinite(p_z) or not np.isfinite(p_r) or not np.isfinite(p_f) \
            or r_new > max_radius or r_new < 0:
        return max_radius / 2, 0, 0, 0, 0, 0, 0

    # xi_step_p was too big, particle is out of window after correction
    if r_k_1_2 < 0 or r_k_1_2 > max_radius:
        return max_radius / 2, 0, 0, 0, 0, path, gammam * dl_t

    return r_new, p_r, p_f, p_z, q, path, gammam * dl_t


@njit(parallel=True, cache=True, nogil=True)
def _move_particles_with_substepping(E_r, E_f, E_z, B_f, B_z,
                                     r_array, p_r_array, p_f_array, p_z_array,
                                     q_array, m_array, age_array,
                                     noise_amplitude, r_step, xi_step_p,
                                     max_radius):
    """Move particles in fields, correct steps if necessary."""
    for j in prange(r_array.size):
        xi_particle = 0
        xi_step_cur = xi_step_p

        r = r_array[j]
        q = q_array[j]; m = m_array[j]
        p_r = p_r_array[j]; p_f = p_f_array[j]; p_z = p_z_array[j]

        while xi_particle <= 0.99999 * xi_step_p:
            if q == 0:
                break
            r, p_r, p_f, p_z, q, path, age = _move_one_particle(
                E_r, E_f, E_z, B_f, B_z, r, p_r, p_f, p_z, q, m, noise_amplitude,
                r_step, xi_step_cur, max_radius,
            )
            if path >= r_step and xi_step_cur / xi_step_p > 1.e-4:
                # xi_step_cur is too big, substepping
                xi_step_cur /= 2
                # get particle from last good state
                r = r_array[j]
                p_r = p_r_array[j]
                p_f = p_f_array[j]
                p_z = p_z_array[j]
                q = q_array[j]
            else:
                xi_particle += xi_step_cur
                # save as good state
                r_array[j] = r
                p_r_array[j] = p_r
                p_f_array[j] = p_f
                p_z_array[j] = p_z
                q_array[j] = q
                age_array[j] -= age

        r_array[j] = r
        p_r_array[j] = p_r
        p_f_array[j] = p_f
        p_z_array[j] = p_z
        q_array[j] = q


def get_plasma_particles_mover(config: Config):
    grid_step_size = config.getfloat('window-width-step-size')
    max_radius     = config.getfloat('window-width')

    # Move particles one D_XIP step forward
    def move_particles(fields, particles_prev, noise_amplitude, xi_step_size):
        particles = particles_prev.copy()

        _move_particles_with_substepping(
            fields.E_r, fields.E_f, fields.E_z, fields.B_f, fields.B_z,
            particles.r, particles.p_r, particles.p_f, particles.p_z,
            particles.q, particles.m, particles.age,
            noise_amplitude, grid_step_size, xi_step_size, max_radius)
        
        return particles
    
    return move_particles
