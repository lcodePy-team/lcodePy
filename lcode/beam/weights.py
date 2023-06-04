import numba as nb
import numpy as np


@nb.njit(cache=True)
def add_at(out, idx, value):
    for i in np.arange(len(idx)):
        out[idx[i]] += value[i]


@nb.njit()
def particles_weights(r, xi, xi_end, r_step, xi_step_p):
    j = np.floor(r / r_step).astype(np.int_)
    dxi = (xi_end - xi) / xi_step_p + 1
    dr = r / r_step - j
    a0_xi = dxi
    a1_xi = (1 - dxi)
    a0_r = dr
    a1_r = (1 - dr)
    a00 = a0_xi * a0_r
    a01 = a0_xi * a1_r
    a10 = a1_xi * a0_r
    a11 = a1_xi * a1_r
    return j, a00, a01, a10, a11


@nb.njit(cache=True)
def single_particle_weights(r, xi, xi_end, r_step, xi_step_p):
    j = int(np.floor(r / r_step))
    dxi = (xi_end - xi) / xi_step_p + 1
    dr = r / r_step - j
    a1_xi = (1 - dxi)
    a0_xi = dxi
    a1_r = (1 - dr)
    a0_r = dr
    a00 = a0_xi * a0_r
    a01 = a0_xi * a1_r
    a10 = a1_xi * a0_r
    a11 = a1_xi * a1_r
    return j, a00, a01, a10, a11


@nb.njit(cache=True)
def deposit_particles(value, out0, out1, j, a00, a01, a10, a11):
    add_at(out0, j + 0, a00 * value)
    add_at(out0, j + 1, a01 * value)
    add_at(out1, j + 0, a10 * value)
    add_at(out1, j + 1, a11 * value)


@nb.njit(cache=True)
def interpolate_particle(value0, value1, j, a00, a01, a10, a11):
    return a00 * value0[j + 0] \
           + a01 * value0[j + 1] \
           + a10 * value1[j + 0] \
           + a11 * value1[j + 1]


@nb.njit
def particle_fields(r_vec, E_r_k_1, E_f_k_1, E_z_k_1, B_f_k_1, B_z_k_1, E_r_k,
                    E_f_k, E_z_k, B_f_k, B_z_k, xi_k_1, r_step, xi_step_p):
    r = np.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2)
    j, a00, a01, a10, a11 = single_particle_weights(r, r_vec[2], xi_k_1, r_step, xi_step_p)
    e_x = interpolate_particle(E_r_k, E_r_k_1, j, a00, a01, a10, a11)
    e_y = interpolate_particle(E_f_k, E_f_k_1, j, a00, a01, a10, a11)
    e_z = interpolate_particle(E_z_k, E_z_k_1, j, a00, a01, a10, a11)
    b_x = -e_y  # Due to symmetry
    b_y = interpolate_particle(B_f_k, B_f_k_1, j, a00, a01, a10, a11)
    b_z = interpolate_particle(B_z_k, B_z_k_1, j, a00, a01, a10, a11)
    e_vec = np.array((e_x, e_y, e_z))
    b_vec = np.array((b_x, b_y, b_z))
    return e_vec, b_vec
