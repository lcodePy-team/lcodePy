import numba as nb
import numpy as np

from lcode2dPy.beam.weights import (
    deposit_particles,
    particle_fields,
    particles_weights,
)


@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64)], cache=True)
def beam_substepping_step(q_m, p_z, substepping_energy):
    dt = 1.0
    gamma_mass = np.sqrt(1 / q_m ** 2 + p_z ** 2)
    max_dt = np.sqrt(gamma_mass / substepping_energy)
    while dt > max_dt:
        dt /= 2
    return dt


@nb.njit
def cross_nb(vec1, vec2):
    result = np.zeros_like(vec1)
    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return result


@nb.njit
def in_layer(r_vec, xi_k_1):
    return xi_k_1 <= r_vec[2]


@nb.njit
def is_lost(r_vec, r_max):
    return r_vec[0] ** 2 + r_vec[1] ** 2 >= r_max ** 2


@nb.njit
def beam_to_vec(beam, idx):
    p_x = beam.p_r[idx]
    p_y = beam.M[idx] / beam.r[idx]
    p_vec = np.array((p_x, p_y, beam.p_z[idx]))
    r_vec = np.array((beam.r[idx], 0, beam.xi[idx]))
    return p_vec, r_vec, beam.remaining_steps[idx]


@nb.njit
def vec_to_beam(beam, idx, r_vec, p_vec, steps_left, lost, magnetic_field):
    x = r_vec[0]
    y = r_vec[1]
    beam.r[idx] = np.sqrt(x ** 2 + y ** 2)
    beam.xi[idx] = r_vec[2]
    beam.p_r[idx] = (x * p_vec[0] + y * p_vec[1]) / beam.r[idx]
    beam.p_z[idx] = p_vec[2]
    if magnetic_field == 0:
        beam.M[idx] = x * p_vec[1] - y * p_vec[0]
    beam.remaining_steps[idx] = steps_left
    if lost:
        beam.id[idx] = -np.abs(beam.id[idx])


def configure_move_beam_slice(config):
    r_step = float(config.get('r-step'))
    xi_step_p = float(config.get('xi-step'))
    max_radius = float(config.get('window-width'))
    lost_boundary = max(0.9 * max_radius, max_radius - 1)
    magnetic_field = float(config.get('magnetic-field'))
    locals_spec = {'r_step': nb.float64,
                   'xi_step': nb.float64,
                   'lost_boundary': nb.float64,
                   'magnetic_field': nb.float64,}

    # Moves particles as far as possible on its xi layer
    @nb.njit(locals=locals_spec)
    def move_beam_slice(beam_slice, xi_layer, fields_after_slice, fields_before_slice):
        xi_end = xi_layer * -xi_step_p
        if beam_slice.size == 0:
            return
        for idx in np.arange(beam_slice.size):
            q_m = beam_slice.q_m[idx]
            dt = beam_slice.dt[idx]
            lost = False

            # Initial impulse and position vectors
            p_vec, r_vec, steps = beam_to_vec(beam_slice, idx)
            while steps > 0:
                # Compute approximate position of the particle in the middle of the step
                gamma_mass = np.sqrt((1 / q_m) ** 2 + np.sum(p_vec ** 2))
                r_vec_half_step = r_vec + dt / 2 * p_vec / gamma_mass
                # Add time shift correction (dxi = (v_z - c)*dt)
                r_vec_half_step[2] -= dt / 2

                if not in_layer(r_vec_half_step, xi_end):
                    break
                if is_lost(r_vec_half_step, lost_boundary):
                    # Particle hit the wall and is now lost
                    #beam_slice.mark_lost(idx)
                    break

                # Interpolate fields and compute new impulse
                (e_vec, b_vec) = particle_fields(
                    r_vec_half_step,
                    fields_after_slice,
                    fields_before_slice,
                    xi_end,
                    r_step,
                    xi_step_p,
                )

                p_vec_half_step = p_vec + dt / 2 * np.sign(q_m) * (e_vec + cross_nb(p_vec / gamma_mass, b_vec))  # Just Lorentz

                # Compute final coordinates and impulses
                gamma_mass = np.sqrt((1 / q_m) ** 2 + np.sum(p_vec_half_step ** 2))
                r_vec += dt * p_vec_half_step / gamma_mass
                # Add time shift correction (dxi = (v_z - c)*dt)
                r_vec[2] -= dt
                p_vec = 2 * p_vec_half_step - p_vec
                steps -= 1

                if is_lost(r_vec, lost_boundary):
                    # Particle hit the wall and is now lost
                    #beam_slice.mark_lost(idx)
                    break
            vec_to_beam(beam_slice, idx, r_vec, p_vec, steps, lost, magnetic_field)

    return move_beam_slice


def layout_beam_slice(
    beam_slice, xi_layer_idx, prev_rho_layout, r_step, xi_step_p,
):
    n_cells = prev_rho_layout.size
    rho_layout = np.zeros_like(prev_rho_layout)
    xi_end = xi_layer_idx * -xi_step_p
    if beam_slice.size != 0:
        j, a00, a01, a10, a11 = particles_weights(
            beam_slice.r, beam_slice.xi, xi_end, r_step, xi_step_p,
        )
        deposit_particles(beam_slice.q_norm, prev_rho_layout, rho_layout, j, a00, a01, a10, a11)

    prev_rho_layout, rho_layout = rho_layout, prev_rho_layout
    rho_layout /= r_step ** 2
    rho_layout[0] *= 6
    rho_layout[1:] /= np.arange(1, n_cells)

    return rho_layout, prev_rho_layout


@nb.njit
def init_substepping(beam_slice, time_step, substepping_energy):
    mask = beam_slice.dt == 0
    dt = beam_substepping_step(beam_slice.q_m[mask], beam_slice.p_z[mask], substepping_energy)
    steps = (1 / dt).astype(np.int_)
    beam_slice.dt[mask] = dt * time_step
    beam_slice.remaining_steps[mask] = steps


def beam_slice_mover(config):
    time_step = float(config.get('time-step'))
    substepping_energy = config.getfloat('beam-substepping-energy')
    move_particles = configure_move_beam_slice(config)

    @nb.njit
    def move_beam_slice(
        beam_slice, xi_layer_idx, fields_after_slice, fields_before_slice,
    ):
        if beam_slice.size == 0:
            return
        if substepping_energy == 0:
            # Particles with dt != 0 came from the previous time step and
            # need no initialization
            beam_slice.dt[beam_slice.dt == 0] = time_step
        else:
            init_substepping(beam_slice, time_step, substepping_energy)
        move_particles(
            beam_slice, xi_layer_idx, fields_after_slice, fields_before_slice,
        )

    return move_beam_slice
