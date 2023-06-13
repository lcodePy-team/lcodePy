import numpy as np
from numba import njit

from ..config.config import Config
from .fields import get_field_computer
from .move import get_plasma_particles_mover
from .rhoj import get_rhoj_computer


@njit
def noise_amplitude(rho, enabled):
    if not enabled:
        return np.zeros_like(rho)
    noise_ampl = np.zeros_like(rho)
    noise_ampl[1:-1] = 2 * rho[1:-1] - rho[2:] - rho[:-2]
    noise_ampl[-1] = -noise_ampl[-2]
    noise_ampl /= 4
    return noise_ampl


class CylindricalPlasmaSolver(object):
    def __init__(self, config: Config):
        self.compute_fields = get_field_computer(config)
        self.move_particles = get_plasma_particles_mover(config)
        self.compute_rhoj   = get_rhoj_computer(config)

        self.rho_beam_arr = []

        self.xi_step_size = config.getfloat('xi-step')
        self.substepping_sensitivity = config.getfloat('substepping-sensitivity')
        self.substepping_max_depth = config.getint('substepping-depth')
        self.path_lim = config.getfloat('trapped-path-limit')
        self.noisereductor_enabled = config.getbool('noise-reductor-enabled')
        self.corrector_steps = config.getint('corrector-steps')

    # Performs one full step along xi
    def step_dxi(self, particles, fields, currents, pl_const_arrays, 
                 rho_beam, rho_beam_prev):
        substeps = 0
        substepping_depth = 0
        step_begin = True
        substepping_state = np.zeros(self.substepping_max_depth + 1)
        xi_step_p = self.xi_step_size
        fields_new = fields

        while True:
            if step_begin and self.path_lim != 0:
                captured_mask = np.logical_and(
                    particles.q != 0, particles.age <= 0,
                )
                particles.q[captured_mask] = 0
                particles.age[captured_mask] = 0
            step_begin = True

            # Predictor step
            particles_new = self.move_particles(
                fields, particles,
                noise_amplitude(currents.rho, self.noisereductor_enabled),
                xi_step_p)

            currents_new = self.compute_rhoj(particles_new)
            charge_move = np.abs(xi_step_p * currents_new.j_z).max()
            need_substepping = charge_move > self.substepping_sensitivity

            if need_substepping and substepping_depth < self.substepping_max_depth:
                substepping_depth += 1
                substepping_state[substepping_depth] = 10
                xi_step_p /= 10
                step_begin = False
                continue

            for _ in np.arange(self.corrector_steps):
                fields_new, fields_average = self.compute_fields(
                    fields_new, fields, rho_beam,
                    currents, currents_new,
                    xi_step_p,
                )
                particles_new = self.move_particles(
                    fields_average, particles,
                    noise_amplitude(currents_new.rho, self.noisereductor_enabled),
                    xi_step_p)

                currents_new = self.compute_rhoj(particles_new)
            substeps += 1
            while substepping_depth > 0 and substepping_state[substepping_depth] == 1:
                substepping_state[substepping_depth] = 0
                substepping_depth -= 1
                xi_step_p *= 10
            substepping_state[substepping_depth] -= 1
            if substepping_depth == 0:
                break
            fields = fields_new
            particles = particles_new
            currents = currents_new

        return particles_new, fields_new, currents_new
