import numpy as np
from numba import njit

from .fields import FieldComputer
from .move import ParticleMover
from .rhoj import RhoJComputer


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
    def __init__(self, config):
        self.fields = FieldComputer(config)
        self.xi_step_p = config.getfloat('xi-step')
        self.particles_mover = ParticleMover(config)
        self.currents_computer = RhoJComputer(config)
        self.rho_beam_arr = []
        self.substepping_sensitivity = config.getfloat('substepping-sensitivity')
        self.substepping_max_depth = config.getint('substepping-depth')
        self.path_lim = config.getfloat('trapped-path-limit')
        self.noisereductor_enabled = config.getbool('noise-reductor-enabled')
        self.corrector_steps = config.getint('corrector-steps')

    # Performs one full step along xi
    def step_dxi(self, particles, fields, currents, rho_beam):
        substeps = 0
        substepping_depth = 0
        step_begin = True
        substepping_state = np.zeros(self.substepping_max_depth + 1)
        xi_step_p = self.xi_step_p
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
            particles_new = self.particles_mover.move_particles(
                fields, particles,
                noise_amplitude(currents.rho, self.noisereductor_enabled),
                xi_step_p
            )
            currents_new = self.currents_computer.compute_rhoj(particles_new)
            charge_move = np.abs(xi_step_p * currents_new.j_z).max()
            need_substepping = charge_move > self.substepping_sensitivity

            if need_substepping and substepping_depth < self.substepping_max_depth:
                substepping_depth += 1
                substepping_state[substepping_depth] = 10
                xi_step_p /= 10
                step_begin = False
                continue

            for _ in np.arange(self.corrector_steps):
                fields_new, fields_average = self.fields.compute_fields(
                    fields_new, fields, rho_beam,
                    currents, currents_new,
                    xi_step_p,
                )
                particles_new = self.particles_mover.move_particles(
                    fields_average, particles,
                    noise_amplitude(currents_new.rho, self.noisereductor_enabled),
                    xi_step_p
                )
                currents_new = self.currents_computer.compute_rhoj(particles_new)
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

        return particles_new, fields_new, currents_new, substeps
