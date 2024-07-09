"""Module for calculator of one full step along xi."""
from ..config.config import Config
from ..plasma3d.rhoj import get_rhoj_computer
from ..plasma3d.fields import get_field_computer
from ..plasma3d.move import get_plasma_particles_mover
from ..plasma3d.noise_filter import get_noise_filter


class Plane2d3vPlasmaSolver(object):
    def __init__(self, config: Config):
        self.xi_step = config.getfloat('xi-step')
        self.declustering_enabled = config.getbool('declustering-enabled')
        self.substepping_sensitivity = config.getfloat('substepping-sensitivity')
        self.substepping_max_depth = config.getint('substepping-depth')
        self.xp = config.xp
        
        self.compute_rhoj = get_rhoj_computer(config)
        self.compute_fields = get_field_computer(config)
        self.move_particles_wo_fields, self.move_particles = \
            get_plasma_particles_mover(config)

        if self.declustering_enabled:
            self.noise_filter = get_noise_filter(config)
        else:
            self.noise_filter = None


    # Perfoms one full step along xi.
    # To understand the numerical scheme, read values as following:
    # *_prev = * on the previous xi step, an index number = k
    # *_half = * on the halfstep, an index number = k + 1/2
    # *_full = * on the next xi step (fullstep), an index number = k + 1
    # *_prevprev = * on the xi step with an index number k - 1
    def step_dxi(self, particles_prev, fields_prev, currents_prev, 
                 const_arrays, rho_beam_full, rho_beam_prev):

        xp = self.xp
        dxi = self.xi_step
        substeps = 0
        substepping_depth = 0
        substepping_state = [0] * (self.substepping_max_depth + 1)

        while True:
            particles_full = self.move_particles_wo_fields(
                dxi, const_arrays, particles_prev
            )

            particles_full = self.move_particles(
                dxi, const_arrays, 
                fields_prev, particles_prev, particles_full
            )

            currents_full = self.compute_rhoj(const_arrays, particles_full)
            
            charge_move = dxi * xp.abs(currents_full.jz).max()
            need_substepping = charge_move > self.substepping_sensitivity
            
            if (need_substepping  
                    and substepping_depth < self.substepping_max_depth):
                substepping_depth += 1
                substepping_state[substepping_depth] = 10
                dxi /= 10
                continue

            _, fields_half = self.compute_fields(
                dxi, const_arrays, 
                fields_prev, fields_prev, 
                rho_beam_prev, rho_beam_full,
                currents_prev, currents_full
            )

            particles_full = self.move_particles(
                dxi, const_arrays, 
                fields_half, particles_prev, particles_full
            )

            currents_full = self.compute_rhoj(const_arrays, particles_full)

            fields_full, fields_half = self.compute_fields(
                dxi, const_arrays, 
                fields_prev, fields_half, 
                rho_beam_prev, rho_beam_full,
                currents_prev, currents_full
            )

            particles_full = self.move_particles(
                dxi, const_arrays, 
                fields_half, particles_prev, particles_full
            )

            # Here we perform noise filtering after the end of the movement:
            if self.declustering_enabled:
                particles_full["electrons"] = self.noise_filter(
                    particles_prev["electrons"], particles_full["electrons"])

            currents_full = self.compute_rhoj(const_arrays, particles_full)

            substeps += 1
            while (substepping_depth > 0 
                   and substepping_state[substepping_depth] == 1):
                substepping_state[substepping_depth] = 0
                substepping_depth -= 1
                dxi *= 10
            substepping_state[substepping_depth] -= 1
            if substepping_depth == 0:
                break

            fields_prev = fields_full
            particles_prev = particles_full
            currents_prev = currents_full

        return particles_full, fields_full, currents_full
