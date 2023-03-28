"""Module for calculator of one full step along xi."""
from ..config.config import Config
from ..plasma3d.rhoj import get_rhoj_computer
from ..plasma3d.fields import get_field_computer
from ..plasma3d.move import get_plasma_particles_mover
from ..plasma3d.noise_filter import get_noise_filter


class Plane2d3vPlasmaSolver(object):
    def __init__(self, config: Config):
        self.compute_rhoj = get_rhoj_computer(config)
        self.compute_fields = get_field_computer(config)
        self.move_particles, self.move_particles_wo_fields = \
            get_plasma_particles_mover(config)

        self.noise_filter = get_noise_filter(config)

    # Perfoms one full step along xi.
    # To understand the numerical scheme, read values as following:
    # *_prev = * on the previous xi step, an index number = k
    # *_half = * on the halfstep, an index number = k + 1/2
    # *_full = * on the next xi step (fullstep), an index number = k + 1
    # *_prevprev = * on the xi step with an index number k - 1
    def step_dxi(
        self, particles_prev, fields_prev, currents_prev, const_arrays,
        rho_beam_full, rho_beam_prev
    ):
        particles_full = self.move_particles_wo_fields(particles_prev)

        rho_noisy = currents_prev.ro + rho_beam_prev
        particles_full = self.move_particles(
            fields_prev, particles_prev, particles_full, const_arrays, rho_noisy
        )
        currents_full = self.compute_rhoj(
            particles_full, const_arrays
        )

        _, fields_half = self.compute_fields(
            fields_prev, fields_prev, const_arrays, rho_beam_full, rho_beam_prev,
            currents_prev, currents_full
        )


        rho_noisy = (currents_full.ro + rho_beam_full +
                     currents_prev.ro + rho_beam_prev) / 2
        particles_full = self.move_particles(
            fields_half, particles_prev, particles_full, const_arrays, rho_noisy
        )
        currents_full = self.compute_rhoj(
            particles_full, const_arrays
        )

        fields_full, fields_half = self.compute_fields(
            fields_half, fields_prev, const_arrays, rho_beam_full, rho_beam_prev,
            currents_prev, currents_full
        )

        rho_noisy = (currents_full.ro + rho_beam_full +
                     currents_prev.ro + rho_beam_prev) / 2
        particles_full = self.move_particles(
            fields_half, particles_prev, particles_full, const_arrays, rho_noisy
        )
        currents_full = self.compute_rhoj(
            particles_full, const_arrays
        )

        # Here we perform noise filtering after the end of the movement:
        particles_full = self.noise_filter(particles_full)

        return particles_full, fields_full, currents_full
