"""Module for calculator of one full step along xi."""
from lcode2dPy.config.config import Config
from lcode2dPy.plasma3d_gpu.data import GPUArrays
from lcode2dPy.plasma3d_gpu.fields import FieldComputer
from lcode2dPy.plasma3d_gpu.move import ParticleMover
from lcode2dPy.plasma3d_gpu.rhoj import RhoJComputer

class Plane2d3vPlasmaSolver(object):
    def __init__(self, config: Config):
        self.FComputer = FieldComputer(config)
        self.PMover = ParticleMover(config)
        self.CComputer = RhoJComputer(config)

    # Perfoms one full step along xi.
    # To understand the numerical scheme, read values as following:
    # *_prev = * on the previous xi step, an index number = k
    # *_half = * on the halfstep, an index number = k + 1/2
    # *_full = * on the next xi step (fullstep), an index number = k + 1
    # *_prevprev = * on the xi step with an index number k - 1
    def step_dxi(
        self, particles_prev: GPUArrays, fields_prev: GPUArrays,
        currents_prev: GPUArrays, const_arrays: GPUArrays,
        rho_beam_full, rho_beam_prev
    ):
        particles_full = self.PMover.move_particles_wo_fields(particles_prev)


        particles_full = self.PMover.move_particles(
            fields_prev, particles_prev, particles_full
        )
        currents_full = self.CComputer.compute_rhoj(
            particles_full, const_arrays
        )

        _, fields_half = self.FComputer.compute_fields(
            fields_prev, fields_prev, const_arrays, rho_beam_full, rho_beam_prev,
            currents_prev, currents_full
        )


        particles_full = self.PMover.move_particles(
            fields_half, particles_prev, particles_full
        )
        currents_full = self.CComputer.compute_rhoj(
            particles_full, const_arrays
        )

        fields_full, fields_half = self.FComputer.compute_fields(
            fields_half, fields_prev, const_arrays, rho_beam_full, rho_beam_prev,
            currents_prev, currents_full
        )

        particles_full = self.PMover.move_particles(
            fields_half, particles_prev, particles_full
        )
        currents_full = self.CComputer.compute_rhoj(
            particles_full, const_arrays
        )

        return particles_full, fields_full, currents_full
