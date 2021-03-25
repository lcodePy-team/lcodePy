"""Module for calculator of one full step along xi."""
import numpy as np
import numba as nb

from lcode2dPy.plasma3d.fields import FieldComputer
from lcode2dPy.plasma3d.move import ParticleMover
from lcode2dPy.plasma3d.rhoj import RhoJComputer

class Plane2d3vPlasmaSolver(object):
    def __init__(self, config):
        self.FComputer = FieldComputer(config)
        self.PMover = ParticleMover(config)
        self.CComputer = RhoJComputer(config)

    # Perfoms one full step along xi
    def step_dxi(self, particles_prev, fields_prev, currents_prev,
                 const_arrays, rho_beam):
        particles = self.PMover.move_particles_wo_fields(particles_prev)


        particles = self.PMover.move_particles(fields_prev,
                                               particles_prev, particles)
        currents = self.CComputer.compute_rhoj(particles, const_arrays)

        _, fields_avg = self.FComputer.compute_fields(fields_prev, fields_prev,
                                               const_arrays, rho_beam,
                                               currents_prev, currents)  

        
        particles = self.PMover.move_particles(fields_avg,
                                               particles_prev, particles)
        currents = self.CComputer.compute_rhoj(particles, const_arrays)
        
        fields, fields_avg = self.FComputer.compute_fields(fields_avg, fields_prev,
                                               const_arrays, rho_beam,
                                               currents_prev, currents)
        
        particles = self.PMover.move_particles(fields_avg,
                                               particles_prev, particles)
        currents = self.CComputer.compute_rhoj(particles, const_arrays)
      
        return particles, fields, currents
