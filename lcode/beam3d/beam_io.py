import numpy as np

from ..config.config import Config
from .data import BeamParticles


class BeamSource:
    """
    This class helps to extract a beam layer from beam particles array.
    """
    # Do we really need this class?
    def __init__(self, config: Config, beam_particles):
        # From input parameters:
        self.xp = config.xp
        self.xi_step_size = config.getfloat('xi-step')

        # Get the whole beam or a beam layer:
        if type(beam_particles) == np.ndarray:
            beam = BeamParticles(self.xp)
            beam.init_generated(beam_particles)
        else:
            beam = beam_particles

        beam.xi_sorted()
        self.beam = beam

    def get_beam_layer_to_layout(self, plasma_layer_idx):
        """
        Find all beam particles between plasma_layer_idx and
        plasma_layer_idx + 1, return them as a layer (class BeamParticles).
        """
        xi_min = - self.xi_step_size * plasma_layer_idx
        xi_max = - self.xi_step_size * (plasma_layer_idx + 1)

        array_to_search = self.beam.xi
        # Here we find the length of a layer where requisite particles lay.
        if array_to_search.size != 0:
            layer_length = self.xp.sum(
                (self.xp.asarray(xi_max) <= array_to_search) *
                (array_to_search < self.xp.asarray(xi_min)))
        else:
            layer_length = 0

        beam_layer_to_layout, self.beam = self.beam.cut_beam_layer(layer_length)
        return beam_layer_to_layout


class BeamDrain:
    """
    This class is used to store beam particles when the calculation of their
    movement ends.
    """
    def __init__(self, config: Config):
        # We create two empty BeamParticles classes. Don't really like how it
        # is done. We need to change this procces.
        self.beam_buffer = BeamParticles(config.xp)
        self.lost_buffer = BeamParticles(config.xp)

    def push_beam_slice(self, beam_layer: BeamParticles):
        """
        Add a beam layer that was moved to the beam buffer.
        """
        if beam_layer.id.size > 0:
            self.beam_buffer.append(beam_layer)

    def push_beam_lost(self, lost_layer: BeamParticles):
        """
        Add lost beam particles to the buffer of lost particles.
        """
        if lost_layer.id.size > 0:
            self.lost_buffer.append(lost_layer)
    
    def beam_slice(self):
        return self.beam_buffer
