import numpy as np
import functools

from lcode2dPy.alt_beam_generator.eshape import EShape
from lcode2dPy.alt_beam_generator.rshape import RShape
from lcode2dPy.alt_beam_generator.xishape import XiShape


class BeamShape:
    def __init__(self, current=0.01, particles_in_layer=2000, rng_seed=1):
        self.current = current
        self.particles_in_layer = particles_in_layer
        self.segments = []
        self.rng_seed = rng_seed
        self.rigid = False

    def add_segment(self, beam_segment):
        self.segments.append(beam_segment)
        beam_segment.set_beam_shape(self)

    def get_segment(self, xi):
        segment_start = 0
        for segment in self.segments:
            if 0 < segment_start - xi < segment.length:
                return segment, segment_start
            segment_start -= segment.length
        return None, 0

    @property
    def total_length(self):
        return functools.reduce(lambda x, y: x + y.length, self.segments, 0)

    def initial_current(self, xi):
        segment, segment_start = self.get_segment(xi)
        if segment is None:
            return 0
        dxi = segment_start - xi
        return segment.initial_current(dxi)


class BeamSegmentShape:
    def __init__(self):
        self.beam_shape = None  # Beam shape owning this segment
        self.length = 5.01 # 2 * np.pi
        self.ampl = 5.
        self.xishape = XiShape.get_shape('cos')
        self.radius = 1.
        self.energy = 1000.
        self.rshape = RShape.get_shape('gaussian')
        self.angspread = 1.0e-5
        self.angshape = XiShape.get_shape('l')
        self.espread = 0
        self.eshape = EShape.get_shape('monoenergetic')
        self.mass_charge_ratio = 1  # m/q in manual

    def set_beam_shape(self, beam_shape):
        self.beam_shape = beam_shape

    def initial_sigma_impulse(self, dxi):
        return self.angspread * self.energy * self.angshape.value(dxi,
                                                                  self.length)

    # Returns (r_b, p_br, M_b)
    def get_r_values2d(self, random, dxi, r_max, size):
        return self.rshape.values2d(random, self.radius,
                                    self.initial_sigma_impulse(dxi), r_max, size)
    
        # Returns (x, y, p_x, p_y)
    def get_r_values3d(self, random, dxi, r_max, size):
        return self.rshape.values3d(random, self.radius,
                                    self.initial_sigma_impulse(dxi), r_max, size)

    def get_pz(self, random, dxi, size):
        return self.eshape.value(random, self.energy, self.espread, dxi,
                                 self.length, size)

    @property
    def current(self):
        if self.beam_shape is None:
            return 0
        return self.ampl * self.beam_shape.current

    def initial_current(self, dxi):
        if dxi < 0 or dxi > self.length:
            return 0
        return self.current * self.xishape.value(dxi, self.length)
