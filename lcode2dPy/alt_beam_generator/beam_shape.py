import functools

from lcode2dPy.alt_beam_generator.beam_segment_shape import BeamSegmentShape

class BeamShape:
    def __init__(self, current=0.01, particles_in_layer=2000, rng_seed=1):
        self.current = current
        self.particles_in_layer = particles_in_layer
        self.segments: list[BeamSegmentShape] = []
        self.rng_seed = rng_seed
        self.rigid = False

    def add_segment(self, beam_segment: BeamSegmentShape):
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
