import functools

from .beam_segment_shape import BeamSegmentShape

class BeamShape:
    def __init__(self, **beam_shape_params):
        self.current = beam_shape_params['current']
        self.particles_in_layer = beam_shape_params['particles_in_layer']
        self.rng_seed = beam_shape_params['rng_seed']

        self.segments: list[BeamSegmentShape] = []
        # self.rigid = False

    def add_segment(self, beam_segment: BeamSegmentShape):
        """
        Adds a beam segment to the beam shape that describes the beam that is
        later created in beam_generator.py.
        """
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
