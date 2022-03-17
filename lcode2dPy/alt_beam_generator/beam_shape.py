import functools

from lcode2dPy.alt_beam_generator.beam_segment_shape import BeamSegmentShape

class BeamShape:
    def __init__(self, **beam_parameters):
        if 'current' in beam_parameters.keys():
            self.current = beam_parameters['current']
        else:
            self.current = 0.01
                
        if 'particles_in_layer' in beam_parameters.keys():
            self.particles_in_layer = beam_parameters['particles_in_layer']
        else:
            self.particles_in_layer = 2000

        if 'rng_seed' in beam_parameters.keys():
            self.rng_seed = beam_parameters['rng_seed']
        else:
            self.rng_seed = 1

        self.segments: list[BeamSegmentShape] = []
        # self.rigid = False

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
