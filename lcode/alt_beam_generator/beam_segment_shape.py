from .eshape import EShape
from .rshape import RShape
from .xishape import XiShape

class BeamSegmentShape:
    def __init__(self, **beam_segment_params):
        # Default parameters for 'test 1' beam
        self.beam_shape = None  # Beam shape owning this segment
        self.length = beam_segment_params['length']
        self.ampl = beam_segment_params['ampl']
        self.xishape: XiShape = XiShape.get_shape(beam_segment_params['xishape'])
        self.radius = beam_segment_params['radius']
        self.energy = beam_segment_params['energy']
        self.xshift = beam_segment_params['xshift']
        self.yshift = beam_segment_params['yshift']
        self.rshape: RShape = RShape.get_shape(beam_segment_params['rshape'])
        self.angspread = beam_segment_params['angspread']
        self.angshape: XiShape = XiShape.get_shape(beam_segment_params['angshape'])
        self.espread = beam_segment_params['espread']
        self.eshape: EShape = EShape.get_shape(beam_segment_params['eshape'])
        self.mass_charge_ratio = beam_segment_params['mass_charge_ratio'] # m/q in manual

    def set_beam_shape(self, beam_shape):
        """
        Sets a beam shape for this beam segment. Is used only in
        BeamShape.add_segment.
        """
        self.beam_shape = beam_shape

    def initial_sigma_impulse(self, dxi):
        return self.angspread * self.energy * self.angshape.value(dxi,
                                                                  self.length)

    # Returns (r_b, p_br, M_b)
    def get_r_values2d(self, random, dxi, r_max, size):
        return self.rshape.values2d(
            random, self.radius, self.initial_sigma_impulse(dxi), r_max, size,
            self.xshift, self.yshift
        )
    
        # Returns (x, y, p_x, p_y)
    def get_r_values3d(self, random, dxi, r_max, size):
        return self.rshape.values3d(
            random, self.radius, self.initial_sigma_impulse(dxi), r_max, size,
            self.xshift, self.yshift
        )

    def get_pz(self, random, dxi, size):
        return self.eshape.value(
            random, self.energy, self.espread, dxi, self.length, size
        )

    @property
    def current(self):
        if self.beam_shape is None:
            return 0
        return self.ampl * self.beam_shape.current

    def initial_current(self, dxi):
        if dxi < 0 or dxi > self.length:
            return 0
        return self.current * self.xishape.value(dxi, self.length)
