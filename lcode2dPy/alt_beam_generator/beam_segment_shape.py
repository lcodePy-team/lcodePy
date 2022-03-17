from lcode2dPy.alt_beam_generator.eshape import EShape
from lcode2dPy.alt_beam_generator.rshape import RShape
from lcode2dPy.alt_beam_generator.xishape import XiShape

class BeamSegmentShape:
    def __init__(self, **beam_parameters):
        # Default parameters for 'test 1' beam
        self.beam_shape = None  # Beam shape owning this segment
        self.length = beam_parameters['length']
        self.ampl = beam_parameters['ampl']
        self.xishape: XiShape = XiShape.get_shape(beam_parameters['xishape'])
        self.radius = beam_parameters['radius']
        self.energy = beam_parameters['energy']
        self.rshape: RShape = RShape.get_shape(beam_parameters['rshape'])
        self.angspread = beam_parameters['angspread']
        self.angshape: XiShape = XiShape.get_shape(beam_parameters['angshape'])
        self.espread = beam_parameters['espread']
        self.eshape: EShape = EShape.get_shape(beam_parameters['eshape'])
        self.mass_charge_ratio = beam_parameters['mass_charge_ratio'] # m/q in manual

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
