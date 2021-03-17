import numpy as np
from numba import float64, jitclass

class BeamParticles(object):
    def __init__(self, size):
        self.r = np.copy(r)
        self.p_r = np.copy(p_r)
        self.p_f = np.copy(p_f)
        self.p_z = np.copy(p_z)
        self.q = np.copy(q)
        self.m = np.copy(m)
        self.age = np.copy(age)

    def copy(self):
        return Particles(self.r, self.p_r, self.p_f, self.p_z, self.q, self.m, self.age)
    