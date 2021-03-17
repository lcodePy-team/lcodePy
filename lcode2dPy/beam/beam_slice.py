import numba as nb
import numpy as np

particle_dtype = np.dtype([('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'), ('q_m', 'f8'),
                           ('q_norm', 'f8'), ('id', 'i8')])

_beamParticlesSliceSpec = [
    ('size', nb.int64),
    ('particles', nb.types.Array(dtype=nb.from_dtype(particle_dtype), ndim=1, layout='C')),

    ('xi', nb.float64[:]),
    ('r', nb.float64[:]),
    ('p_r', nb.float64[:]),
    ('p_z', nb.float64[:]),
    ('M', nb.float64[:]),
    ('q_m', nb.float64[:]),
    ('q_norm', nb.float64[:]),
    ('id', nb.int64[:]),

    ('dt', nb.float64[:]),
    ('remaining_steps', nb.int64[:]),
    ('status', nb.int64[:]),
    ('nlost', nb.int64),
    ('lost', nb.int64[:])
]


# jitclass is necessary to pass objects to jit-compiled functions
@nb.jitclass(spec=_beamParticlesSliceSpec)
class BeamSlice:
    def __init__(self, size, particles=None):
        self.size = size
        if particles is not None:
            self.particles = particles
        else:
            self.particles = np.zeros(size, dtype=particle_dtype)
        self.xi = self.particles['xi']
        self.r = self.particles['r']
        self.p_z = self.particles['p_z']
        self.p_r = self.particles['p_r']
        self.M = self.particles['M']
        self.q_m = self.particles['q_m']
        self.q_norm = self.particles['q_norm']
        self.id = self.particles['id']

        # Additional particle properties for substepping, not stored in beam
        self.dt = np.zeros(size, dtype=np.float64)
        self.remaining_steps = np.full(size, 1, dtype=np.int64)

        self.nlost = 0
        self.lost = np.zeros(10, dtype=np.int64)

    def swap_particles(self, i, j):
        self.particles[i], self.particles[j] = self.particles[j], self.particles[i]
        self.dt[i], self.dt[j] = self.dt[j], self.dt[i]
        self.remaining_steps[i], self.remaining_steps[j] = self.remaining_steps[j], self.remaining_steps[i]
        self.status[i], self.status[j] = self.status[j], self.status[i]

    def get_subslice(self, begin, end):
        temp_particles = self.particles[begin:end]
        sub_slice = BeamSlice(len(temp_particles), temp_particles)
        sub_slice.dt[:] = self.dt[begin:end]
        sub_slice.remaining_steps[:] = self.remaining_steps[begin:end]
        return sub_slice

    def concatenate(self, other_slice):
        self.particles = np.concatenate((
            self.particles, other_slice.particles,
        ))
        self.size = self.particles.size
        self.xi = self.particles['xi']
        self.r = self.particles['r']
        self.p_z = self.particles['p_z']
        self.p_r = self.particles['p_r']
        self.M = self.particles['M']
        self.q_m = self.particles['q_m']
        self.q_norm = self.particles['q_norm']
        self.id = self.particles['id']

        self.dt = np.concatenate((self.dt, other_slice.dt))
        self.remaining_steps = np.concatenate((
            self.remaining_steps, other_slice.remaining_steps,
        ))
        self.status = np.concatenate((self.status, other_slice.status))
        return self
