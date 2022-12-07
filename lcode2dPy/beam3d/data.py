import numpy as np

from ..config.config import Config

particle_dtype3d = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
                             ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
                             ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])

# We don't really need this class. It's more convenient
# to have something like GPUArrays from plasma3d_gpu.
# A new class for BeamParticles that is similar
# to GPUArrays in plasma3d_gpu.data.
# TODO: Use only one type of classes, we don't
#       really need BeamParticles.

class BeamParticles:
    def __init__(self, xp: np, size:int=0):
        """
        Create a new empty array of beam particles. Can be used both as
        a whole beam particles array and as a layer of beam particles.
        """
        self.xp = xp
        self.size = size

        self.xi = xp.zeros(size, dtype=xp.float64)
        self.x  = xp.zeros(size, dtype=xp.float64)
        self.y  = xp.zeros(size, dtype=xp.float64)
        self.px = xp.zeros(size, dtype=xp.float64)
        self.py = xp.zeros(size, dtype=xp.float64)
        self.pz = xp.zeros(size, dtype=xp.float64)
        self.q_m = xp.zeros(size, dtype=xp.float64)
        self.q_norm = xp.zeros(size, dtype=xp.float64)
        self.id = xp.zeros(size, dtype=xp.int64)
        self.dt = xp.zeros(size, dtype=xp.float64)
        self.remaining_steps = xp.zeros(size,
                                     dtype=xp.int64)

    def init_generated(self, beam_array: particle_dtype3d):
        self.xi = self.xp.array(beam_array['xi'])
        self.x = self.xp.array(beam_array['x'])
        self.y = self.xp.array(beam_array['y'])
        self.px = self.xp.array(beam_array['px'])
        self.py = self.xp.array(beam_array['py'])
        self.pz = self.xp.array(beam_array['pz'])
        self.q_m = self.xp.array(beam_array['q_m'])
        self.q_norm = self.xp.array(beam_array['q_norm'])
        self.id = self.xp.array(beam_array['id'])

        self.dt = self.xp.zeros_like(self.q_norm, dtype=self.xp.float64)
        self.remaining_steps = self.xp.zeros_like(self.id, dtype=self.xp.int64)

        self.size = len(self.dt)

    def load(self, *args, **kwargs):
        with self.xp.load(*args, **kwargs) as loaded:
            self.size = len(loaded['xi'])
            self.xi = loaded['xi']
            self.x = loaded['x']
            self.y = loaded['y']
            self.px = loaded['px']
            self.py = loaded['py']
            self.pz = loaded['pz']
            self.q_m = loaded['q_m']
            self.q_norm = loaded['q_norm']
            self.id = loaded['id']
            self.dt = self.xp.zeros(self.size, dtype=self.xp.float64)
            self.remaining_steps = self.xp.zeros(self.size,
                                          dtype=self.xp.int64)

    def save(self, *args, **kwargs):
        self.xp.savez_compressed(
            *args, **kwargs, xi = self.xi, x = self.x, y = self.y,
            px = self.px, py = self.py, pz = self.pz, q_m = self.q_m,
            q_norm = self.q_norm, id = self.id)

    # Essentials for beam layer calculations #

    def xi_sorted(self):
        """
        Sort beam particles along xi axis.
        """
        sort_idxes = self.xp.argsort(-self.xi)

        self.xi = self.xi[sort_idxes]
        self.x = self.x[sort_idxes]
        self.y = self.y[sort_idxes]
        self.px = self.px[sort_idxes]
        self.py = self.py[sort_idxes]
        self.pz = self.pz[sort_idxes]
        self.q_m = self.q_m[sort_idxes]
        self.q_norm = self.q_norm[sort_idxes]
        self.id = self.id[sort_idxes]
        self.dt = self.dt[sort_idxes]
        self.remaining_steps = self.remaining_steps[sort_idxes]
        # self.lost = self.lost[sort_idxes]

    def get_layer(self, indexes_arr):
        """
        Return a layer with indexes from indexes_arr.
        """
        # TODO: Find a better method of getting a layer!
        #       Have a look at plasma3d_gpu.data for examples.
        new_beam_layer = BeamParticles(self.xp, indexes_arr.size)

        new_beam_layer.xi = self.xi[indexes_arr]
        new_beam_layer.x = self.x[indexes_arr]
        new_beam_layer.y = self.y[indexes_arr]
        new_beam_layer.px = self.px[indexes_arr]
        new_beam_layer.py = self.py[indexes_arr]
        new_beam_layer.pz = self.pz[indexes_arr]
        new_beam_layer.q_m = self.q_m[indexes_arr]
        new_beam_layer.q_norm = self.q_norm[indexes_arr]
        new_beam_layer.id = self.id[indexes_arr]
        new_beam_layer.dt = self.dt[indexes_arr]
        new_beam_layer.remaining_steps = self.remaining_steps[indexes_arr]
        # new_beam_layer.lost =   self.lost[indexes_arr]

        return new_beam_layer


def concatenate_beam_layers(b_layer_1: BeamParticles, b_layer_2: BeamParticles):
    """
    Concatenate two beam particles layers.
    """
    xp = b_layer_1.xp

    new_b_layer = BeamParticles(xp, b_layer_1.size + b_layer_2.size)
    # TODO: The same task as for self.get_sublayer()

    new_b_layer.xi =     xp.concatenate((b_layer_1.xi, b_layer_2.xi))
    new_b_layer.x =      xp.concatenate((b_layer_1.x, b_layer_2.x))
    new_b_layer.y =      xp.concatenate((b_layer_1.y, b_layer_2.y))
    new_b_layer.px =     xp.concatenate((b_layer_1.px, b_layer_2.px))
    new_b_layer.py =     xp.concatenate((b_layer_1.py, b_layer_2.py))
    new_b_layer.pz =     xp.concatenate((b_layer_1.pz, b_layer_2.pz))
    new_b_layer.q_m =    xp.concatenate((b_layer_1.q_m, b_layer_2.q_m))
    new_b_layer.q_norm = xp.concatenate((b_layer_1.q_norm, b_layer_2.q_norm))
    new_b_layer.id =     xp.concatenate((b_layer_1.id, b_layer_2.id))
    new_b_layer.dt =     xp.concatenate((b_layer_1.dt, b_layer_2.dt))
    new_b_layer.remaining_steps = xp.concatenate((b_layer_1.remaining_steps,
                                            b_layer_2.remaining_steps))

    return new_b_layer

#TODO: The BeamParticles class makes jitting harder. And we don't really need
#      this class. Get rid of it.
