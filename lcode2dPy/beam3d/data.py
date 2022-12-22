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

# NOTE: We know that here we are repeating the functionality of structured
#       arrays from NumPy. But as of December 2022, there are no structured
#       arrays in CuPy, so we have no other choice but to implement the
#       functinality ourselves.

class BeamParticles:
    def __init__(self, xp: np):
        """
        Create a new empty array of beam particles. Can be used both as
        a whole beam particles array and as a layer of beam particles.
        """
        self.xp = xp

        # Do we need this?
        self.xi = xp.zeros(0, dtype=xp.float64)
        self.x  = xp.zeros(0, dtype=xp.float64)
        self.y  = xp.zeros(0, dtype=xp.float64)
        self.px = xp.zeros(0, dtype=xp.float64)
        self.py = xp.zeros(0, dtype=xp.float64)
        self.pz = xp.zeros(0, dtype=xp.float64)
        self.q_m = xp.zeros(0, dtype=xp.float64)
        self.q_norm = xp.zeros(0, dtype=xp.float64)
        self.id = xp.zeros(0, dtype=xp.int64)
        self.dt = xp.zeros(0, dtype=xp.float64)
        self.remaining_steps = xp.zeros(0, dtype=xp.int64)

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

    def load(self, *args, **kwargs):
        with self.xp.load(*args, **kwargs) as loaded:
            self.xi = loaded['xi']
            self.x = loaded['x']
            self.y = loaded['y']
            self.px = loaded['px']
            self.py = loaded['py']
            self.pz = loaded['pz']
            self.q_m = loaded['q_m']
            self.q_norm = loaded['q_norm']
            self.id = loaded['id']
            self.dt = self.xp.zeros(len(loaded['id']), dtype=self.xp.float64)
            self.remaining_steps = self.xp.zeros(len(loaded['id']),
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

    def get_layer(self, indexes_arr):
        """
        Return a layer with indexes from indexes_arr.
        """
        # TODO: Find a better method of getting a layer!
        #       Have a look at plasma3d_gpu.data for examples.
        new_beam_layer = BeamParticles(self.xp)

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

        return new_beam_layer

    def cut_beam_layer(self, layer_length):
        """
        Cut off a beam layer with a set layer_length from the side of the head.
        """
        beam_layer = BeamParticles(self.xp)

        beam_layer.xi, self.xi = self.xi[:layer_length], self.xi[layer_length:]
        beam_layer.x, self.x   = self.x[:layer_length], self.x[layer_length:]
        beam_layer.y, self.y   = self.y[:layer_length], self.y[layer_length:]
        beam_layer.px, self.px = self.px[:layer_length], self.px[layer_length:]
        beam_layer.py, self.py = self.py[:layer_length], self.py[layer_length:]
        beam_layer.pz, self.pz = self.pz[:layer_length], self.pz[layer_length:]
        beam_layer.q_m, self.q_m = \
            self.q_m[:layer_length], self.q_m[layer_length:]
        beam_layer.q_norm, self.q_norm = \
            self.q_norm[:layer_length], self.q_norm[layer_length:]
        beam_layer.id, self.id = self.id[:layer_length], self.id[layer_length:]
        beam_layer.dt, self.dt = self.dt[:layer_length], self.dt[layer_length:]
        beam_layer.remaining_steps = self.remaining_steps[:layer_length]
        self.remaining_steps       = self.remaining_steps[layer_length:]

        return beam_layer, self

    def append(self, another_beam_layer):
        """
        Append another beam layer to the end of the beam layer.
        """
        self.xi = self.xp.concatenate((self.xi, another_beam_layer.xi))
        self.x  = self.xp.concatenate((self.x , another_beam_layer.x ))
        self.y  = self.xp.concatenate((self.y , another_beam_layer.y ))
        self.px = self.xp.concatenate((self.px, another_beam_layer.px))
        self.py = self.xp.concatenate((self.py, another_beam_layer.py))
        self.pz = self.xp.concatenate((self.pz, another_beam_layer.pz))
        self.q_m = self.xp.concatenate((self.q_m, another_beam_layer.q_m))
        self.q_norm = self.xp.concatenate(
            (self.q_norm, another_beam_layer.q_norm))
        self.id = self.xp.concatenate((self.id, another_beam_layer.id))
        self.dt = self.xp.concatenate((self.dt, another_beam_layer.dt))
        self.remaining_steps = self.xp.concatenate(
            (self.remaining_steps, another_beam_layer.remaining_steps))

        return self
