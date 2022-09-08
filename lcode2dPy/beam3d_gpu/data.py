import cupy as cp


particle_dtype3d = cp.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
                             ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
                             ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'f8')])

# A new class for BeamParticles that is similar
# to GPUArrays in plasma3d_gpu.data.
# TODO: Use only one type of classes, we don't
#       really need BeamParticles.

class BeamParticles:
    def __init__(self, size:int=0):
        """
        Create a new empty array of beam particles. Can be used both as
        a whole beam particles array and as a layer of beam particles.
        """
        self.size = size

        self.xi = cp.zeros(size,     dtype=cp.float64)
        self.x = cp.zeros(size,      dtype=cp.float64)
        self.y = cp.zeros(size,      dtype=cp.float64)
        self.px = cp.zeros(size,     dtype=cp.float64)
        self.py = cp.zeros(size,     dtype=cp.float64)
        self.pz = cp.zeros(size,     dtype=cp.float64)
        self.q_m = cp.zeros(size,    dtype=cp.float64)
        self.q_norm = cp.zeros(size, dtype=cp.float64)
        self.id = cp.zeros(size,     dtype=cp.int64)
        self.dt = cp.zeros(size,     dtype=cp.float64)
        self.remaining_steps = cp.zeros(size,
                                     dtype=cp.int64)

    def init_generated(self, beam_array: particle_dtype3d):
        self.xi = cp.array(beam_array['xi'])
        self.x = cp.array(beam_array['x'])
        self.y = cp.array(beam_array['y'])
        self.px = cp.array(beam_array['px'])
        self.py = cp.array(beam_array['py'])
        self.pz = cp.array(beam_array['pz'])
        self.q_m = cp.array(beam_array['q_m'])
        self.q_norm = cp.array(beam_array['q_norm'])
        self.id = cp.array(beam_array['id'])

        self.dt = cp.zeros_like(self.q_norm, dtype=cp.float64)
        self.remaining_steps = cp.zeros_like(self.id, dtype=cp.int64)
    
        self.size = len(self.dt)

    def load(self, *args, **kwargs):
        with cp.load(*args, **kwargs) as loaded:
            self.size = loaded['xi'].size
            self.xi = loaded['xi']
            self.x = loaded['x']
            self.y = loaded['y']
            self.px = loaded['px']
            self.py = loaded['py']
            self.pz = loaded['pz']
            self.q_m = loaded['q_m']
            self.q_norm = loaded['q_norm']
            self.id = loaded['id']
            self.dt = cp.zeros(self.size, dtype=cp.float64)
            self.remaining_steps = cp.zeros(self.size, dtype=cp.int64)

    def save(self, *args, **kwargs):
        cp.savez_compressed(*args, **kwargs,
                            xi = self.xi,
                            x = self.x,
                            y = self.y,
                            px = self.px,
                            py = self.py,
                            pz = self.pz,
                            q_m = self.q_m,
                            q_norm = self.q_norm,
                            id = self.id)

    # Essentials for beam layer calculations #

    def xi_sorted(self):
        """
        Sort beam particles along xi axis.
        """
        sort_idxes = cp.argsort(-self.xi)

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
        new_beam_layer = BeamParticles(indexes_arr.size)

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
    new_b_layer = BeamParticles(b_layer_1.size + b_layer_2.size)
    # TODO: The same task as for self.get_sublayer()

    new_b_layer.xi =     cp.concatenate((b_layer_1.xi, b_layer_2.xi))
    new_b_layer.x =      cp.concatenate((b_layer_1.x, b_layer_2.x))
    new_b_layer.y =      cp.concatenate((b_layer_1.y, b_layer_2.y))
    new_b_layer.px =     cp.concatenate((b_layer_1.px, b_layer_2.px))
    new_b_layer.py =     cp.concatenate((b_layer_1.py, b_layer_2.py))
    new_b_layer.pz =     cp.concatenate((b_layer_1.pz, b_layer_2.pz))
    new_b_layer.q_m =    cp.concatenate((b_layer_1.q_m, b_layer_2.q_m))
    new_b_layer.q_norm = cp.concatenate((b_layer_1.q_norm, b_layer_2.q_norm))
    new_b_layer.id =     cp.concatenate((b_layer_1.id, b_layer_2.id))
    new_b_layer.dt =     cp.concatenate((b_layer_1.dt, b_layer_2.dt))
    new_b_layer.remaining_steps = cp.concatenate((b_layer_1.remaining_steps,
                                            b_layer_2.remaining_steps))

    return new_b_layer

#TODO: The BeamParticles class makes jitting harder. And we don't really need
#      this class. Get rid of it.
