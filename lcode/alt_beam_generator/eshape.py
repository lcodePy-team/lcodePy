import numpy as np


class EShape:
    shapes = {}

    def value(self, random, energy, espread, dxi, l_s, size):
        raise NotImplementedError

    @classmethod
    def register_shape(cls, names, shape):
        for name in names:
            if name in cls.shapes:
                raise KeyError
            cls.shapes[name] = shape

    @classmethod
    def register_shape_func(cls, names, shape_function):
        for name in names:
            if name in cls.shapes:
                raise KeyError
            cls.shapes[name] = FuncEShape(shape_function)

    @classmethod
    def get_shape(cls, name):
        return cls.shapes[name]


class FuncEShape(EShape):
    def __init__(self, func):
        self.func = func

    def value(self, random, energy, espread, dxi, l_s, size):
        return self.func(random, energy, espread, dxi, l_s, size)


EShape.register_shape_func(('monoenergetic', 'm'),
lambda random, energy, espread, dxi, l_s, size: np.full(size, energy))
EShape.register_shape_func(('uniform', 'u'),
lambda random, energy, espread, dxi, l_s, size: random.uniform(energy, espread, size))
EShape.register_shape_func(('linear', 'l'),
lambda random, energy, espread, dxi, l_s, size: np.full(size, energy * dxi / l_s +
                                                            espread * (1 - dxi / l_s)))
EShape.register_shape_func(('gaussian', 'g'),
lambda random, energy, espread, dxi, l_s, size: random.normal(energy, espread, size))


def multienergetic_generator(n):
    return lambda random, energy, espread, dxi, l_s, size: (np.full(size, espread)
                            + random.randint(0, n) * (energy - espread) / (n - 1))


for i in range(2, 10):
    EShape.register_shape_func((str(i),), multienergetic_generator(i))