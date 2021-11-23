import numpy as np


class XiShape:
    shapes = {}

    def value(self, dxi, l_s):
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
            cls.shapes[name] = FuncXiShape(shape_function)

    @classmethod
    def get_shape(cls, name):
        return cls.shapes[name]


class FuncXiShape(XiShape):
    def __init__(self, func):
        self.func = func

    def value(self, dxi, l_s):
        return self.func(dxi, l_s)


XiShape.register_shape_func(('cos', 'c'),
                    lambda dxi, l_s: 0.5 * (1 - np.cos(2 * np.pi * dxi / l_s)))
XiShape.register_shape_func(('t',),
                    lambda dxi, l_s: dxi / l_s)
XiShape.register_shape_func(('T',),
                    lambda dxi, l_s: 1 - dxi / l_s)
XiShape.register_shape_func(('l',),
                    lambda dxi, l_s: 1)
XiShape.register_shape_func(('half-cos', 'h'),
                    lambda dxi, l_s: 0.5 * (1 - np.cos(np.pi * dxi / l_s)))
XiShape.register_shape_func(('b',),
                    lambda dxi, l_s: 0.5 * (1 + np.cos(np.pi * dxi / l_s)))
XiShape.register_shape_func(('gaussian', 'g'),
                    lambda dxi, l_s: 0.5 * np.exp(-18 * (dxi / l_s) ** 2))