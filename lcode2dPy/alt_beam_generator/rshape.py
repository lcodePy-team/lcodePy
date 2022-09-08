import numpy as np


class RShape:
    shapes = {}

    # Returns (r_b, p_br, M_b)
    def values2d(
        self, random, radius, sigma_impulse, r_max, size, xshift, yshift
    ):
        raise NotImplementedError
    
    # Returns (x, y, p_x, p_y)
    def values3d(
        self, random, radius, sigma_impulse, r_max, size, xshift, yshift
    ):
        raise NotImplementedError

    @classmethod
    def register_shape(cls, names, shape):
        for name in names:
            if name in cls.shapes:
                raise KeyError
            cls.shapes[name] = shape

    @classmethod
    def register_shape_func(cls, names, func2d, func3d):
        for name in names:
            if name in cls.shapes:
                raise KeyError
            cls.shapes[name] = FuncRShape(func2d, func3d)

    @classmethod
    def get_shape(cls, name):
        return cls.shapes[name]


class FuncRShape(RShape):
    def __init__(self, func2d, func3d):
        self.func2d = func2d
        self.func3d = func3d

    # Returns (r_b, p_br, M_b)
    def values2d(
        self, random, radius, sigma_impulse, r_max, size, xshift, yshift
    ):
        return self.func2d(
            random, radius, sigma_impulse, r_max, size, xshift, yshift
        )
    
    # Returns (r_b, p_br, M_b)
    def values3d(
        self, random, radius, sigma_impulse, r_max, size, xshift, yshift
    ):
        return self.func3d(
            random, radius, sigma_impulse, r_max, size, xshift, yshift
        )


def gauss2d(random, radius, sigma_impulse, r_max, size, xshift, yshift):
    a = 1 - np.exp(-r_max ** 2 / (2 * radius ** 2))
    r_b = radius * np.sqrt(-2 * np.log(1 - random.random_sample(size) * a))
    p_br = random.normal(0, sigma_impulse, size)
    p_bf = random.normal(0, sigma_impulse, size)
    M_b = r_b * p_bf
    return r_b, p_br, M_b

def gauss3d(random, radius, sigma_impulse, r_max, size, xshift, yshift):
    a = 1 - np.exp(-r_max ** 2 / (2 * radius ** 2))
    r = radius * np.sqrt(-2 * np.log(1 - random.random_sample(size) * a))
    phi = random.uniform(0, 2 * np.pi, size)
    x = r * np.cos(phi) + xshift
    y = r * np.sin(phi) + yshift
    p_x = random.normal(0, sigma_impulse, size)
    p_y = random.normal(0, sigma_impulse, size)
    return x, y, p_x, p_y


RShape.register_shape_func(('gaussian', 'g'), gauss2d, gauss3d)