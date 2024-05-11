"""Module for setting classes for Fields, Currents and Partciles data types."""
import numpy as np

# Grouping CPU/GPU arrays, with optional transparent RAM<->GPU copying #

class Arrays:
    """
    A convenient way to group several CPU/GPU arrays and access them with a dot.
    `x = Arrays(xp=cupy, something=numpy_array, something_else=another_array)`
    will create `x` with `x.something` and `x.something_else` being CPU/GPU arrays.
    Do not add more attributes later, specify them all at construction time.
    """
    def __init__(self, xp: np, **kwargs):
        """
        Convert the keyword arguments to `cupy` arrays and assign them
        to the object attributes.
        Amounts to, e.g., `self.something = cp.asarray(numpy_array)`,
        and `self.something_else = cp.asarray(another_array)`,
        see class doctring.
        """
        self.xp = xp
        for name, array in kwargs.items():
            setattr(self, name, xp.array(array)) # or asarray?

    def copy(self):
        """
        Create an indentical copy of the group of `cupy` arrays.
        """
        return Arrays(**self.__dict__)

    def save(self, *args, **kwargs):
        data = self.__dict__.copy()
        data.pop('xp')
        self.xp.savez_compressed(*args, **kwargs, **data)