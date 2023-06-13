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


# NOTE: The implementation may be complicated, but the usage is simple.
class ArraysView:
    """
    This is a magical wrapper around Arrays that handles GPU-RAM data
    transfer transparently.
    Accessing `view.something` will automatically copy array to host RAM,
    setting `view.something = ...` will copy the changes back to GPU RAM.
    Usage: `view = ArraysView(gpu_arrays); view.something`
    Do not add more attributes later, specify them all at construction time.
    NOTE: repeatedly accessing an attribute will result in repeated copying!
    """
    def __init__(self, gpu_arrays):
        """
        Wrap `gpu_arrays` and transparently copy data to/from GPU.
        """
        # Could've been written as `self._arrs = gpu_arrays`
        # if only `__setattr__` was not overwritten!
        # `super(ArraysView) is the proper way to obtain the parent class
        # (`object`), which has a regular boring `__setattr__` that we can use.
        super(ArraysView, self).__setattr__('_arrs', gpu_arrays)

    def __dir__(self):
        """
        Make `dir()` also show the wrapped `gpu_arrays` attributes.
        """
        # See `ArraysView.__init__` for the explanation how we access the
        # parent's plain `__dir__()` implementation (and avoid recursion).
        return list(set(super(ArraysView, self).__dir__() +
                        dir(self._arrs)))

    def __getattr__(self, attrname):
        """
        Intercept access to (missing) attributes, access the wrapped object
        attributes instead and copy the arrays from GPU to RAM.
        """
        try: # cupy
            return getattr(self._arrs, attrname).get()  # auto-copies to host RAM
        except AttributeError: # numpy
            return getattr(self._arrs, attrname)

    def __setattr__(self, attrname, value):
        """
        Intercept setting attributes, access the wrapped object attributes
        instead and reassign their contents, copying the arrays from RAM
        to GPU in the process.
        """
        getattr(self._arrs, attrname)[...] = value  # copies to GPU RAM
        # TODO: just copy+reassign it without preserving identity and shape?
