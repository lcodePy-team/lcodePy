"""Module for setting classes for Fields, Currents and Partciles data types."""
import cupy as cp


# Grouping GPU arrays, with optional transparent RAM<->GPU copying #

class GPUArrays:
    """
    A convenient way to group several GPU arrays and access them with a dot.
    `x = GPUArrays(something=numpy_array, something_else=another_array)`
    will create `x` with `x.something` and `x.something_else` being GPU arrays.
    Do not add more attributes later, specify them all at construction time.
    """
    def __init__(self, **kwargs):
        """
        Convert the keyword arguments to `cupy` arrays and assign them
        to the object attributes.
        Amounts to, e.g., `self.something = cp.asarray(numpy_array)`,
        and `self.something_else = cp.asarray(another_array)`,
        see class doctring.
        """
        for name, array in kwargs.items():
            setattr(self, name, cp.array(array)) # or asarray?

    def copy(self):
        """
        Create an indentical copy of the group of `cupy` arrays.
        """
        return GPUArrays(**self.__dict__)


# NOTE: The implementation may be complicated, but the usage is simple.
class GPUArraysView:
    """
    This is a magical wrapper around GPUArrays that handles GPU-RAM data
    transfer transparently.
    Accessing `view.something` will automatically copy array to host RAM,
    setting `view.something = ...` will copy the changes back to GPU RAM.
    Usage: `view = GPUArraysView(gpu_arrays); view.something`
    Do not add more attributes later, specify them all at construction time.
    NOTE: repeatedly accessing an attribute will result in repeated copying!
    """
    def __init__(self, gpu_arrays):
        """
        Wrap `gpu_arrays` and transparently copy data to/from GPU.
        """
        # Could've been written as `self._arrs = gpu_arrays`
        # if only `__setattr__` was not overwritten!
        # `super(GPUArraysView) is the proper way to obtain the parent class
        # (`object`), which has a regular boring `__setattr__` that we can use.
        super(GPUArraysView, self).__setattr__('_arrs', gpu_arrays)

    def __dir__(self):
        """
        Make `dir()` also show the wrapped `gpu_arrays` attributes.
        """
        # See `GPUArraysView.__init__` for the explanation how we access the
        # parent's plain `__dir__()` implementation (and avoid recursion).
        return list(set(super(GPUArraysView, self).__dir__() +
                        dir(self._arrs)))

    def __getattr__(self, attrname):
        """
        Intercept access to (missing) attributes, access the wrapped object
        attributes instead and copy the arrays from GPU to RAM.
        """
        return getattr(self._arrs, attrname).get()  # auto-copies to host RAM

    def __setattr__(self, attrname, value):
        """
        Intercept setting attributes, access the wrapped object attributes
        instead and reassign their contents, copying the arrays from RAM
        to GPU in the process.
        """
        getattr(self._arrs, attrname)[...] = value  # copies to GPU RAM
        # TODO: just copy+reassign it without preserving identity and shape?


def fields_average(fields1: GPUArrays, fields2: GPUArrays):
        return GPUArrays(
            Ex = (fields1.Ex + fields2.Ex) / 2,
            Ey = (fields1.Ey + fields2.Ey) / 2,
            Ez = (fields1.Ez + fields2.Ez) / 2,
            Bx = (fields1.Bx + fields2.Bx) / 2,
            By = (fields1.By + fields2.By) / 2,
            Bz = (fields1.Bz + fields2.Bz) / 2,
            Phi = (fields1.Phi + fields2.Phi) / 2
        )
