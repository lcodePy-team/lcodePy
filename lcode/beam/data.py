import numpy as np
from typing_extensions import Annotated, get_origin, get_args, Union, Self, Tuple
from itertools import chain

particle_dtype = np.dtype([('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'),
                           ('p_r', 'f8'), ('M', 'f8'), ('q_m', 'f8'),
                           ('q_norm', 'f8'), ('id', 'i8')])


class Particles:
    """
    A base class for managing particle data using dynamic field initialization 
    based on type annotations.

    Attributes:
        xp (module): The array library module (numpy or cupy).
        fields (list): Metadata for standard fields defined in annotations.
        extra_fields (list): Metadata for auxiliary fields marked as 'extra'.
    """
    _fields_info = []
    _extra_fields_info = []
    _all_field_names = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        fields_dict = {n: t for n, t in getattr(cls, '_fields_info', [])}
        extra_dict = {n: t for n, t in getattr(cls, '_extra_fields_info', [])}

        ann = getattr(cls, '__annotations__', {})
        for name, type_ in ann.items():
            if get_origin(type_) is Annotated:
                t, *metadata = get_args(type_)
                dtype = metadata[0]
                if len(metadata) > 1 and metadata[1] == 'extra':
                    extra_dict[name] = dtype
                else:
                    fields_dict[name] = dtype
            else:
                fields_dict[name] = 'f8'

        cls._fields_info = list(fields_dict.items())
        cls._extra_fields_info = list(extra_dict.items())

        cls._all_field_names = set(fields_dict.keys()) | set(extra_dict.keys())

    _DTYPE_MAP = {
        'f8': np.float64,
        'i8': np.int64,
        'f4': np.float32,
        'i4': np.int32,
        '?': bool
    }

    @property
    def all_fields(self):
        """Returns a combined list of standard and extra fields."""
        return self._fields_info + self._extra_fields_info

    def __init__(self, xp: np = np, beam_array: np.ndarray = None,
                 size:int = 0, _empty: bool = False):
        """
        Initializes particle fields based on class annotations.

        Args:
            xp: The array backend to use (defaults to numpy).
            beam_array: Optional initial data. Can be a structured numpy array 
                or a 2D array where columns match the order of defined fields.
        
        Raises:
            ValueError: If an annotated type is not found in _DTYPE_MAP.
        """
        self.xp = xp

        if _empty:
            return
        
        dtype_map = self._DTYPE_MAP
            
        if beam_array is not None:
            if hasattr(beam_array, 'dtype') and beam_array.dtype.names:
                for name, type_ in self._fields_info:
                    setattr(self, name, xp.array(beam_array[name], dtype_map[type_]))
            else: 
                for i, (name, type_) in enumerate(self._fields_info):
                    setattr(self, name, xp.array(beam_array[:,i], dtype_map[type_]))
            
            for name, type_ in self._extra_fields_info:
                setattr(self, name, xp.zeros(len(beam_array), dtype_map[type_]))
        else:
            for name, type_ in self.all_fields:
                setattr(self, name, xp.zeros(size, dtype_map.get(type_, xp.float64)))
 
        self._check_integrity()

    def as_array(self) -> np.ndarray:
        """
        Stacks the standard fields into a single 2D array.

        Returns:
            An array where each column corresponds to a field in self.fields.
        """
        data = [self[name] for name, _ in self._fields_info]  
        stacked = self.xp.column_stack(data)
        return stacked

    def save(self, *args, **kwargs) -> None:
        """
        Saves standard fields to a compressed .npz file.
        
        Args:
            *args: Positional arguments passed to xp.savez_compressed.
            **kwargs: Keyword arguments passed to xp.savez_compressed.
        """
        data = {
            name: getattr(self, name)
            for name, _ in self._fields_info
        }
        self.xp.savez_compressed(*args, **kwargs, **data)

    def load(self, *args, **kwargs) -> None:
        """
        Loads field data from an .npz file and initializes extra fields to zero.

        Args:
            *args: Positional arguments passed to xp.load.
            **kwargs: Keyword arguments passed to xp.load.
        """
        with self.xp.load(*args, **kwargs) as data:
            files = data.files if self.xp == np else data.npz_file.files
            for field in files:
                setattr(self, field, self.xp.array(data[field]))
            
            loaded_size = len(data[files[0]]) if files else 0
            for field, type_ in self._extra_fields_info:
                setattr(self, field, self.xp.zeros(loaded_size, dtype=self._DTYPE_MAP[type_]))
        self._check_integrity()

    def __getitem__(self, key):
        """
        Provides access to fields by name or slices the entire particle set.

        Args:
            key (str, int, slice, or array): If str, returns the specific array.
                Otherwise, returns a new instance containing the sliced particles.
        
        Returns:
            The requested field array or a new Particles instance.
        """     
        if isinstance(key, str):
            return getattr(self, key)
        
        result = self.__class__(xp = self.xp, _empty = True)
        
        for name, _ in self.all_fields:
            setattr(result, name, getattr(self, name)[key])

        return result
    
    def __setitem__(self, key, value):
        """
        Sets the value of a specific field.

        Args:
            key (str): Field name.
            value: Array-like data to assign.

        Raises:
            ValueError: If the key is invalid or data length does not match current size.
        """
        if not isinstance(key, str) or key not in self._all_field_names:
            raise ValueError("Invalid key")
        
        value = self.xp.asarray(value)
        if len(value) != len(self[key]):
            raise ValueError("Length mismatch")
        
        setattr(self, key, value)

    def append(self, other: Self) -> Self:
        """
        Concatenates another Particles object to the current one.

        Args:
            other: An instance of the same Particles class.

        Returns:
            Self: The modified instance with appended data.
        """
        if type(self) != type(other):
            raise ValueError("Incompatible types")
        for name, _ in self.all_fields:
            setattr(self, name, self.xp.concatenate((self[name], other[name])))
        self._check_integrity()
        return self
    
    def sort(self, key) -> Self:
        """
        Sorts all fields based on the provided key.

        Args:
            key (str or array-like): Field name or array to use for sorting indices.

        Returns:
            Self: The modified instance with sorted fields.
        """
        if isinstance(key, str):
            key = getattr(self, key)
        if len(key) != self.size:
            raise ValueError("Length mismatch")
        sort_idxes = self.xp.argsort(key)
        for name, _ in self.all_fields:
            self[name] = self[name][sort_idxes]
        return self

    @property
    def size(self) -> int:
        """Returns the number of particles (length of the first field)."""
        return len(self[self._fields_info[0][0]]) \
            if len(self._fields_info) else 0

    def __len__(self) -> int:
        """Returns the number of particles."""
        return self.size
    
    def _check_integrity(self) -> None:
        """
        Validates that all internal arrays have consistent lengths.

        Raises:
            RuntimeError: If field lengths are inconsistent.
        """
        sizes = {len(getattr(self, name)) for name, _ in self.all_fields}
        if len(sizes) > 1:
            raise RuntimeError("Inconsistent field sizes")
    

class BeamParticlesBase(Particles):
    """Base class for beam-specific particle operations."""

    def sort_by_xi(self) -> None:
        """Sorts particles by their longitudinal position (xi) in descending order."""
        self.sort(-self.xi)

    def cut_beam_layer(self, layer_length: int) -> Tuple[Self, Self]:
        """
        Splits the beam into two parts.

        Args:
            layer_length (int): The number of particles to include in the first part.

        Returns:
            Tuple[Self, Self]: A tuple of (beam_layer, remaining_particles).
        """
        beam_layer = self[:layer_length]
        remaining = self[layer_length:]

        return beam_layer, remaining
    

class BeamParticles(BeamParticlesBase):
    """
    Concrete implementation of beam particles with specific physical fields.
    
    Fields include longitudinal and radial coordinates, momenta, mass, 
    charge-to-mass ratio, normalized charge, and unique IDs.
    Extra fields track simulation metadata like time-step and loss status.
    """

    xi: Annotated[np.ndarray, 'f8']
    r: Annotated[np.ndarray, 'f8']
    p_z: Annotated[np.ndarray, 'f8']
    p_r: Annotated[np.ndarray, 'f8']
    M: Annotated[np.ndarray, 'f8']
    q_m: Annotated[np.ndarray, 'f8']
    q_norm: Annotated[np.ndarray, 'f8']
    id: Annotated[np.ndarray, 'i8']

    dt: Annotated[np.ndarray, 'f8', 'extra']
    remaining_steps: Annotated[np.ndarray, 'i8', 'extra']
    lost: Annotated[np.ndarray, '?', 'extra']

    @property  
    def nlost(self) -> int:
        return int(self.xp.sum(self.lost))