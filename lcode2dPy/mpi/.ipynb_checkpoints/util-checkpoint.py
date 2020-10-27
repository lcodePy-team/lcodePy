from typing import Dict, ClassVar

import numpy as np
from mpi4py import MPI

particle_dtype = np.dtype([('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'), ('q_m', 'f8'),
                           ('q_norm', 'f8'), ('id', 'i8')])

mapped_dtypes: Dict[str, MPI.Datatype] = {
    'int8': MPI.INT8_T,
    'int16': MPI.INT16_T,
    'int32': MPI.INT32_T,
    'int64': MPI.INT64_T,
    'uint8': MPI.UINT8_T,
    'uint16': MPI.UINT16_T,
    'uint32': MPI.UINT32_T,
    'uint64': MPI.UINT64_T,
    'float32': MPI.FLOAT,
    'float64': MPI.DOUBLE,
}


def _mpi_type_from_builtin(dtype: np.dtype) -> MPI.Datatype:
    if dtype.isbuiltin == 1:
        if dtype.name in mapped_dtypes:
            return mapped_dtypes[dtype.name]
        else:
            raise ValueError(f'builtin numpy dtype "{dtype}" is not supported')
    else:
        raise ValueError(f'numpy dtype "{dtype}" is not builtin')


def mpi_type_from_dtype(dtype: np.dtype) -> MPI.Datatype:
    if dtype.isbuiltin == 1:
        return _mpi_type_from_builtin(dtype)
    elif dtype.isbuiltin == 2:
        raise ValueError(f'fully custom numpy dtype "{dtype}" is not supported')
    total_size = particle_dtype.itemsize
    mpi_types = []
    mpi_displacements = []
    for field_name, field_descr in dtype.fields.items():
        field_dtype, field_displ = field_descr
        mpi_displacements.append(field_displ)
        mpi_types.append(mpi_type_from_dtype(field_dtype))
    struct_type = MPI.Datatype.Create_struct([1] * len(mpi_types), mpi_displacements, mpi_types)
    struct_type = struct_type.Create_resized(0, total_size)
    struct_type.Commit()
    return struct_type


class MPIWorker:
    particles_type: ClassVar[MPI.Datatype] = mpi_type_from_dtype(particle_dtype)

    def __init__(self, steps: int):
        self._comm: MPI.Comm = MPI.COMM_WORLD
        self._rank: int = self._comm.rank
        self._size: int = self._comm.size
        self._total_steps: int = steps
        self._processed_steps: int = 0
        self._remaining_steps: int = steps // self._size + (1 if steps % self._size > self._rank else 0)
        if self._size == 1:
            print('Single process available')

    @property
    def single_process(self):
        return self._size == 1

    @property
    def first_step(self):
        return self._rank == 0 and self._processed_steps == 0

    @property
    def last_step(self):
        return self._rank == ((self._total_steps - 1) % self._size) and self._remaining_steps == 1

    @property
    def prev_node(self):
        return (self._rank + self._size - 1) % self._size

    @property
    def next_node(self):
        return (self._rank + 1) % self._size
