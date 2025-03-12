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
    total_size = dtype.itemsize
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
