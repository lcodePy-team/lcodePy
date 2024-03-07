from mpi4py import MPI
import numpy as np
from .util import mpi_type_from_dtype

__id = 100
def get_unique_tag() -> int:
    global __id
    __id += 1
    return __id
    


class MPITransport:

    def __init__(self, steps:int, dtype: np.dtype, tag: int = get_unique_tag()):
        self.steps: int = steps
        self.dtype: int = dtype
        self.mpi_dtype: MPI.Datatype = mpi_type_from_dtype(dtype)
        self.tag: int = tag
        self._comm: MPI.Comm = MPI.COMM_WORLD
        self._rank: int = self._comm.Get_rank()
        self._size: int = self._comm.Get_size()
        self.processed_steps: int = 0

    @property
    def next_node(self) -> int:
        return (self._rank + 1) % self._size
    
    @property
    def prev_node(self) -> int:
        return (self._rank + self._size - 1) % self._size
    
    @property
    def single_process(self) -> bool:
        return self._size == 1
    
    @property
    def first_step(self) -> bool:
        return self._rank == 0 and self.processed_steps == 0
    
    @property
    def final_step(self) -> bool:
        return self.processed_steps*self._size + self._rank == self.steps - 1

    @property
    def last_step(self) -> bool:
        return self.processed_steps == self.steps_per_node - 1
    
    @property
    def steps_per_node(self) -> int:
        return self.steps // self._size + (1 if self.steps % self._size > self._rank else 0)
    
    def send(self, data: np.ndarray):
        self._comm.Send([data, self.mpi_dtype], self.next_node, self.tag)

    def recv(self) -> np.ndarray:
        status = MPI.Status()
        self._comm.Probe(source=self.prev_node, status=status)
        size = status.Get_count(datatype=self.mpi_dtype)
        buf = np.empty(size, dtype=self.dtype)
        self._comm.Recv([buf, self.mpi_dtype], source=self.prev_node, tag=self.tag)
        return buf
