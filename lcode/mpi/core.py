import numpy as np

try:
    from mpi4py import MPI as _MPI
    _mpi_available = True
except ImportError:
    _mpi_available = False


class MPIContext:
    """Thin wrapper over MPI communicator. Falls back to single-process if mpi4py is unavailable."""

    def __init__(self):
        if _mpi_available:
            self._comm = _MPI.COMM_WORLD
            self.rank = self._comm.Get_rank()
            self.size = self._comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

    @property
    def is_single(self) -> bool:
        return self.size == 1

    def send(self, data: np.ndarray, dest: int, tag: int) -> None:
        flat = np.ascontiguousarray(data.ravel(), dtype=np.float64)
        self._comm.Send([flat, _MPI.DOUBLE], dest=dest, tag=tag)

    def isend(self, data: np.ndarray, dest: int, tag: int):
        """Non-blocking send. Returns (request, buffer) — caller must keep buffer alive until wait."""
        flat = np.ascontiguousarray(data.ravel(), dtype=np.float64)
        req = self._comm.Isend([flat, _MPI.DOUBLE], dest=dest, tag=tag)
        return req, flat

    def irecv(self, buf: np.ndarray, source: int, tag: int) -> '_IrecvRequest':
        """Post non-blocking receive into pre-allocated buf. Call .wait() to get data."""
        req = self._comm.Irecv([buf, _MPI.DOUBLE], source=source, tag=tag)
        return _IrecvRequest(req, buf)

    def recv(self, source: int, tag: int) -> np.ndarray:
        status = _MPI.Status()
        self._comm.Probe(source=source, tag=tag, status=status)
        count = status.Get_count(_MPI.DOUBLE)
        buf = np.empty(count, dtype=np.float64)
        self._comm.Recv([buf, _MPI.DOUBLE], source=source, tag=tag)
        return buf


class _IrecvRequest:
    """Wraps an Irecv request together with its buffer."""

    def __init__(self, req, buf: np.ndarray):
        self._req = req
        self._buf = buf

    def wait(self) -> np.ndarray:
        status = _MPI.Status()
        self._req.Wait(status)
        count = status.Get_count(_MPI.DOUBLE)
        return self._buf[:count].copy()
