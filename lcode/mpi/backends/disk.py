import os
import numpy as np

from ..core import MPIContext

_NOTIFY = np.array([0.0], dtype=np.float64)


class DiskBackend:
    """Transfers layer data between ranks via .npy files.

    File naming: tmp/<name>/<sender_rank>_<step>_<seq>.npy
    The receiver knows sender's rank (MPI source) and reconstructs the path.
    Sequencing is kept in sync because recv blocks until the notify arrives
    (which only happens after the file is written).
    """

    def __init__(self, ctx: MPIContext, layer_tag: int, full_tag: int, name: str):
        self.ctx = ctx
        self.layer_tag = layer_tag
        self.full_tag = full_tag
        self.name = name
        self._step = 0
        self._send_seq = 0
        self._recv_seq = 0
        os.makedirs(f'tmp/{name}', exist_ok=True)

    def _layer_path(self, sender_rank: int, seq: int) -> str:
        return f'tmp/{self.name}/{sender_rank}_{self._step}_{seq}.npy'

    def _full_path(self, sender_rank: int) -> str:
        return f'tmp/{self.name}/{sender_rank}_{self._step}_full.npy'

    def send_layer(self, data: np.ndarray, dest: int) -> None:
        path = self._layer_path(self.ctx.rank, self._send_seq)
        self._send_seq += 1
        np.save(path, data)
        self.ctx.send(_NOTIFY, dest=dest, tag=self.layer_tag)

    def recv_layer(self, source: int) -> np.ndarray:
        self.ctx.recv(source=source, tag=self.layer_tag)  # wait for file
        path = self._layer_path(source, self._recv_seq)
        self._recv_seq += 1
        data = np.load(path)
        os.remove(path)
        return data

    def send_full(self, data: np.ndarray, dest: int) -> None:
        path = self._full_path(self.ctx.rank)
        np.save(path, data)
        self.ctx.send(_NOTIFY, dest=dest, tag=self.full_tag)

    def recv_full(self, source: int) -> np.ndarray:
        self.ctx.recv(source=source, tag=self.full_tag)  # wait for file
        path = self._full_path(source)
        data = np.load(path)
        os.remove(path)
        return data

    def advance_step(self) -> None:
        self._step += 1
        self._send_seq = 0
        self._recv_seq = 0
