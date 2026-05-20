import numpy as np

from ..core import MPIContext

_PREFETCH_BUF_ELEMENTS = 1 << 20  # 8 MB — enough for any realistic beam layer


class MemoryBackend:
    """Transfers layer data between ranks via RAM (MPI messages)."""

    def __init__(self, ctx: MPIContext, layer_tag: int, full_tag: int):
        self.ctx = ctx
        self.layer_tag = layer_tag
        self.full_tag = full_tag
        self._prefetch = None    # _IrecvRequest pre-posted for the next pull
        self._prefetch_buf = np.empty(_PREFETCH_BUF_ELEMENTS, dtype=np.float64)

    def _post_irecv(self, source: int) -> None:
        self._prefetch = self.ctx.irecv(
            self._prefetch_buf, source=source, tag=self.layer_tag)

    def send_layer(self, data: np.ndarray, dest: int) -> None:
        # Blocking send: completes as soon as dest's pre-posted Irecv matches.
        # This limits the sender to at most 1 layer ahead of the receiver,
        # preventing fast ranks from running far ahead and serializing the pipeline.
        self.ctx.send(data, dest=dest, tag=self.layer_tag)

    def recv_layer(self, source: int) -> np.ndarray:
        if self._prefetch is None:
            self._post_irecv(source)
        data = self._prefetch.wait()
        self._post_irecv(source)
        return data

    def send_full(self, data: np.ndarray, dest: int) -> None:
        self.ctx.send(data, dest=dest, tag=self.full_tag)

    def recv_full(self, source: int) -> np.ndarray:
        return self.ctx.recv(source=source, tag=self.full_tag)

    def advance_step(self) -> None:
        pass

    def cancel_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch._req.Cancel()
            self._prefetch = None
