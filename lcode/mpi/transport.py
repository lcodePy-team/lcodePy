from __future__ import annotations

from typing import Callable, List
import numpy as np

from .core import MPIContext
from .backends.memory import MemoryBackend
from .backends.disk import DiskBackend

_next_tag: int = 100


def _alloc_tags(n: int) -> range:
    global _next_tag
    tags = range(_next_tag, _next_tag + n)
    _next_tag += n
    return tags


_EMPTY = np.array([], dtype=np.float64)


class LayerTransport:
    """Generic xi-layer transport across MPI ranks.

    Works at the numpy level — data in, data out as flat float64 arrays.
    Callers wrap it with domain-specific adapters (e.g. MPIBeamSource/Drain).

    Pipeline:
      - rank R processes layers 0..xi_steps-1 and sends each result to rank R+1.
      - After all layers, next_step() transfers the full accumulated buffer from
        rank size-1 back to rank 0 for the next round of time steps.

    Args:
        ctx:               MPI context (single-process-aware).
        steps:             Total number of time steps in the simulation.
        name:              Label used for disk-backend temp files.
        backend:           "memory" or "disk".
        initial_source_fn: layer_index -> flat float64 array, used by rank 0.
        source_factory:    flat float64 array -> (layer_index -> flat float64 array).
                           Recreates source for rank 0 after receiving data from rank size-1.
    """

    def __init__(
        self,
        ctx: MPIContext,
        steps: int,
        name: str,
        backend: str,
        initial_source_fn: Callable[[int], np.ndarray],
        source_factory: Callable[[np.ndarray], Callable[[int], np.ndarray]],
    ):
        self.ctx = ctx
        self.steps = steps
        self._source_fn = initial_source_fn
        self._source_factory = source_factory

        tags = _alloc_tags(2)
        if backend == 'memory':
            self._backend = MemoryBackend(ctx, layer_tag=tags[0], full_tag=tags[1])
        elif backend == 'disk':
            self._backend = DiskBackend(ctx, layer_tag=tags[0], full_tag=tags[1], name=name)
        else:
            raise ValueError(f"Unknown MPI backend '{backend}'. Use 'memory' or 'disk'.")

        self._processed_steps: int = 0
        self._skip_first: bool = True
        self._drain: List[np.ndarray] = []


    # ------------------------------------------------------------------
    # Step-counting helpers
    # ------------------------------------------------------------------

    @property
    def steps_per_node(self) -> int:
        size, rank = self.ctx.size, self.ctx.rank
        return self.steps // size + (1 if self.steps % size > rank else 0)

    @property
    def _first_step(self) -> bool:
        return self.ctx.rank == 0 and self._processed_steps == 0

    @property
    def _final_step(self) -> bool:
        return (self._processed_steps * self.ctx.size + self.ctx.rank
                == self.steps - 1)

    @property
    def _last_step(self) -> bool:
        return self._processed_steps == self.steps_per_node - 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pull(self, layer_index: int) -> np.ndarray:
        """Return flat float64 array for layer_index."""
        if self.ctx.rank == 0:
            return self._source_fn(layer_index)
        return self._backend.recv_layer(source=self.ctx.rank - 1)

    def push(self, layer_index: int, data: np.ndarray) -> None:
        """Store processed layer; forward to next rank when appropriate."""
        is_last_rank = self.ctx.rank == self.ctx.size - 1

        if is_last_rank or self._final_step:
            self._drain.append(data)
            return

        if self._skip_first:
            self._skip_first = False
            return

        self._drain.append(data)
        self._backend.send_layer(data, dest=self.ctx.rank + 1)

    def next_step(self) -> None:
        """Advance to the next time step.

        rank size-1 sends its accumulated drain to rank 0.
        rank 0 receives and rebuilds source_fn for the next round.
        All ranks seed the first pull() of rank+1 with an empty array.
        """
        is_final = self._final_step
        is_last_local = self._last_step

        self._processed_steps += 1
        self._skip_first = True

        if is_final:
            self._drain = []
            return

        if self.ctx.is_single:
            data = self._drain_as_array()
            self._source_fn = self._source_factory(data)
            self._drain = []
            return

        if self.ctx.rank == self.ctx.size - 1:
            self._backend.send_full(self._drain_as_array(), dest=0)
            self._drain = []
            self._backend.advance_step()
            return

        # Seed first pull() of rank+1 for the next xi-loop.
        self._backend.send_layer(_EMPTY, dest=self.ctx.rank + 1)
        self._drain = []

        if self.ctx.rank == 0 and not is_last_local:
            data = self._backend.recv_full(source=self.ctx.size - 1)
            self._source_fn = self._source_factory(data)

        self._backend.advance_step()

    def close(self) -> None:
        """Cancel any pending prefetch receive."""
        if hasattr(self._backend, 'cancel_prefetch'):
            self._backend.cancel_prefetch()

    def drain_as_array(self) -> np.ndarray:
        """Return accumulated drain content as a flat float64 array."""
        return self._drain_as_array()

    def _drain_as_array(self) -> np.ndarray:
        if not self._drain:
            return _EMPTY
        return np.concatenate(self._drain)
