"""MPI adapters that connect LayerTransport to the BeamSource/BeamDrain interface."""

from __future__ import annotations

from typing import Callable
import numpy as np

from .transport import LayerTransport
from ..beam.beam_io import BeamSource, BeamDrain


class MPIBeamSource(BeamSource):
    """BeamSource that pulls float64 arrays from LayerTransport and reconstructs particles."""

    def __init__(
        self,
        transport: LayerTransport,
        reconstruct_fn: Callable[[np.ndarray], object],
    ):
        """
        Args:
            transport:       Shared LayerTransport instance.
            reconstruct_fn:  flat float64 array -> BeamParticles.
        """
        self._transport = transport
        self._reconstruct = reconstruct_fn

    def pull(self, layer_index: int):
        arr = self._transport.pull(layer_index)
        return self._reconstruct(arr)


class MPIBeamDrain(BeamDrain):
    """BeamDrain that serialises particles to float64 and pushes into LayerTransport."""

    def __init__(
        self,
        transport: LayerTransport,
        reconstruct_fn: Callable[[np.ndarray], object],
    ):
        """
        Args:
            transport:       Shared LayerTransport instance.
            reconstruct_fn:  flat float64 array -> BeamParticles (used for save()).
        """
        self._transport = transport
        self._reconstruct = reconstruct_fn

    def push(self, layer_index: int, data) -> None:
        self._transport.push(layer_index, data.as_array().ravel())

    def push_lost(self, layer_index: int, data) -> None:
        pass  # lost particles are not transferred between time steps

    def finish_layer(self, xi: float) -> None:
        pass

    def beam_slice(self):
        """Return accumulated particles as a BeamParticles object."""
        arr = self._transport.drain_as_array()
        return self._reconstruct(arr)

    def save(self, *args, **kwargs) -> None:
        self.beam_slice().save(*args, **kwargs)
