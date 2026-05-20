"""Beam source/drain abstractions and in-memory 2D implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import numpy as np
import numba

from .data import BeamParticles
from ..config.config import Config


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class BeamSource(ABC):
    @abstractmethod
    def pull(self, layer_index: int) -> BeamParticles:
        """Return beam particles for the given xi layer."""


class BeamDrain(ABC):
    @abstractmethod
    def push(self, layer_index: int, data: BeamParticles) -> None:
        """Receive processed particles from the given xi layer."""

    @abstractmethod
    def push_lost(self, layer_index: int, data: BeamParticles) -> None:
        """Receive lost particles from the given xi layer."""

    def finish_layer(self, xi: float) -> None:
        """Optional hook called when a xi layer is fully processed."""


# ---------------------------------------------------------------------------
# 2D in-memory implementations
# ---------------------------------------------------------------------------

@numba.njit
def _find_sub_slice(xi_array, used_count, xi_max, xi_min):
    """Find the slice of xi_array (sorted descending) with xi_min <= xi <= xi_max."""
    start = used_count
    end = xi_array.size
    flag = 0
    for i in np.arange(start, end):
        if xi_array[i] - xi_min < 0:
            end = i
            break
        if xi_array[i] - xi_max > 0:
            end = start
            flag = 1
            break
    used_count += end - start
    return start, end, used_count, flag


class MemoryBeamSource2D(BeamSource):
    """Supplies 2D beam particles from an in-memory buffer, filtered by xi layer."""

    def __init__(self, config: Config, beam_particles):
        self._dxi = config.getfloat('xi-step')

        if isinstance(beam_particles, np.ndarray):
            beam = BeamParticles(beam_array=beam_particles)
        else:
            beam = beam_particles

        if beam.size == 0:
            self._beam = beam
            self._used_count = 0
            return

        beam.sort_by_xi()
        # Remove legacy stub particle (xi = -100000).
        if (beam.xi[-1] + 100000) < 1:
            beam = beam[:-1]
        beam.dt.fill(0.0)
        beam.remaining_steps.fill(1.0)
        self._beam = beam
        self._used_count = 0

    def pull(self, layer_index: int) -> BeamParticles:
        xi_max = -layer_index * self._dxi
        xi_min = -(layer_index + 1) * self._dxi

        if (self._used_count == 0
                and self._beam.xi.size
                and self._beam.xi[0] > xi_max):
            logging.debug(
                'MemoryBeamSource2D: particles skipped ahead of first plasma slice '
                f'(xi = {round(xi_min, 7)})'
            )
            _, _, self._used_count, _ = _find_sub_slice(
                self._beam.xi, self._used_count, 0, xi_max
            )

        start, end, self._used_count, flag = _find_sub_slice(
            self._beam.xi, self._used_count, xi_max, xi_min
        )
        if flag:
            logging.debug('MemoryBeamSource2D: wrong particle order detected')
        logging.debug(f'MemoryBeamSource2D: sourced {end - start} particles')
        return self._beam[start:end]


class MemoryBeamDrain2D(BeamDrain):
    """Collects 2D beam particles in memory."""

    def __init__(self, config: Config):
        self._buffer: list[BeamParticles] = []
        self._lost_buffer: list[BeamParticles] = []

    def push(self, layer_index: int, data: BeamParticles) -> None:
        if data.size > 0:
            logging.debug(f'MemoryBeamDrain2D: drained {data.size} particles')
            self._buffer.append(data)

    def push_lost(self, layer_index: int, data: BeamParticles) -> None:
        if data.size > 0:
            self._lost_buffer.append(data)

    def finish_layer(self, xi: float) -> None:
        pass

    def beam_slice(self) -> BeamParticles:
        if not self._buffer:
            return BeamParticles(size=0)
        result = BeamParticles(size=0)
        for bp in self._buffer:
            result.append(bp)
        return result

    def save(self, *args, **kwargs) -> None:
        self.beam_slice().save(*args, **kwargs)


# ---------------------------------------------------------------------------
# Debug wrappers
# ---------------------------------------------------------------------------

class DebugBeamSource(BeamSource):
    def __init__(self, source: BeamSource):
        self._source = source
        self._log: list[BeamParticles] = []

    def pull(self, layer_index: int) -> BeamParticles:
        particles = self._source.pull(layer_index)
        self._log.append(BeamParticles(beam_array=np.copy(particles.as_array())))
        return particles

    def get_debug_log(self) -> list:
        return self._log


class DebugBeamDrain(BeamDrain):
    def __init__(self, drain: BeamDrain):
        self._drain = drain
        self._log: list[BeamParticles] = []

    def push(self, layer_index: int, data: BeamParticles) -> None:
        self._log.append(data)
        self._drain.push(layer_index, data)

    def push_lost(self, layer_index: int, data: BeamParticles) -> None:
        self._drain.push_lost(layer_index, data)

    def finish_layer(self, xi: float) -> None:
        self._drain.finish_layer(xi)

    def get_debug_log(self) -> list:
        return self._log
