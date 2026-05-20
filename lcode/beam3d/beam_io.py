"""In-memory 3D beam source/drain implementations."""

from __future__ import annotations

import numpy as np

from .data import BeamParticles
from ..beam.beam_io import BeamSource, BeamDrain
from ..config.config import Config


class MemoryBeamSource3D(BeamSource):
    """Supplies 3D beam particles filtered by plasma layer index."""

    def __init__(self, config: Config, beam_particles):
        self._xp = config.xp
        self._dxi = config.getfloat('xi-step')

        if not isinstance(beam_particles, BeamParticles):
            beam = BeamParticles(config.xp, beam_particles)
        else:
            beam = beam_particles

        beam.sort_by_xi()
        self._beam = beam

    def pull(self, layer_index: int) -> BeamParticles:
        xi_min = -self._dxi * layer_index
        xi_max = -self._dxi * (layer_index + 1)

        if self._beam.xi.size and self._beam.xi[0] > xi_min:
            print(
                'MemoryBeamSource3D: particles skipped ahead of first plasma slice '
                f'(xi = {round(xi_min, 7)})'
            )
            layer_length = self._xp.sum(self._beam.xi > self._xp.asarray(xi_min))
            _, self._beam = self._beam.cut_beam_layer(layer_length)

        if self._beam.xi.size:
            layer_length = int(self._xp.sum(
                (self._xp.asarray(xi_max) <= self._beam.xi) &
                (self._beam.xi < self._xp.asarray(xi_min))
            ))
        else:
            layer_length = 0

        layer, self._beam = self._beam.cut_beam_layer(layer_length)
        return layer


class MemoryBeamDrain3D(BeamDrain):
    """Collects 3D beam particles in memory."""

    def __init__(self, config: Config):
        self._xp = config.xp
        self._beam = BeamParticles(config.xp)
        self._lost = BeamParticles(config.xp)

    def push(self, layer_index: int, data: BeamParticles) -> None:
        if data.id.size > 0:
            self._beam.append(data)

    def push_lost(self, layer_index: int, data: BeamParticles) -> None:
        if data.id.size > 0:
            self._lost.append(data)

    def finish_layer(self, xi: float) -> None:
        pass

    def beam_slice(self) -> BeamParticles:
        return self._beam

    def save(self, *args, **kwargs) -> None:
        self._beam.save(*args, **kwargs)


# ---------------------------------------------------------------------------
# Rigid beam (no actual particle tracking needed)
# ---------------------------------------------------------------------------

class RigidBeamSource3D(BeamSource):
    def __init__(self, config: Config, charge_distribution_fn):
        self._fn = charge_distribution_fn

    def pull(self, layer_index: int):
        return self._fn


class RigidBeamDrain3D(BeamDrain):
    def __init__(self, config: Config):
        pass

    def push(self, layer_index: int, data) -> None:
        pass

    def push_lost(self, layer_index: int, data) -> None:
        pass

    def beam_slice(self):
        return None
