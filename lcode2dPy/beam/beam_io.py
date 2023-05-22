import logging
from abc import ABC, abstractmethod

import numpy as np
import numba

from .beam_slice import BeamSlice
from ..config.config import Config


class BeamSource(ABC):
    """
    BeamSource abstracts source of beam particles.
    """

    @abstractmethod
    def get_beam_slice(self, xi_max: float, xi_min: float) -> BeamSlice:
        """
        Returns slice from source.

        BeamSource guarantees, that all particles in returned slice lie between xi_max and xi_min and are not lost yet.
        xi_max should be bigger than xi_min.
        """
        pass


class BeamDrain(ABC):
    """
    BeamDrain abstracts output of beam particles.
    """

    @abstractmethod
    def push_beam_slice(self, beam_slice: BeamSlice) -> None:
        """
        Write particles from beam slice to output.

        Particles in `beam_slice` must finish their movement in time step and must not be lost.
        Writing particles with xi from finished layer are not written to output.
        """
        pass

    @abstractmethod
    def finish_layer(self, xi: float) -> None:
        """
        Finish xi layer.

        Particles written after finishing layer should have xi less than xi of finished layer.
        """
        pass

    @abstractmethod
    def push_lost(self, time: float, beam_slice: BeamSlice) -> None:
        """
        Write lost particles.
        """
        pass


particle_dtype = np.dtype([('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'), ('q_m', 'f8'),
                           ('q_norm', 'f8'), ('id', 'i8')])

lost_particle_dtype = np.dtype([('time', 'f8'), ('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'),
                                ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])

@numba.njit
def find_sub_slice(beam_slice, used_count, xi_max, xi_min):
    start = used_count
    end = beam_slice.size
    flag = 0
    for i in np.arange(used_count, beam_slice.size):
        if beam_slice.xi[i] - xi_min < 0:
            end = i
            break
        if beam_slice.xi[i] - xi_max > 0:
            end = start
            flag = 1
            break
    used_count += end - start
    return start, end, used_count, flag


class MemoryBeamSource(BeamSource):
    def __init__(self, config: Config, beam_slice):
        if type(beam_slice) == np.ndarray:
            beam_slice = BeamSlice(beam_slice.size, beam_slice)

        self._beam_slice = beam_slice
        self._used_count = 0
        if beam_slice.particles.size == 0:
            return
        sorted_idxes = np.argsort(-self._beam_slice.xi)
        self._beam_slice.particles = self._beam_slice.particles[sorted_idxes]
        # Remove stub particle for compatibility (xi = -100000)
        if (self._beam_slice.xi[-1] + 100000) < 1:
            self._beam_slice.particles = self._beam_slice.particles[:-1]
        self._beam_slice.dt.fill(0.0)
        self._beam_slice.remaining_steps.fill(1.0)
        

    def get_beam_slice(self, xi_max: float, xi_min: float) -> BeamSlice:
        assert xi_min < xi_max
        start, end, self._used_count, flag = find_sub_slice(self._beam_slice,
                                                            self._used_count,
                                                            xi_max, xi_min)
        if flag:
            logging.debug(f'Wrong order of the particles')
        logging.debug(f'MemoryBeamSource: sourced {end - start} particles')
        return BeamSlice(end - start, particles=self._beam_slice.particles[start:end])


class MemoryBeamDrain(BeamDrain):
    def finish_layer(self, xi):
        pass

    def __init__(self, config: Config):
        self._beam_buffer = []
        self._beam_buffer_lost = []

    def push_beam_slice(self, beam_slice: BeamSlice):
        if beam_slice.size > 0:
            logging.debug(f'MemoryBeamDrain: drained {beam_slice.size} particles')
            self._beam_buffer.append(beam_slice)

    def push_lost(self, time, beam_slice: BeamSlice):
        if beam_slice.size > 0:
            print(f'MemoryBeamDrain: lost {beam_slice.size} particles')
            self._beam_buffer_lost.append(beam_slice)

    def beam_slice(self):
        return np.concatenate([beam_slice.particles for beam_slice in self._beam_buffer]) if len(self._beam_buffer )> 0 else np.array([], dtype = particle_dtype)


class DebugSource(BeamSource):
    def __init__(self, source):
        self._source = source
        self._beam_buffer = []

    def get_beam_slice(self, xi_start, xi_end) -> BeamSlice:
        slice = self._source.get_beam_slice(xi_start, xi_end)
        self._beam_buffer.append(BeamSlice(slice.size, particles=np.copy(slice.particles)))
        return slice

    def get_debug_slice(self):
        return np.concatenate([beam_slice.particles for beam_slice in self._beam_buffer])


class DebugDrain(BeamDrain):
    def finish_layer(self, xi: float) -> None:
        self._drain.finish_layer(xi)

    def __init__(self, drain):
        self._drain = drain
        self._beam_buffer = []
        self._beam_buffer_lost = []

    def push_beam_slice(self, beam_slice: BeamSlice):
        self._beam_buffer.append(beam_slice)
        self._drain.push_beam_slice(beam_slice)

    def push_lost(self, time, beam_slice: BeamSlice):
        if beam_slice.size > 0:
            self._beam_buffer_lost.append(beam_slice)

    def get_beam_slice(self):
        return np.concatenate([beam_slice.particles for beam_slice in self._beam_buffer])
