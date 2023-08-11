import logging

import numpy as np
from mpi4py import MPI

from ..beam import BeamParticles
from ..beam.beam_io import BeamSource, BeamDrain
from .util import MPIWorker, particle_dtype

MPI_TAG_PARTICLES = 1

finish_layer_particle = BeamParticles(1, particles=np.zeros(1, dtype=particle_dtype))


class MPIBeamSource(BeamSource, MPIWorker):
    def __init__(self, steps: int, initial_source: BeamSource):
        MPIWorker.__init__(self, steps)
        if self._rank == 0:
            self._initial_source: BeamSource = initial_source
        self.xi_finished = 100000.0
        logging.debug(f'MPIBeamSource: rank {self._rank}, size {self._size}, {steps} steps')

    def _try_read_particles(self):
        status = MPI.Status()
        self._comm.Probe(source=self.prev_node, tag=MPI_TAG_PARTICLES, status=status)
        particles_bufsize = status.Get_count(datatype=self.particles_type)
        logging.debug(f'MPIBeamSource: probed {particles_bufsize} particles')
        particles_buf = np.zeros(particles_bufsize, dtype=particle_dtype)
        self._comm.Recv([particles_buf, MPIWorker.particles_type], source=self.prev_node, tag=MPI_TAG_PARTICLES)
        return particles_buf

    def get_beam_slice(self, xi_max: float, xi_min: float) -> BeamParticles:
        if self.first_step:
            beam_slice = self._initial_source.get_beam_slice(xi_max, xi_min)
            return beam_slice
        assert xi_min < xi_max
        if xi_min > self.xi_finished:
            return BeamParticles(0, particles=np.zeros(0, dtype=particle_dtype))

        beam_buffer = []
        while xi_min <= self.xi_finished:
            particles_buf = self._try_read_particles()
            if len(particles_buf) == 1 and particles_buf[0]['id'] == 0:
                self.xi_finished = particles_buf[0]['xi']
                logging.debug(f'MPIBeamSource: layer xi = {self.xi_finished:.4} finished')
                break
            for particle in particles_buf:
                if particle['xi'] < xi_min:
                    logging.debug(f'MPIBeamSource: Wrong particle order, {particle["xi"]} < {xi_min}')
                if particle['xi'] > xi_max:
                    logging.debug(f'MPIBeamSource: Wrong particle order, {particle["xi"]} > {xi_max}')
            beam_buffer.append(particles_buf)
        if len(beam_buffer) != 0:
            particles_buf = np.concatenate(beam_buffer)
            return BeamParticles(len(particles_buf), particles=particles_buf)
        else:
            return BeamParticles(0)


class MPIBeamDrain(BeamDrain, MPIWorker):
    def __init__(self, steps: int, final_drain: BeamDrain):
        MPIWorker.__init__(self, steps)
        self._final_drain = final_drain
        if self._rank == (self._total_steps - 1) % self._size:
            self._final_drain = final_drain
        logging.debug(f'MPIBeamDrain: rank {self._rank}, size {self._size}, {steps} steps')
        self.xi_finished = 100000.0

    def finish_layer(self, xi: float) -> None:
        if xi >= self.xi_finished:
            logging.debug(f'MPIBeamDrain: repeated finish_layer')
            return
        if self.last_step:
            self._final_drain.finish_layer(xi)
            return
        finish_xi_particle = np.zeros(1, dtype=particle_dtype)
        finish_xi_particle[0]['xi'] = xi
        logging.debug(f'MPIBeamDrain: finish_layer {xi:.4}')
        self._comm.Send([finish_xi_particle, self.particles_type], self.next_node, MPI_TAG_PARTICLES)
        self.xi_finished = xi

    def push_beam_slice(self, beam_slice: BeamParticles):
        if self.last_step:
            self._final_drain.push_beam_slice(beam_slice)
            return
        self._comm.Send([beam_slice.particles, self.particles_type], self.next_node, MPI_TAG_PARTICLES)
        logging.debug(f'MPIBeamDrain: sent {len(beam_slice.particles)} particles')

    def push_lost(self, time: float, beam_slice: BeamParticles):
        pass
