import numpy as np
from ..beam.data import BeamParticles
from .transport import MPITransport
from ..beam3d.data import BeamParticles as BeamParticles3D

class MPIBeamTransport(MPITransport):
    def __init__(self, cfg, steps: int, initial_beam, dtype: np.dtype, BeamSource, BeamDrain):
        MPITransport.__init__(self, steps, dtype)
        
        self.BeamSource = BeamSource
        self.BeamDrain = BeamDrain
        
        self.cfg = cfg
        self.initial_source = BeamSource(cfg, initial_beam)
        self.final_drain = BeamDrain(cfg)

        self.skip_first = True
    
    def push(self, beam_slice) -> None:
        if self.final_step or self._rank == self._size - 1:
            self.final_drain.push_beam_slice(beam_slice)
            return
        if self.skip_first:
            self.skip_first = False
            return
        self.final_drain.push_beam_slice(beam_slice)
        self.send(beam_slice.particles)
    
    def push3d(self, beam_layer):
        if self.final_step or self._rank == self._size - 1:
            self.final_drain.push_beam_layer(beam_layer)
            return
        
        if self.skip_first:
            self.skip_first = False
            return
        self.final_drain.push_beam_layer(beam_layer)
        self.send(beam_layer.particles)

    def pull(self, xi_max, xi_min) -> BeamParticles:
        if self.first_step or self._rank == 0:
            return self.initial_source.get_beam_slice(xi_max, xi_min)
        
        beam = self.recv()
        return BeamParticles(beam.size, beam)
    
    def pull3d(self, plasma_layer_idx):
        if self.first_step or self._rank == 0:
            return self.initial_source.get_beam_layer_to_layout(plasma_layer_idx)
        buf = self.recv()
        beam = BeamParticles3D(self.cfg.xp)
        beam.init_generated(buf)
     
        return beam
    
    
    class MPIBeamDrain:
        def __init__(self, beam_transport):
            self.beam_transport = beam_transport
        def push_beam_slice(self, beam_slice):
            self.beam_transport.push(beam_slice)
        def beam_slice(self):
            return self.beam_transport.final_drain.beam_slice()
        
        def push_beam_layer(self, beam_layer):
            self.beam_transport.push3d(beam_layer)

        def save(self, *args, **kwargs):
            self.beam_transport.final_drain.save(*args, **kwargs)
        
    
    class MPIBeamSource:
        def __init__(self, beam_transport):
            self.beam_transport = beam_transport
        def get_beam_slice(self, xi_max, xi_min) -> BeamParticles:
            return self.beam_transport.pull(xi_max, xi_min)
        def get_beam_layer_to_layout(self, plasma_layer_idx):
            return self.beam_transport.pull3d(plasma_layer_idx)
        

    def get_transports(self):
        if self.single_process:
            return [self.initial_source, self.final_drain]
        return [self.MPIBeamSource(self), self.MPIBeamDrain(self)]
        
    def next_step(self):

        is_final = self.final_step
        is_last = self.last_step
        
        self.processed_steps += 1
        self.skip_first = True

        if is_final:
            return
        
        if self.single_process:
            particles = self.final_drain.beam_slice()
            self.initial_source = self.BeamSource(self.cfg, particles)
            self.final_drain = self.BeamDrain(self.cfg)
            return 
        
        if self._rank == self._size - 1:
            particles = self.final_drain.beam_slice()
            if type(particles) == np.ndarray:
                self.send(particles)
            else:
                self.send(particles.particles)
            self.final_drain = self.BeamDrain(self.cfg)
            return
        self.send(np.array([]))

        if self._rank == 0 and not is_last:
            particles = self.recv()
            self.initial_source = self.BeamSource(self.cfg, particles)
            return