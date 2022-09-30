from mpi4py import MPI
import time
from lcode2dPy.push_solver import PusherAndSolver
from lcode2dPy.beam.beam_slice import BeamSlice
from lcode2dPy.beam.beam_io import MemoryBeamSource, MemoryBeamDrain
#  from lcode2dPy.push_solver_3d import PushAndSolver3d
import numpy as np
from lcode2dPy.config.default_config import default_config
from lcode2dPy.beam.beam_generator import make_beam
from lcode2dPy.plasma.initialization import init_plasma
from lcode2dPy.mpi.beam_io import MPIBeamDrain, MPIBeamSource


class Simulation:
    def __init__(self, config=default_config,
                 beam_generator=make_beam, beam_pars=None,
                 diagnostics=None):
        self.config = config
        if config.get('geometry') == '3d':
            pass
#            self.push_solver = PushAndSolver3d(self.config) # 3d
        elif config.get('geometry') == 'circ' or config.get('geometry') == 'c':
            self.push_solver = PusherAndSolver(self.config)  # circ
        elif config.get('geometry') == 'plane':
            self.push_solver = PusherAndSolver(self.config)  # 2d_plane

        self.beam_generator = beam_generator
        self.beam_pars = beam_pars

        self.current_time = 0.
        self.beam_source = None
        self.beam_drain = None
        self.t_step = config.getfloat('time-step')
        self.current_time = 0. + self.t_step*(1+MPI.COMM_WORLD.rank)
        self.diagnostics = diagnostics

    def step(self, N_steps):
        # t step function, makes N_steps time steps.
        # Beam generation
        if self.beam_source is None:
            if MPI.COMM_WORLD.rank == 0:
                beam_particles =\
                    self.beam_generator(self.config, **self.beam_pars)
                beam_particle_dtype = \
                    np.dtype([('xi', 'f8'), ('r', 'f8'),
                              ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'),
                              ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])
                beam_particles = np.array(
                    list(map(tuple, beam_particles.to_numpy())),
                    dtype=beam_particle_dtype)

                beam_slice = BeamSlice(beam_particles.size, beam_particles)
            else:
                pdtype =\
                    np.dtype([('xi', 'f8'), ('r', 'f8'),
                              ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'),
                              ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])

                beam_slice = BeamSlice(0, particles=np.zeros(0, dtype=pdtype))
        rank = MPI.COMM_WORLD.rank
        size = MPI.COMM_WORLD.size
        steps = N_steps//size + (1 if N_steps % size != 0 else 0)
        for t_i in range(steps):
            n = min(size, N_steps - t_i * size)
            if rank == 0:
                t_start = time.time()
                print(f"Start of round {t_i}, {n} workers")
            if rank >= N_steps - t_i * size:
                return
            self.beam_source = MPIBeamSource(n, MemoryBeamSource(beam_slice))
            self.beam_drain = MPIBeamDrain(n, MemoryBeamDrain())

            fields, plasma_particles = init_plasma(self.config)
            plasma_particles_new, fields_new = self.push_solver.step_dt(
                plasma_particles, fields, self.beam_source, self.beam_drain,
                self.current_time, self.diagnostics)
            if N_steps - t_i * size < size:
                return
            self.current_time += self.t_step*size
            beam_slice = self.beam_drain.refresh()
            if rank == 0:
                t_end = time.time()
                diff = (t_end-t_start)/60
                print(f"Round time: {diff:.2f} minutes")
                print(f"Estimate {diff*(steps-t_i+1):.2f} minutes\n")
