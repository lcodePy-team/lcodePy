import numpy as np
from typing_extensions import Annotated

from lcode.config import Config
from lcode.beam.data import BeamParticlesBase

particle_dtype3d = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
                             ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
                             ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])


class BeamParticles(BeamParticlesBase):

    xi: Annotated[np.ndarray, 'f8']
    x: Annotated[np.ndarray, 'f8']
    y: Annotated[np.ndarray, 'f8']
    px: Annotated[np.ndarray, 'f8']
    py: Annotated[np.ndarray, 'f8']
    pz: Annotated[np.ndarray, 'f8']
    q_m: Annotated[np.ndarray, 'f8']
    q_norm: Annotated[np.ndarray, 'f8']
    id: Annotated[np.ndarray, 'i8']

    dt: Annotated[np.ndarray, 'f8', 'extra']
    remaining_steps: Annotated[np.ndarray, 'i8', 'extra']
    lost: Annotated[np.ndarray, '?', 'extra']

