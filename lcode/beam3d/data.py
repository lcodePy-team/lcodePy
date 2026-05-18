import numpy as np
from typing_extensions import Annotated

from lcode.config import Config
from lcode.beam.data import BeamParticlesBase

particle_dtype3d = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
                             ('ux', 'f8'), ('uy', 'f8'), ('uz', 'f8'),
                             ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])


class BeamParticles(BeamParticlesBase):

    xi: Annotated[np.ndarray, 'f8']
    x: Annotated[np.ndarray, 'f8']
    y: Annotated[np.ndarray, 'f8']
    ux: Annotated[np.ndarray, 'f8']
    uy: Annotated[np.ndarray, 'f8']
    uz: Annotated[np.ndarray, 'f8']
    q_m: Annotated[np.ndarray, 'f8']
    q_norm: Annotated[np.ndarray, 'f8']
    id: Annotated[np.ndarray, 'i8']

    dt: Annotated[np.ndarray, 'f8', 'extra']
    remaining_steps: Annotated[np.ndarray, 'i8', 'extra']
    lost: Annotated[np.ndarray, '?', 'extra']

