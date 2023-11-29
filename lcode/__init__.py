"""Quasistatic code for plasma wakefield simulation."""

from .alt_beam_generator.beam_generator import generate_beam
# from .simulation import test
from .simulation import Simulation #as Cartesian3dSimulation
from .diagnostics.diagnostics_3d import (
    DiagnosticsFXi, DiagnosticsColormaps, DiagnosticsTransverse, SaveRunState
)

__version__ = "0.1.0"
