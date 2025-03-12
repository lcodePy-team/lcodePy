
from ...config import Config
from ..utils import Diagnostic
from .StrategyCircular import _CIRC_FXi
from .Strategy3D import _3D_FXi


class Strategy:
    def __new__(cls, config: Config, diagnostic: Diagnostic):
        if config.get('geometry') == '3d':
            return _3D_FXi(config, diagnostic)
        if config.get('geometry') == '2d':
            return _CIRC_FXi(config, diagnostic)
        raise ValueError(f'Unknown geometry: {config.get("geometry")}')