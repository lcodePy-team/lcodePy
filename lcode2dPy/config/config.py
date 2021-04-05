from copy import copy
from typing import Dict, Any, Optional


class Config:
    config_values: Dict[str, str]

    def __init__(self):
        self.config_values = {}

    def get(self, option_name: str, fallback: str = '') -> str:
        if option_name in self.config_values:
            return self.config_values.get(option_name)
        return fallback

    def getbool(self, option_name: str, fallback: bool = 0) -> bool:
        str_value = self.config_values.get(option_name).lower()
        if str_value == 'true':
            return True
        elif str_value == 'false':
            return False
        else:
            return fallback

    def getint(self, option_name: str, fallback: int = 0) -> int:
        try:
            return int(self.config_values.get(option_name))
        except ValueError:
            return fallback

    def getfloat(self, option_name: str, fallback: float = 0.0) -> float:
        try:
            return float(self.get(option_name))
        except ValueError:
            return fallback

    def set(self, option_name: str, option_value: Any):
        self.config_values[option_name] = str(option_value)

    def __copy__(self) -> 'Config':
        ret = Config()
        ret.config_values = copy(self.config_values)
        return ret