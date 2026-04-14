from importlib.metadata import version

from . import pl, pp, tl
from .data_paths import data

__all__ = ["pl", "pp", "tl", "data"]

__version__ = version("Bacotype")
