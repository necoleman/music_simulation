"""luthier - simulate stringed (and, later, wind) instruments before you build them."""
from . import materials
from .fem import Model, Modes, Section, modal_analysis
from .materials import Orthotropic

__version__ = "0.1.0"
__all__ = ["Model", "Modes", "Section", "Orthotropic", "modal_analysis", "materials"]
