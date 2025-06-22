# Make the model_1 and scCausalVI classes available directly from the model package
from .model_1 import Model1
from .scCausalVI import scCausalVIModel as scCausalVI

# Export the registry keys
from .base import SCCAUSALVI_REGISTRY_KEYS

__all__ = ["Model1", "scCausalVI", "SCCAUSALVI_REGISTRY_KEYS"]