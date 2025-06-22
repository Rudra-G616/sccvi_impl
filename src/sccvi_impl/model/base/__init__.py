# Import from scvi library for base registry keys
from scvi import REGISTRY_KEYS

# Import the local _REGISTRY_KEYS_NT class
from ._utils import _REGISTRY_KEYS_NT

# Create a constant for the registry keys
SCCAUSALVI_REGISTRY_KEYS = _REGISTRY_KEYS_NT()