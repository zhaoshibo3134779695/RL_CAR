"""Yellow-track perception, local planning, and safe control."""

from .config import ConfigError, load_config, validate_config
from .pipeline import AvoidancePipeline

__all__ = ["AvoidancePipeline", "ConfigError", "load_config", "validate_config"]
