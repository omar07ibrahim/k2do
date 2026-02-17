"""Configuration module for K2DO."""

from k2do.config.loader import load_config, get_config_path
from k2do.config.schema import Config

__all__ = ["Config", "load_config", "get_config_path"]
