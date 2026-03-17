"""
Simple configuration system using plain Python and YAML.

Replaces OmegaConf/Hydra with zero external dependencies beyond PyYAML.
"""

import yaml
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field


class Config(dict):
    """
    Dict subclass with attribute access.

    Allows cfg.model.embed_dim instead of cfg['model']['embed_dim']
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict) and not isinstance(value, Config):
                value = Config(value)
                self[name] = value
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def load_yaml(path: Path) -> Config:
    """Load a YAML file into a Config object."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data) if data else Config()


def merge_configs(*configs: dict) -> Config:
    """Deep merge multiple configs, later ones override earlier."""
    result = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return Config(result)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config(config_path: str | Path) -> Config:
    """
    Load config with defaults system.

    Looks for 'defaults' key and loads referenced sub-configs.
    """
    config_path = Path(config_path)
    cfg = load_yaml(config_path)

    # Handle defaults
    if "defaults" in cfg:
        config_dir = config_path.parent
        merged = Config()

        for default in cfg["defaults"]:
            if isinstance(default, dict):
                for key, name in default.items():
                    sub_path = config_dir / key / f"{name}.yaml"
                    if sub_path.exists():
                        sub_cfg = load_yaml(sub_path)
                        merged = merge_configs(merged, {key: sub_cfg})

        # Remove defaults key and merge base config
        del cfg["defaults"]
        cfg = merge_configs(merged, cfg)

    return cfg


def config_to_dict(cfg: Config) -> dict:
    """Convert Config to plain dict (for serialization)."""
    result = {}
    for key, value in cfg.items():
        if isinstance(value, Config):
            result[key] = config_to_dict(value)
        else:
            result[key] = value
    return result
