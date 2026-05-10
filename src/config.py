"""Configuration loader from YAML files."""
from __future__ import annotations

from pathlib import Path

import yaml


class Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str):
        val = self._data.get(name)
        if isinstance(val, dict):
            return Config(val)
        return val

    def get(self, key: str, default=None):
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return Config(val)
        return val

    def to_dict(self) -> dict:
        return self._data

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as f:
            return cls(yaml.safe_load(f))
