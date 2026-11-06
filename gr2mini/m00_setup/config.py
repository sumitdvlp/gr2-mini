"""YAML config loading with single inheritance via an `extends:` key.  (see README §1)"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import copy

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a NEW dict equal to `base` recursively updated with `override`.

    Rules:
        * key in both, and BOTH values are dicts  -> merge those dicts recursively
        * otherwise                               -> the value from `override` wins
          (lists are replaced, not concatenated)
        * keys only in one of them are kept
        * neither `base` nor `override` (nor their nested dicts) may be mutated

    Example:
        deep_merge({"a": {"x": 1, "y": 2}, "b": [1, 2]}, {"a": {"y": 3}, "b": [9]})
        == {"a": {"x": 1, "y": 3}, "b": [9]}

    Hint: `copy.deepcopy` is your friend for "don't mutate".
    """
    merged = {}
    
    # Process all keys in override
    for key, override_value in override.items():
        base_value = base.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = copy.deepcopy(override_value) 

    # Process remaining keys unique to base
    for key, base_value in base.items():
        if key not in merged:
            merged[key] = copy.deepcopy(base_value)
            
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving an optional top-level `extends:` key.

    * `extends: other.yaml` is resolved RELATIVE TO THE DIRECTORY OF `path`.
    * Parents can themselves extend other files (recurse).
    * Child values are deep-merged on top of the parent's.
    * The returned dict does NOT contain the key "extends".
    * Use `yaml.safe_load`; an empty file means {}.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    yaml_payload = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(yaml_payload, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping (dict) at the top level.")
    if "extends" in yaml_payload:
        parent_path = config_path.parent / yaml_payload["extends"]
        parent_config = load_config(parent_path)
        child_config = {k: v for k, v in yaml_payload.items() if k != "extends"}
        return deep_merge(parent_config, child_config)
    else:
        return yaml_payload


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Return a NEW config with command-line overrides applied (does not mutate `cfg`).

    Each override is "dotted.key.path=value", e.g. "rqvae.use_ema=false" or "rqvae.hidden_dims=[64, 32]".
        * The value is parsed with yaml.safe_load, so "false" -> False, "0.1" -> 0.1, "null" -> None,
          "[64, 32]" -> [64, 32], and "abc" stays the string "abc".
          Gotcha: PyYAML follows YAML 1.1, where "1e-3" (no dot) is a STRING. Write 0.001 or 1.0e-3.
        * Missing intermediate dicts are created.
        * Split on the FIRST "=" only (values may contain "=").
        * An override without "=" -> ValueError.
    None or [] -> an unchanged copy.

    Used by every run.py as:  --set rqvae.use_ema=false --set rqvae.epochs=10
    """
    new_cfg = copy.deepcopy(cfg)
    if overrides is None:
        return new_cfg
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must contain '=': {override}")
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")
        value = __import__("yaml").safe_load(value_str)
        d = new_cfg
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    return new_cfg
