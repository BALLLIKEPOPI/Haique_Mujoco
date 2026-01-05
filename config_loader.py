import json
from pathlib import Path
from typing import Any, Dict, Optional


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load config from a JSON-syntax YAML file.

    This repository uses a .yaml file written in strict JSON syntax so it can be
    parsed with Python stdlib (no PyYAML dependency).
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse {cfg_path} as JSON-syntax YAML: {e}"
        ) from e


def get_mode_config(mode: str, path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(path)
    if mode not in cfg or not isinstance(cfg[mode], dict):
        raise KeyError(f"Missing mode section '{mode}' in {path}")

    common = cfg.get("common")
    if isinstance(common, dict):
        return _deep_merge(common, cfg[mode])
    return cfg[mode]


def get_value(cfg: Dict[str, Any], keys: str, default: Any = None) -> Any:
    """Fetch nested key path like 'physical.mass'."""
    cur: Any = cfg
    for part in keys.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
