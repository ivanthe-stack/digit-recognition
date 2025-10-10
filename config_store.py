import json
import threading
from pathlib import Path
from typing import Dict, Any

_DEFAULT_CONFIG: Dict[str, Any] = {
    "network_size": [784, 16, 16, 10],
    "learning_rate": 0.1,
    "number_of_epochs": 1000,
    "images_to_train_on": 10000,
    "data_root": "data",
    "param_path": "params.pkl",
    "max_test_examples": None,
}

_CONFIG_LOCK = threading.RLock()
_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def _ensure_config_file() -> None:
    if not _CONFIG_PATH.exists():
        save_config(_DEFAULT_CONFIG)


def _read_config_file() -> Dict[str, Any]:
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a JSON object")
    return data


def _sanitize(config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(_DEFAULT_CONFIG)
    for key in sanitized.keys():
        if key in config:
            sanitized[key] = config[key]
    return sanitized


def _validate(config: Dict[str, Any]) -> None:
    if not isinstance(config["network_size"], (list, tuple)):
        raise ValueError("network_size must be a sequence of integers")
    if len(config["network_size"]) < 2:
        raise ValueError("network_size must contain at least input and output layers")
    for layer in config["network_size"]:
        if not isinstance(layer, int) or layer <= 0:
            raise ValueError("network_size entries must be positive integers")

    for key in ("learning_rate",):
        if config[key] is None:
            raise ValueError(f"{key} cannot be null")
        if not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive number")

    for key in ("number_of_epochs", "images_to_train_on"):
        if config[key] is None:
            raise ValueError(f"{key} cannot be null")
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    if config.get("max_test_examples") is not None:
        value = config["max_test_examples"]
        if not isinstance(value, int) or value <= 0:
            raise ValueError("max_test_examples must be a positive integer or null")

    for key in ("data_root", "param_path"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a non-empty string")


def load_config() -> Dict[str, Any]:
    with _CONFIG_LOCK:
        _ensure_config_file()
        data = _read_config_file()
        config = _sanitize(data)
        _validate(config)
        return config


def save_config(config: Dict[str, Any]) -> None:
    sanitized = _sanitize(config)
    _validate(sanitized)
    with _CONFIG_LOCK:
        with _CONFIG_PATH.open("w", encoding="utf-8") as fh:
            json.dump(sanitized, fh, indent=2, sort_keys=True)


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    with _CONFIG_LOCK:
        current = load_config()
        current.update(updates)
        save_config(current)
        return load_config()


def reset_config() -> Dict[str, Any]:
    with _CONFIG_LOCK:
        save_config(_DEFAULT_CONFIG)
        return _DEFAULT_CONFIG.copy()
