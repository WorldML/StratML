"""
cli/config.py
-------------
Config loading, merging, validation, and CLI override application.
No I/O side effects — pure functions only.
"""

from __future__ import annotations

import sys
import yaml
from copy import deepcopy

DEFAULT_CONFIG: dict = {
    "mode": "beginner",
    "dataset": {"path": None, "target_column": None},
    "execution": {"max_iterations": 5, "timeout_per_run": 300, "random_seed": 42, "tune": False},
    "split": {"method": "stratified", "test_size": 0.2},
    "logging": {"enable_mlflow": False, "enable_tensorboard": False, "log_level": "info"},
    "constraints": {"max_memory": None, "max_cpu": None},
    "deep_learning": {
        "enabled": False,
        "architecture": "MLP",
        "epochs": 20,
        "learning_rate": 0.001,
        "batch_size": 32,
    },
    "ablations": {
        "enable_meta_memory": True,
        "enable_value_model": True,
        "condition": "full",
    },
}


def load_yaml(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_cli_overrides(config: dict, args) -> dict:
    config = deepcopy(config)
    if getattr(args, "mode", None) is not None:
        config["mode"] = args.mode
    if getattr(args, "max_iter", None) is not None:
        config["execution"]["max_iterations"] = args.max_iter
    if getattr(args, "path", None) is not None:
        config["dataset"]["path"] = args.path
    dl = config.setdefault("deep_learning", {})
    if getattr(args, "dl", False):
        dl["enabled"] = True
    if getattr(args, "architecture", None) is not None:
        dl["architecture"] = args.architecture
    if getattr(args, "epochs", None) is not None:
        dl["epochs"] = args.epochs
    if getattr(args, "lr", None) is not None:
        dl["learning_rate"] = args.lr
    if getattr(args, "batch_size", None) is not None:
        dl["batch_size"] = args.batch_size
    if getattr(args, "tune", False):
        config["execution"]["tune"] = True

    # P1-21: Ablation controls via CLI flags
    abl = config.setdefault("ablations", {"enable_meta_memory": True, "enable_value_model": True, "condition": "full"})
    if getattr(args, "disable_meta_memory", False):
        abl["enable_meta_memory"] = False
    if getattr(args, "disable_value_model", False):
        abl["enable_value_model"] = False
    ablation_arg = getattr(args, "ablation", None)
    if ablation_arg is not None:
        ab_clean = str(ablation_arg).strip().lower()
        if ab_clean in ("metamemory off", "metamemory_off", "no_metamemory", "disable_metamemory"):
            abl["enable_meta_memory"] = False
        elif ab_clean in ("value model off", "value_model_off", "no_value_model", "disable_value_model"):
            abl["enable_value_model"] = False
        elif ab_clean in ("full", "none"):
            abl["enable_meta_memory"] = True
            abl["enable_value_model"] = True

    mm = abl.get("enable_meta_memory", True)
    vm = abl.get("enable_value_model", True)
    if mm and vm:
        abl["condition"] = "full"
    elif not mm and vm:
        abl["condition"] = "MetaMemory OFF"
    elif mm and not vm:
        abl["condition"] = "Value Model OFF"
    else:
        abl["condition"] = "MetaMemory OFF, Value Model OFF"

    return config


def validate_config(config: dict) -> None:
    """Raise ValueError if config is invalid."""
    if config["dataset"]["path"] is None:
        raise ValueError("dataset.path is required")
    if config["dataset"]["target_column"] is None:
        raise ValueError("dataset.target_column is required")
    if config["mode"] not in ("beginner", "intermediate", "expert"):
        raise ValueError("mode must be beginner | intermediate | expert")


def enforce_mode_rules(config: dict) -> dict:
    if config["mode"] == "beginner":
        config.pop("expert", None)
        config.pop("intermediate", None)
    return config


def resolve(yaml_path: str, args) -> dict:
    """Full config resolution pipeline: load → merge → overrides → validate."""
    config = deep_merge(DEFAULT_CONFIG, load_yaml(yaml_path))
    config = apply_cli_overrides(config, args)
    config = enforce_mode_rules(config)
    return config
