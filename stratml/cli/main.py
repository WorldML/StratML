#!/usr/bin/env python
"""
stratml/cli/main.py
-------------------
Entry point — argument parsing and command dispatch only.
All command logic lives in cli/commands/.
All config logic lives in cli/config.py.
"""

import sys
from pathlib import Path

# Ensure project root is importable when invoked as a script (e.g. via the
# shell wrapper installed by install.sh: `python stratml/cli/main.py ...`)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import argparse


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)


def apply_cli_overrides(config, args):
    config = deepcopy(config)
    if getattr(args, "mode", None) is not None:
        config["mode"] = args.mode
    if getattr(args, "max_iter", None) is not None:
        config["execution"]["max_iterations"] = args.max_iter
    if getattr(args, "path", None) is not None:
        config["dataset"]["path"] = args.path
    return config


def validate_config(config):
    if config["dataset"]["path"] is None:
        raise ValueError("dataset.path is required")
    if config["dataset"]["target_column"] is None:
        raise ValueError("dataset.target_column is required")
    if config["mode"] not in ["beginner", "intermediate", "expert"]:
        raise ValueError("Invalid mode")
    return True


def enforce_mode_rules(config):
    mode = config["mode"]
    if mode == "beginner":
        config.pop("expert", None)
        config.pop("intermediate", None)
    return config


def run_pipeline(args):
    yaml_config = load_yaml(args.config)
    config = deep_merge(DEFAULT_CONFIG, yaml_config)
    config = apply_cli_overrides(config, args)
    config = enforce_mode_rules(config)

    try:
        validate_config(config)
    except Exception as e:
        print(f"\n  [Invalid config] {e}\n")
        sys.exit(1)

    sep = "-" * 44
    d = config["dataset"]
    e = config["execution"]

    if args.dry_run:
        print()
        print("  Dry Run - Resolved Config")
        print(f"  {sep}")
        print(f"  Mode          : {config['mode']}")
        print(f"  Dataset       : {d['path']}")
        print(f"  Target        : {d['target_column']}")
        print(f"  Max iterations: {e['max_iterations']}")
        print(f"  Timeout/run   : {e['timeout_per_run']}s")
        print(f"  Random seed   : {e['random_seed']}")
        print(f"  MLflow        : {config['logging']['enable_mlflow']}")
        print(f"  TensorBoard   : {config['logging']['enable_tensorboard']}")
        print(f"  {sep}")
        print()
        return

    # -- DEMO INTERCEPT (delete this block when real pipeline is ready) ----
    _DEMO_MAP = {
        "titanic":             "demo.demo_titanic",
        "pima":                "demo.demo_pima",
        "wine_quality_red":    "demo.demo_wine_quality",
        "california_housing":  "demo.demo_california_housing",
        "creditcard":          "demo.demo_creditcard",
        "mnist":               "demo.demo_mnist",
        "mnist_dl":            "demo.demo_mnist_dl",
        "energydata_complete": "demo.demo_energy",
        "cifar10":             "demo.demo_cifar10",
        "imdb":                "demo.demo_imdb",
    }
    _dataset_stem = Path(d["path"]).stem
    if _dataset_stem in _DEMO_MAP:
        import importlib
        _root = str(Path(__file__).resolve().parents[2])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        _mod = importlib.import_module(_DEMO_MAP[_dataset_stem])
        from datetime import datetime as _dt, timezone as _tz
        _run_id = f"{_dataset_stem}_{_dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S')}"
        _mod.run(run_id=_run_id)
        return
    # -- END DEMO INTERCEPT -----------------------------------------------

    print()
    print("  AutoML Pipeline Starting")
    print(f"  {sep}")
    print(f"  Mode    : {config['mode']}")
    print(f"  Dataset : {d['path']}")
    print(f"  Target  : {d['target_column']}")
    print(f"  Budget  : {e['max_iterations']} iterations")
    print(f"  {sep}")

    _root = str(Path(__file__).resolve().parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from stratml.decision.engine import DecisionEngine
    from stratml.orchestration.orchestrator import ExecutionOrchestrator
    from stratml.execution.schemas import SplitConfig
    from stratml.reporting.report_generator import generate_report
    from pathlib import Path as _Path
    import shutil as _shutil
    import uuid as _uuid
    from datetime import datetime as _dt, timezone as _tz

    allowed_models = (
        config.get("intermediate", {}).get("allowed_models")
        or config.get("expert", {}).get("allowed_models")
        or None
    )

    seed = e.get("random_seed", 42)
    dataset_name = _Path(d["path"]).stem
    run_id  = f"{dataset_name}_{_dt.now(_tz.utc).strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"
    out_dir = _Path("outputs") / run_id

    engine = DecisionEngine(
        max_iterations=e["max_iterations"],
        time_budget=e.get("timeout_per_run"),
        allowed_models=allowed_models,
        run_id=run_id,
        seed=seed,
        llm_mode=config.get("llm_mode", e.get("llm_mode")),
    )

    def _log(msg): print(msg)

    orchestrator = ExecutionOrchestrator(
        send_profile=engine.receive_profile,
        send_result=engine.receive_result,
        split_config=SplitConfig(
            method=config["split"]["method"],
            test_size=config["split"]["test_size"],
            random_seed=seed,
        ),
        time_budget=e.get("timeout_per_run"),
        run_id=run_id,
        log=_log,
    )

    orchestrator.run(d["path"], d["target_column"])
    print("  Run complete.\n")
    sep = "-" * 44
    print(f"  {sep}")
    print(f"  Run ID  : {run_id}")
    print(f"  Output  : {out_dir}")
    print(f"  {sep}\n")

    # PDF Report + comparison files + model.py
    from stratml.reporting.report_generator import generate_model_script
    import json as _json
    try:
        pdf = generate_report(run_id=run_id, dataset_name=dataset_name, output_dir=out_dir)
        # Load records for model script
        log_dir = out_dir / "decision_logs"
        records = [_json.loads(f.read_text(encoding="utf-8")) for f in sorted(log_dir.glob(f"{run_id}_*.json"))]
        model_script = generate_model_script(run_id=run_id, output_dir=out_dir, records=records)
        print(f"  Report    : {pdf}")
        print(f"  Comparison: {out_dir / 'comparison.csv'}")
        print(f"  Model.py  : {model_script}\n")
    except Exception as ex:
        print(f"  [Warning] Report/model generation failed: {ex}\n")

    # Model download prompt — files are already in outputs/<run_id>/
    model_pkl    = out_dir / "artifacts" / run_id / "model.pkl"
    model_script = out_dir / "model.py"
    if model_pkl.exists():
        answer = input("  Download best model files (model.pkl + model.py)? [y/N]: ").strip().lower()
        if answer == "y":
            print(f"  Files saved at:{model_pkl} and {model_script}")


def validate_config_cmd(args):
    config = load_yaml(args.config)
    try:
        validate_config(config)
        print(f"\n  Config OK — {args.config}\n")
    except Exception as e:
        print(f"\n  Invalid config: {e}\n")
        sys.exit(1)


def profile_data(args):
    import json

    # Ensure stratml package is importable (project root = 2 levels up from stratml/cli/)
    _root = str(Path(__file__).resolve().parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from stratml.execution.data.loader import load_dataframe
    from stratml.execution.data.validator import build_dataset
    from stratml.execution.data.profiler import build_profile

    outputs_dir = Path(__file__).resolve().parents[3] / "outputs"

    df, dataset_name = load_dataframe(args.dataset)
    dataset = build_dataset(df, dataset_name, args.target)
    profile = build_profile(dataset)

    out_dir = outputs_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data_profile.json"
    out_file.write_text(json.dumps(profile.model_dump(), indent=2))

    p = profile
    sep = "-" * 44

    print()
    print("  Dataset Profile")
    print(f"  {sep}")
    print(f"  Dataset       : {p.dataset_name}")
    print(f"  Type          : {p.dataset_type}  |  Problem: {p.problem_type}")
    print(f"  Shape         : {p.rows} rows x {p.columns} columns")
    print(f"  Target        : {p.target_column}")
    print(f"  {sep}")
    print(f"  Features      : {len(p.numerical_columns)} numerical, {len(p.categorical_columns)} categorical")
    print(f"  Missing ratio : {p.missing_value_ratio:.2%}")

    if p.class_distribution:
        dist_str = "  |  ".join(f"{k}: {v}" for k, v in p.class_distribution.items())
        print(f"  Classes       : {dist_str}")

    print(f"  {sep}")
    print(f"  Feature Summary")
    print(f"  {'Name':<24} {'Type':<10} {'Unique':>6}  {'Missing':>8}  {'Dist'}")
    print(f"  {'-'*24} {'-'*10} {'-'*6}  {'-'*8}  {'-'*10}")
    for f in p.feature_summary:
        print(f"  {f.name:<24} {f.dtype:<10} {f.unique_values:>6}  {f.missing_percentage:>7.1f}%  {f.distribution}")

    print(f"  {sep}")
    print(f"  Recommended metrics : {', '.join(p.recommended_metrics)}")
    print(f"  Saved to            : {out_file}")
    print()


def init_config():
    with open("config.yaml", "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, sort_keys=False)
    print("\n  Created config.yaml with default settings.")
    print("  Edit dataset.path and dataset.target_column before running.\n")


def doctor_check():
    import importlib
    sep = "-" * 44
    packages = ["pandas", "numpy", "sklearn", "torch", "pydantic", "mlflow", "yaml"]
    print()
    print("  Environment Check")
    print(f"  {sep}")
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            version = getattr(mod, "__version__", "ok")
            print(f"  {'ok':<6} {pkg:<20} {version}")
        except ImportError:
            print(f"  {'MISSING':<6} {pkg}")
    print(f"  {sep}")
    print()


def main():
    parser = argparse.ArgumentParser(prog="stratml")
    sub    = parser.add_subparsers(dest="command", required=True)

    # ── run ───────────────────────────────────────────────────────────────────
    run = sub.add_parser("run", help="Run the AutoML pipeline")
    run.add_argument("config")
    run.add_argument("--path")
    run.add_argument("--mode", choices=["beginner", "intermediate", "expert"])
    run.add_argument("--max-iter", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--dl", action="store_true")
    run.add_argument("--tune", action="store_true", help="Enable RandomizedSearchCV for ML models")
    run.add_argument("--architecture", choices=["MLP", "CNN1D", "RNN"])
    run.add_argument("--epochs", type=int)
    run.add_argument("--lr", type=float)
    run.add_argument("--batch-size", type=int)

    # ── validate-config ───────────────────────────────────────────────────────
    vc = sub.add_parser("validate-config", help="Validate a config file")
    vc.add_argument("config")

    # ── profile-data ──────────────────────────────────────────────────────────
    pd = sub.add_parser("profile-data", help="Profile a dataset")
    pd.add_argument("dataset")
    pd.add_argument("target")

    # ── init / doctor ─────────────────────────────────────────────────────────
    sub.add_parser("init",   help="Create a default config.yaml")
    sub.add_parser("doctor", help="Check environment dependencies")

    args = parser.parse_args()

    if args.command == "run":
        from stratml.cli.commands.run import run_pipeline
        run_pipeline(args)
    elif args.command == "validate-config":
        from stratml.cli.commands.utils import validate_config_cmd
        validate_config_cmd(args)
    elif args.command == "profile-data":
        from stratml.cli.commands.profile import profile_data
        profile_data(args)
    elif args.command == "init":
        from stratml.cli.commands.utils import init_config
        init_config()
    elif args.command == "doctor":
        from stratml.cli.commands.utils import doctor_check
        doctor_check()


if __name__ == "__main__":
    main()
