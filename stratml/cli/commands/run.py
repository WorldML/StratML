"""cli/commands/run.py — `stratml run` command."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

_console = Console(highlight=False)


def run_pipeline(args) -> None:
    from stratml.cli.config import resolve, validate_config

    config = resolve(args.config, args)
    try:
        validate_config(config)
    except ValueError as e:
        _console.print()
        _console.print(Panel.fit(
            f"[bold red]Invalid config[/bold red]\n[white]{e}[/white]",
            border_style="red", padding=(0, 2),
        ))
        _console.print()
        sys.exit(1)

    d = config["dataset"]
    e = config["execution"]

    if args.dry_run:
        _print_dry_run(config, d, e)
        return

    _console.print()
    _console.print(Panel.fit(
        f"[bold white]StratML AutoML[/bold white]  [dim]—[/dim]  [cyan]{Path(d['path']).stem}[/cyan]  [dim]->[/dim]  [green]{d['target_column']}[/green]",
        border_style="bright_blue", padding=(0, 2),
    ))
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim"); grid.add_column()
    grid.add_row("Mode",    f"[bold]{config['mode']}[/bold]")
    grid.add_row("Dataset", f"[white]{d['path']}[/white]")
    grid.add_row("Budget",  f"[bold]{e['max_iterations']}[/bold] iterations")
    _console.print(grid)
    _console.print()

    # -- DEMO INTERCEPT (remove when real pipeline is ready) ------------------
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
    _stem = Path(d["path"]).stem
    if _stem in _DEMO_MAP:
        import importlib
        _root = str(Path(__file__).resolve().parents[3])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        _mod    = importlib.import_module(_DEMO_MAP[_stem])
        run_id  = f"{_stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        _mod.run(run_id=run_id)
        return
    # -- END DEMO INTERCEPT ---------------------------------------------------

    from stratml.decision.engine import DecisionEngine
    from stratml.orchestration.orchestrator import ExecutionOrchestrator
    from stratml.execution.schemas import SplitConfig

    dl_cfg     = config.get("deep_learning", {})
    dl_enabled = dl_cfg.get("enabled", False)

    if dl_enabled:
        arch           = dl_cfg.get("architecture", "MLP").upper()
        allowed_models = [arch if arch in ("MLP", "CNN1D", "RNN") else "MLP"]
        dl_hyperparams = {
            "architecture":  arch,
            "epochs":        dl_cfg.get("epochs", 20),
            "learning_rate": dl_cfg.get("learning_rate", 0.001),
            "batch_size":    dl_cfg.get("batch_size", 32),
        }
    else:
        allowed_models = (
            config.get("intermediate", {}).get("allowed_models")
            or config.get("expert", {}).get("allowed_models")
            or None
        )
        dl_hyperparams = None

    import uuid
    seed = e.get("random_seed", 42)
    dataset_name = Path(d["path"]).stem
    run_id       = f"{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_dir      = Path("outputs") / run_id

    engine = DecisionEngine(
        max_iterations=e["max_iterations"],
        time_budget=e.get("timeout_per_run"),
        allowed_models=allowed_models,
        run_id=run_id,
        dl_hyperparams=dl_hyperparams,
        seed=seed,
        llm_mode=config.get("llm_mode", e.get("llm_mode")),
    )

    ExecutionOrchestrator(
        send_profile=engine.receive_profile,
        send_result=engine.receive_result,
        split_config=SplitConfig(
            method=config["split"]["method"],
            test_size=config["split"]["test_size"],
            random_seed=seed,
        ),
        time_budget=e.get("timeout_per_run"),
        run_id=run_id,
        log=_console.print,
        tune=config.get("execution", {}).get("tune", False),
    ).run(d["path"], d["target_column"])

    _console.print()
    _console.rule("[bold green]Run Complete[/bold green]", style="green")
    _console.print()
    footer = Table.grid(padding=(0, 3))
    footer.add_column(style="dim", width=12); footer.add_column(style="white")
    footer.add_row("Run ID",  f"[bold]{run_id}[/bold]")
    footer.add_row("Output",  str(out_dir))
    _console.print(footer)
    _console.print()

    _generate_report(run_id, dataset_name, out_dir)
    _prompt_download(out_dir)


def _print_dry_run(config: dict, d: dict, e: dict) -> None:
    _console.print()
    _console.print(Panel.fit(
        "[bold white]Dry Run[/bold white]  [dim]— Resolved Config[/dim]",
        border_style="yellow", padding=(0, 2),
    ))
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim", width=18); grid.add_column()
    grid.add_row("Mode",           f"[bold]{config['mode']}[/bold]")
    grid.add_row("Dataset",        f"[white]{d['path']}[/white]")
    grid.add_row("Target",         f"[green]{d['target_column']}[/green]")
    grid.add_row("Max iterations", f"[bold]{e['max_iterations']}[/bold]")
    grid.add_row("Timeout/run",    f"[white]{e['timeout_per_run']}s[/white]")
    grid.add_row("MLflow",         f"[white]{config['logging']['enable_mlflow']}[/white]")
    dl = config.get("deep_learning", {})
    if dl.get("enabled"):
        grid.add_row("DL Mode", f"[bold green]enabled[/bold green]  [dim]|[/dim]  Arch: [cyan]{dl.get('architecture')}[/cyan]  [dim]|[/dim]  Epochs: [white]{dl.get('epochs')}[/white]")
    _console.print(grid)
    _console.print()


def _generate_report(run_id: str, dataset_name: str, out_dir: Path) -> None:
    from stratml.reporting.report_generator import generate_report, generate_model_script
    try:
        pdf     = generate_report(run_id=run_id, dataset_name=dataset_name, output_dir=out_dir)
        log_dir = out_dir / "decision_logs"
        records = [
            json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(log_dir.glob(f"{run_id}_*.json"))
        ]
        script = generate_model_script(run_id=run_id, output_dir=out_dir, records=records)
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="dim", width=12); grid.add_column(style="dim")
        grid.add_row("Report",     str(pdf))
        grid.add_row("Comparison", str(out_dir / "comparison.csv"))
        grid.add_row("Model.py",   str(script))
        _console.print(grid)
        _console.print()
    except Exception as ex:
        _console.print(f"  [yellow]Warning:[/yellow] Report generation failed: {ex}\n")


def _prompt_download(out_dir: Path) -> None:
    model_pkl = out_dir / "artifacts" / "model.pkl"
    if not model_pkl.exists():
        return
    answer = _console.input("  [dim]Download best model files (model.pkl + model.py)?[/dim] [bold]\\[y/N][/bold]: ").strip().lower()
    if answer == "y":
        import shutil
        shutil.copy2(model_pkl, Path.cwd() / "best_model.pkl")
        model_script = out_dir / "artifacts" / "model.py"
        if model_script.exists():
            shutil.copy2(model_script, Path.cwd() / "model.py")
        _console.print(f"  [green]Saved:[/green] best_model.pkl + model.py\n")
