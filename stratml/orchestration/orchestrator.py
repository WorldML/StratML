"""
orchestrator.py
---------------
Phase 9 — Main execution loop. Wires all components together.
Budget enforcement lives here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from stratml.execution.data.loader import load_dataframe
from stratml.execution.data.validator import build_dataset
from stratml.execution.data.profiler import build_profile
from stratml.execution.preprocessing.splitter import split_dataset
from stratml.execution.preprocessing.preprocessor import apply_preprocessing
from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.metrics.metrics_engine import compute_metrics
from stratml.execution.artifacts.artifact_manager import save_artifacts
from stratml.execution.pipelines.ml_pipeline import run_ml_pipeline
from stratml.execution.result_builder import build_experiment_result
from stratml.execution.schemas import (
    ActionDecision, ExperimentResult, DataProfile,
    SplitConfig, ResourceUsage,
)


# Type alias for the two Team B interface callables
SendProfileFn = Callable[[DataProfile], ActionDecision]
SendResultFn  = Callable[[ExperimentResult], ActionDecision]


class ExecutionOrchestrator:
    """
    Drives the full experiment loop.

    Usage:
        orchestrator = ExecutionOrchestrator(
            send_profile=team_b.receive_profile,
            send_result=team_b.receive_result,
        )
        orchestrator.run("data/iris.csv", "species")
    """

    def __init__(
        self,
        send_profile: SendProfileFn,
        send_result: SendResultFn,
        split_config: SplitConfig | None = None,
        time_budget: float | None = None,
        run_id: str = "run",
        log: Optional[Callable[[str], None]] = None,
        enable_mlflow: bool = False,
        tune: bool = False,
        max_iterations: int | None = None,
    ) -> None:
        self.send_profile  = send_profile
        self.send_result   = send_result
        self.split_config  = split_config or SplitConfig(method="stratified")
        self.time_budget   = time_budget
        self.run_id        = run_id
        self.log           = log or (lambda msg: None)
        self.enable_mlflow = enable_mlflow
        self.tune          = tune
        self.max_iterations = max_iterations if max_iterations is not None else 5
        self.decision_iterations = 0
        self.actual_fits = 0
        self.actual_evaluations = 0
        self.total_runtime = 0.0
        self.budget_accounting: dict = {}
        self.manifest: dict | None = None

    def run(self, dataset_path: str, target_column: str) -> None:
        # ── Phase 1+2: Ingest and profile ────────────────────────────────────
        self.log("  Loading dataset...")
        df, name = load_dataframe(dataset_path)
        dataset  = build_dataset(df, name, target_column)
        profile  = build_profile(dataset, random_seed=self.split_config.random_seed)
        self.log(f"  Profiled: {profile.rows} rows x {profile.columns} cols | {profile.problem_type}")

        # ── Phase 3: Split once, reuse across all iterations ─────────────────
        split_cfg = SplitConfig(
            method=self.split_config.method
            if profile.problem_type == "classification"
            else "random",
            test_size=self.split_config.test_size,
            val_size=self.split_config.val_size,
            random_seed=self.split_config.random_seed,
        )
        base_split = split_dataset(dataset, split_cfg, profile.problem_type)
        self.log(f"  Split: train={len(base_split.X_train)} | val={len(base_split.X_val)} | test={len(base_split.X_test)}")

        # ── Send DataProfile to Team B, receive first ActionDecision ─────────
        self.log("  Sending profile to Decision Engine...")
        action: ActionDecision = self.send_profile(profile)
        trigger_iter0 = (
            action.reason.trigger
            if hasattr(action.reason, "trigger")
            else (action.reason.get("trigger", str(action.reason)) if isinstance(action.reason, dict) else str(action.reason))
        )
        self.log(f"  Decision [iter 0]: action={action.action_type} | params={action.parameters} | trigger={trigger_iter0}")

        iteration     = 0
        total_runtime = 0.0
        current_model = action.parameters.get("model_name", "LogisticRegression")
        current_hyperparameters: dict = {}
        best_val_score: float = float("-inf")
        best_config: Optional[ExperimentConfig] = None
        best_iteration: Optional[int] = None

        while action.action_type != "terminate":
            iteration += 1
            self.log(f"\n  --- Iteration {iteration} ---")
            if "model_name" not in action.parameters:
                action.parameters["model_name"] = current_model
            # Carry forward previous hyperparameters for capacity/regularization actions
            if action.action_type in ("increase_model_capacity", "decrease_model_capacity", "modify_regularization", "change_optimizer"):
                for k, v in current_hyperparameters.items():
                    action.parameters.setdefault(k, v)
            current_model = action.parameters.get("model_name", current_model)
            self.log(f"  Training : {current_model} ({action.action_type}) ...")

            # ── Phase 4: Translate ActionDecision → ExperimentConfig ─────────
            config = build_experiment_config(action, tune=self.tune, seed=self.split_config.random_seed)
            current_hyperparameters = dict(config.hyperparameters)

            # ── Phase 4b: Apply preprocessing ────────────────────────────────
            clean_split, applied_preprocessing = apply_preprocessing(
                base_split, config.preprocessing, profile, transform_test=False, seed=self.split_config.random_seed
            )

            # ── Phase 5: Train ────────────────────────────────────────────────
            t_start = time.perf_counter()
            dl_result = None
            if config.model_type == "ml":
                pipeline_result = run_ml_pipeline(config, clean_split)
            else:
                from stratml.execution.pipelines.dl_pipeline import run_dl_pipeline
                tb_dir_train = str(Path("outputs") / self.run_id / "tensorboard" / config.experiment_id)
                pipeline_result = run_dl_pipeline(config, clean_split, tensorboard_log_dir=tb_dir_train)
                dl_result = pipeline_result
            run_time = round(time.perf_counter() - t_start, 4)
            total_runtime += run_time
            self.total_runtime = total_runtime
            self.decision_iterations = iteration
            fits_this_iter = getattr(pipeline_result, "fit_count", 1)
            evals_this_iter = getattr(pipeline_result, "eval_count", 1)
            self.actual_fits += fits_this_iter
            self.actual_evaluations += evals_this_iter

            dl_info = ""
            if dl_result is not None:
                dl_info = f" | device={dl_result.device_used} | epochs={dl_result.epochs_run} | early_stopped={dl_result.early_stopped}"
            self.log(f"  Trained in {run_time:.2f}s (fits={fits_this_iter}){dl_info}")

            # ── Phase 6: Metrics ──────────────────────────────────────────────
            metrics = compute_metrics(
                y_true=clean_split.y_val,
                y_pred=pipeline_result.y_val_pred,
                train_curve=pipeline_result.train_curve,
                val_curve=pipeline_result.val_curve,
                problem_type=profile.problem_type,
            )

            primary = (
                metrics.accuracy
                if metrics.accuracy is not None
                else (metrics.r2 if metrics.r2 is not None else 0.0)
            )
            is_best = primary > best_val_score
            if is_best:
                best_val_score = primary
                best_config = config
                best_iteration = iteration

            # ── Phase 7: Artifacts ────────────────────────────────────────────
            tb_dir = str(Path("outputs") / self.run_id / "tensorboard" / config.experiment_id) \
                if config.model_type == "dl" else None

            # Save this iteration's artifacts in its dedicated subfolder
            iter_artifacts_root = Path("outputs") / self.run_id / "artifacts" / f"iter_{iteration}"
            artifacts = save_artifacts(
                experiment_id=config.experiment_id,
                model=pipeline_result.model,
                metrics=metrics,
                config=config,
                tensorboard_log_dir=tb_dir,
                artifacts_root=iter_artifacts_root,
                dl_result=dl_result,
                enable_mlflow=self.enable_mlflow,
            )

            # Preserve model artifact corresponding to the BEST validation score at the root
            if is_best:
                save_artifacts(
                    experiment_id=config.experiment_id,
                    model=pipeline_result.model,
                    metrics=metrics,
                    config=config,
                    tensorboard_log_dir=tb_dir,
                    artifacts_root=Path("outputs") / self.run_id / "artifacts",
                    dl_result=dl_result,
                    enable_mlflow=False,
                )

            # ── Phase 8: Assemble ExperimentResult ───────────────────────────
            gpu_used = dl_result is not None and dl_result.device_used != "cpu"
            result = build_experiment_result(
                config=config,
                metrics=metrics,
                train_curve=pipeline_result.train_curve,
                validation_curve=pipeline_result.val_curve,
                runtime=run_time,
                resource_usage=ResourceUsage(cpu_time_sec=run_time, gpu_used=gpu_used),
                artifacts=artifacts,
                preprocessing_applied=applied_preprocessing,
                iteration=iteration,
                dataset_name=profile.dataset_name,
                early_stopped=dl_result.early_stopped if dl_result else None,
                best_epoch=dl_result.best_epoch if dl_result else None,
            )

            # ── Budget check (soft timeout) ───────────────────────────────────
            if self.time_budget and total_runtime >= self.time_budget:
                self.log(
                    f"  [Budget] Soft timeout reached: runtime {total_runtime:.2f}s >= budget {self.time_budget:.2f}s. "
                    f"Completed iteration {iteration} cleanly without interrupting in-flight fit."
                )
                break

            # ── Send result to Team B, receive next ActionDecision ────────────
            self.log(f"  Result   : primary={primary:.4f} | runtime={run_time:.2f}s")
            self.log("  Evaluating signals & deciding next action...")
            action = self.send_result(result)
            trigger_next = (
                action.reason.trigger
                if hasattr(action.reason, "trigger")
                else (action.reason.get("trigger", str(action.reason)) if isinstance(action.reason, dict) else str(action.reason))
            )
            self.log(f"  Decision : {action.action_type} | trigger={trigger_next} | confidence={action.confidence:.2f} | next={action.parameters}")

        # ── Test set evaluation ──────────────────────────────────────────────
        self.log(f"\n  --- Test Set Evaluation (Best Validation Score: {best_val_score:.4f} from Iteration {best_iteration}) ---")
        best_model_path = Path("outputs") / self.run_id / "artifacts" / "model.pkl"
        if best_config is not None and best_model_path.exists() and best_config.model_type == "ml":
            try:
                import joblib
                best_model = joblib.load(best_model_path)
                # Apply EXACT preprocessing from the best validation iteration to the test split
                test_split, _ = apply_preprocessing(
                    base_split, best_config.preprocessing, profile, transform_test=True, seed=self.split_config.random_seed
                )
                y_test_pred = best_model.predict(test_split.X_test)
                test_metrics = compute_metrics(
                    y_true=test_split.y_test,
                    y_pred=y_test_pred,
                    train_curve=[],
                    val_curve=[],
                    problem_type=profile.problem_type,
                )
                primary_test = (
                    test_metrics.accuracy
                    if test_metrics.accuracy is not None
                    else (test_metrics.r2 if test_metrics.r2 is not None else 0.0)
                )
                self.log(f"  Test metrics (best model {best_config.model_name}): primary={primary_test:.4f}")
                # Persist test metrics alongside the model artifacts
                import json
                test_metrics_path = Path("outputs") / self.run_id / "artifacts" / "test_metrics.json"
                test_metrics_path.write_text(json.dumps(test_metrics.model_dump(), indent=2))
                self.log(f"  Test metrics saved to {test_metrics_path}")
            except Exception as exc:
                self.log(f"  Test set evaluation failed: {exc}")
        elif best_config is not None and best_config.model_type == "dl":
            pth_path = Path("outputs") / self.run_id / "artifacts" / "model.pth"
            if pth_path.exists():
                try:
                    import torch
                    from stratml.execution.pipelines.dl_architectures import build_model
                    ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
                    test_split, _ = apply_preprocessing(
                        base_split, best_config.preprocessing, profile, transform_test=True
                    )
                    X_test_np = test_split.X_test.values.astype(__import__("numpy").float32)
                    input_dim  = X_test_np.shape[1]
                    hp         = ckpt.get("hyperparameters", best_config.hyperparameters)
                    task       = hp.get("task", "classification")
                    classes    = sorted(base_split.y_train.unique()) if task != "regression" else []
                    output_dim = len(classes) if classes else 1
                    arch       = ckpt.get("architecture", hp.get("architecture", "MLP")).upper()
                    model      = build_model(arch, input_dim, output_dim, hp)
                    model.load_state_dict(ckpt["state_dict"])
                    model.eval()
                    import numpy as np
                    with torch.no_grad():
                        out = model(torch.tensor(X_test_np)).numpy()
                    if task == "regression":
                        y_test_pred = out.squeeze(1)
                    else:
                        label_map = {i: c for i, c in enumerate(classes)}
                        y_test_pred = np.array([label_map[i] for i in out.argmax(axis=1)])
                    test_metrics = compute_metrics(
                        y_true=test_split.y_test, y_pred=y_test_pred,
                        train_curve=[], val_curve=[], problem_type=profile.problem_type,
                    )
                    primary_test = (
                        test_metrics.accuracy
                        if test_metrics.accuracy is not None
                        else (test_metrics.r2 if test_metrics.r2 is not None else 0.0)
                    )
                    self.log(f"  DL Test metrics: primary={primary_test:.4f}")
                    import json
                    test_metrics_path = Path("outputs") / self.run_id / "artifacts" / "test_metrics.json"
                    test_metrics_path.write_text(json.dumps(test_metrics.model_dump(), indent=2))
                    self.log(f"  Test metrics saved to {test_metrics_path}")
                except Exception as exc:
                    self.log(f"  DL test set evaluation failed: {exc}")
        else:
            self.log("  Skipped test set evaluation (no trained model).")

        # ── Computational Budget Accounting ──────────────────────────────────
        import json
        artifacts_dir = Path("outputs") / self.run_id / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        budget_accounting = {
            "run_id": self.run_id,
            "configured_budget": {
                "max_iterations": self.max_iterations,
                "timeout_per_run_seconds": self.time_budget,
                "tune": self.tune,
                "budget_type": "exploratory_tuned" if self.tune else "paper_standard",
                "timeout_semantics": "soft_boundary (in-progress iteration completes fully before timeout is enforced at iteration boundary)",
            },
            "actual_consumption": {
                "decision_iterations": self.decision_iterations,
                "model_evaluations": self.actual_evaluations,
                "model_fits": self.actual_fits,
                "runtime_seconds": round(self.total_runtime, 4),
                "timeout_triggered": bool(self.time_budget and self.total_runtime >= self.time_budget),
            },
            "paper_compliance": {
                "is_paper_standard": (not self.tune),
                "target_paper_config": {
                    "max_iterations": 5,
                    "tune": False,
                },
                "boundary_status": "COMPLIANT_PAPER_STANDARD" if not self.tune else "EXPLORATORY_TUNING_ACTIVE",
                "warning": (
                    f"Tuning (--tune) is enabled! This multiplies model fits per iteration ({self.actual_fits} fits vs {self.decision_iterations} iterations) "
                    "and departs from the target paper budget (tune=false)."
                    if self.tune else None
                ),
            },
        }
        self.budget_accounting = budget_accounting
        budget_path = artifacts_dir / "budget_accounting.json"
        budget_path.write_text(json.dumps(budget_accounting, indent=2))
        self.log(f"  Computational budget accounting saved to {budget_path}")

        # ── Canonical Experiment Manifest (P0-30) ─────────────────────────────
        from stratml.execution.pipelines.ml_pipeline import (
            PAPER_CLASSIFICATION_MODELS,
            PAPER_REGRESSION_MODELS,
        )
        from stratml.decision.actions.action_generator import PAPER_CLASSICAL_ACTIONS
        from stratml.execution.config.ml_mutations import get_paper_mutation_space
        from stratml.decision.llm_control import is_llm_enabled

        engine = getattr(self.send_profile, "__self__", None)

        llm_cfg = None
        llm_file = artifacts_dir / "llm_config.json"
        if llm_file.exists():
            try:
                llm_cfg = json.loads(llm_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if llm_cfg is None and hasattr(engine, "llm_config"):
            llm_cfg = getattr(engine, "llm_config")
        if llm_cfg is None:
            llm_cfg = {
                "llm_mode_enabled": is_llm_enabled(),
                "fallback_behavior": "rule_based",
            }

        import subprocess
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).decode().strip()
        except Exception:
            commit_hash = "unknown"

        manifest = {
            "manifest_version": "1.0",
            "run_id": self.run_id,
            "dataset": {
                "name": profile.dataset_name,
                "path": str(dataset_path),
                "rows": profile.rows,
                "columns": profile.columns,
                "target_column": target_column,
            },
            "dataset_fingerprint": getattr(profile, "dataset_fingerprint", None),
            "task": profile.problem_type,
            "seed": self.split_config.random_seed,
            "budget": self.budget_accounting,
            "model_space": list(
                PAPER_REGRESSION_MODELS if profile.problem_type == "regression" else PAPER_CLASSIFICATION_MODELS
            ),
            "action_space": sorted(list(PAPER_CLASSICAL_ACTIONS)),
            "hyperparameter_mutation_space": get_paper_mutation_space(),
            "llm_configuration": llm_cfg,
            "warm_start_mode_corpus": {
                "history_mode": getattr(engine, "history_mode", "independent"),
                "corpus_path": str(Path("runs/decision_logs/decision_dataset.csv")),
                "meta_memory_path": str(Path("runs/decision_logs/meta_memory.jsonl")),
                "enable_meta_memory": getattr(engine, "enable_meta_memory", True),
                "enable_value_model": getattr(engine, "enable_value_model", True),
            },
            "stratml_version": {
                "version": "0.1.0",
                "commit": commit_hash,
            },
            "evaluation_configuration": {
                "split_method": self.split_config.method,
                "test_size": self.split_config.test_size,
                "val_size": self.split_config.val_size,
                "random_seed": self.split_config.random_seed,
                "primary_metric": "r2" if profile.problem_type == "regression" else "accuracy",
                "best_val_score": best_val_score if best_val_score != float("-inf") else None,
                "best_iteration": best_iteration,
                "best_model_name": getattr(best_config, "model_name", None),
            },
        }

        manifest_path = Path("outputs") / self.run_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.manifest = manifest
        self.log(f"  Experiment manifest saved to {manifest_path}")

