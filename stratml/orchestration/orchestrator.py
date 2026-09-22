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
    ) -> None:
        self.send_profile  = send_profile
        self.send_result   = send_result
        self.split_config  = split_config or SplitConfig(method="stratified")
        self.time_budget   = time_budget
        self.run_id        = run_id
        self.log           = log or (lambda msg: None)
        self.enable_mlflow = enable_mlflow
        self.tune          = tune

    def run(self, dataset_path: str, target_column: str) -> None:
        # ── Phase 1+2: Ingest and profile ────────────────────────────────────
        self.log("  Loading dataset...")
        df, name = load_dataframe(dataset_path)
        dataset  = build_dataset(df, name, target_column)
        profile  = build_profile(dataset)
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
        self.log(f"  Decision [iter 0]: action={action.action_type} | params={action.parameters} | trigger={action.reason.trigger}")

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
            config = build_experiment_config(action, tune=self.tune)
            current_hyperparameters = dict(config.hyperparameters)

            # ── Phase 4b: Apply preprocessing ────────────────────────────────
            clean_split, applied_preprocessing = apply_preprocessing(
                base_split, config.preprocessing, profile, transform_test=False
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

            dl_info = ""
            if dl_result is not None:
                dl_info = f" | device={dl_result.device_used} | epochs={dl_result.epochs_run} | early_stopped={dl_result.early_stopped}"
            self.log(f"  Trained in {run_time:.2f}s{dl_info}")

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

            # ── Budget check ──────────────────────────────────────────────────
            if self.time_budget and total_runtime >= self.time_budget:
                break

            # ── Send result to Team B, receive next ActionDecision ────────────
            self.log(f"  Result   : primary={primary:.4f} | runtime={run_time:.2f}s")
            self.log("  Evaluating signals & deciding next action...")
            action = self.send_result(result)
            self.log(f"  Decision : {action.action_type} | trigger={action.reason.trigger} | confidence={action.confidence:.2f} | next={action.parameters}")

        # ── Test set evaluation ──────────────────────────────────────────────
        self.log(f"\n  --- Test Set Evaluation (Best Validation Score: {best_val_score:.4f} from Iteration {best_iteration}) ---")
        best_model_path = Path("outputs") / self.run_id / "artifacts" / "model.pkl"
        if best_config is not None and best_model_path.exists() and best_config.model_type == "ml":
            try:
                import joblib
                best_model = joblib.load(best_model_path)
                # Apply EXACT preprocessing from the best validation iteration to the test split
                test_split, _ = apply_preprocessing(
                    base_split, best_config.preprocessing, profile, transform_test=True
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
