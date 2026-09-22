"""
test_p0_21_30_integrity.py
--------------------------
End-to-end experimental integrity test suite for P0-29.
Verifies the complete loop:
  dataset -> initial state -> candidate generation -> decision -> execution
  -> evaluation -> memory update -> next state

Covers the 8 canonical experimental lifecycle conditions:
  1. Successful experiment
  2. Failed experiment
  3. Improvement
  4. Degradation
  5. Termination
  6. Budget exhaustion
  7. Warm-start experience available
  8. Insufficient warm-start experience
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stratml.core.schemas import (
    ActionDecision,
    CandidateAction,
    DecisionReason,
    PreprocessingConfig,
    StateObject,
)
from stratml.execution.schemas import (
    DataProfile,
    ExperimentMetrics,
    ExperimentResult,
    FeatureInfo,
    ResourceUsage,
    ArtifactRefs,
    SplitConfig,
)
from stratml.decision.engine import DecisionEngine
from stratml.decision.agents import coordinator_agent, evaluator_agent
from stratml.decision.learning import dataset_builder, meta_memory, value_model
from stratml.orchestration.orchestrator import ExecutionOrchestrator


@pytest.fixture(autouse=True)
def disable_external_apis(monkeypatch):
    """Ensure tests run offline without hitting external APIs."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def _make_sample_dataset(tmp_path: Path, n_rows: int = 60, problem_type: str = "classification") -> Path:
    csv_path = tmp_path / "sample_data.csv"
    np.random.seed(42)
    df = pd.DataFrame({
        "num_1": np.random.randn(n_rows),
        "num_2": np.random.randn(n_rows),
        "target": np.random.choice([0, 1], size=n_rows) if problem_type == "classification" else np.random.randn(n_rows),
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_profile(problem_type="classification", fingerprint="fp_int_123"):
    return DataProfile(
        dataset_name="int_test_data",
        dataset_type="tabular",
        rows=100,
        columns=3,
        target_column="target",
        problem_type=problem_type,
        numerical_columns=["num_1", "num_2"],
        categorical_columns=[],
        missing_value_ratio=0.0,
        dataset_fingerprint=fingerprint,
        feature_summary=[
            FeatureInfo(name="num_1", dtype="float64", unique_values=100, missing_percentage=0.0, distribution="normal"),
            FeatureInfo(name="num_2", dtype="float64", unique_values=100, missing_percentage=0.0, distribution="uniform"),
        ],
        recommended_metrics=["accuracy"] if problem_type == "classification" else ["r2"],
    )


def _make_result(run_id: str, iteration: int, accuracy: float = 0.80, r2: float | None = None, failed: bool = False) -> ExperimentResult:
    res = ExperimentResult(
        experiment_id=run_id,
        iteration=iteration,
        dataset_name="int_test_data",
        model_name="RandomForestClassifier" if r2 is None else "RandomForestRegressor",
        model_type="ml",
        hyperparameters={"n_estimators": 25},
        preprocessing_applied=PreprocessingConfig(
            missing_value_strategy="mean",
            scaling="standard",
            encoding="onehot",
            imbalance_strategy="none",
            feature_selection="none",
        ),
        metrics=ExperimentMetrics(
            accuracy=accuracy if r2 is None else None,
            r2=r2,
            precision=accuracy if r2 is None else None,
            recall=accuracy if r2 is None else None,
            f1_score=accuracy if r2 is None else None,
            train_loss=0.20,
            validation_loss=0.25,
        ),
        train_curve=[0.5, 0.2],
        validation_curve=[0.55, 0.25],
        runtime=1.0,
        resource_usage=ResourceUsage(cpu_time_sec=1.0, gpu_used=False),
        artifacts=ArtifactRefs(
            model_path="model.pkl",
            metrics_file="metrics.json",
            tensorboard_logs="",
        ),
        failed=failed,
        status="failed" if failed else "completed",
    )
    return res


class TestP0_29_EndToEndExperimentalIntegrity:
    """
    Focused end-to-end integration tests covering 8 canonical experimental conditions.
    """

    # -----------------------------------------------------------------------
    # 1. Successful Experiment Loop
    # -----------------------------------------------------------------------
    def test_condition_1_successful_experiment(self, tmp_path, monkeypatch):
        run_id = f"int_succ_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)
        csv_path = _make_sample_dataset(tmp_path, n_rows=60)

        engine = DecisionEngine(run_id=run_id, max_iterations=2)
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            max_iterations=2,
            split_config=SplitConfig(method="stratified", random_seed=42),
        )

        orchestrator.run(str(csv_path), "target")

        # Verify artifacts and transitions are coherent
        out_dir = tmp_path / "outputs" / run_id
        assert out_dir.exists()
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "artifacts" / "budget_accounting.json").exists()
        assert (out_dir / "artifacts" / "model.pkl").exists()
        assert (out_dir / "artifacts" / "test_metrics.json").exists()

        # Check decision log records
        log_files = list((out_dir / "decision_logs").glob(f"{run_id}_*.json"))
        assert len(log_files) >= 2, "Expected at least 2 decision cycles logged"

    # -----------------------------------------------------------------------
    # 2. Failed Experiment
    # -----------------------------------------------------------------------
    def test_condition_2_failed_experiment(self, tmp_path, monkeypatch):
        run_id = f"int_fail_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id, max_iterations=3)
        profile = _make_profile()
        d0 = engine.receive_profile(profile)

        # Iteration 1 fails explicitly
        res_fail = _make_result(run_id=run_id, iteration=1, accuracy=0.0, failed=True)
        d1 = engine.receive_result(res_fail)

        # State transition to S_2 reflects previous_action_success = False
        assert engine._last_action_success is False
        assert d1 is not None

        # Evaluator and decision records reflect the failure cleanly
        rec0_path = tmp_path / "outputs" / run_id / "decision_logs" / f"{run_id}_0000.json"
        rec0 = json.loads(rec0_path.read_text(encoding="utf-8"))
        assert rec0["execution_result"]["action_success"] is False

    # -----------------------------------------------------------------------
    # 3. Improvement
    # -----------------------------------------------------------------------
    def test_condition_3_improvement(self, tmp_path, monkeypatch):
        run_id = f"int_impr_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id, max_iterations=4)
        profile = _make_profile()
        engine.receive_profile(profile)

        # Baseline: 0.70
        res1 = _make_result(run_id=run_id, iteration=1, accuracy=0.70)
        engine.receive_result(res1)

        # Improvement: 0.85
        res2 = _make_result(run_id=run_id, iteration=2, accuracy=0.85)
        import stratml.decision.engine as engine_mod
        with patch("stratml.decision.engine.build_state", wraps=engine_mod.build_state) as spy_build:
            engine.receive_result(res2)
            kwargs = spy_build.call_args.kwargs
            assert kwargs.get("previous_action_success") is True
            assert engine._best_val_score == 0.85

    # -----------------------------------------------------------------------
    # 4. Degradation
    # -----------------------------------------------------------------------
    def test_condition_4_degradation(self, tmp_path, monkeypatch):
        run_id = f"int_deg_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id, max_iterations=4)
        profile = _make_profile()
        engine.receive_profile(profile)

        # Baseline: 0.80
        res1 = _make_result(run_id=run_id, iteration=1, accuracy=0.80)
        engine.receive_result(res1)

        # Degradation: 0.65
        res2 = _make_result(run_id=run_id, iteration=2, accuracy=0.65)
        import stratml.decision.engine as engine_mod
        with patch("stratml.decision.engine.build_state", wraps=engine_mod.build_state) as spy_build:
            engine.receive_result(res2)
            kwargs = spy_build.call_args.kwargs
            assert kwargs.get("previous_action_success") is False
            # Best score remains 0.80, not degraded to 0.65
            assert engine._best_val_score == 0.80

    # -----------------------------------------------------------------------
    # 5. Termination
    # -----------------------------------------------------------------------
    def test_condition_5_termination(self, tmp_path, monkeypatch):
        run_id = f"int_term_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id, max_iterations=2)
        profile = _make_profile()
        engine.receive_profile(profile)

        res1 = _make_result(run_id=run_id, iteration=1, accuracy=0.82)
        engine.receive_result(res1)

        # At iteration 2 (max_iterations=2), remaining budget is 0 -> termination
        res2 = _make_result(run_id=run_id, iteration=2, accuracy=0.84)
        d_term = engine.receive_result(res2)

        assert d_term.action_type == "terminate"

        # Verify terminated row is backfilled in decision dataset
        df = pd.read_csv(tmp_path / "outputs" / run_id / "decision_logs" / "decision_dataset.csv")
        last_row = df.iloc[-1]
        assert last_row["action_type"] == "terminate"
        assert str(last_row["observed_gain"]).strip() != ""

    # -----------------------------------------------------------------------
    # 6. Budget Exhaustion
    # -----------------------------------------------------------------------
    def test_condition_6_budget_exhaustion(self, tmp_path, monkeypatch):
        run_id = f"int_bdg_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)
        csv_path = _make_sample_dataset(tmp_path, n_rows=40)

        engine = DecisionEngine(run_id=run_id, max_iterations=1, time_budget=0.001)
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            max_iterations=1,
            time_budget=0.001,
        )

        orchestrator.run(str(csv_path), "target")

        budget_file = tmp_path / "outputs" / run_id / "artifacts" / "budget_accounting.json"
        assert budget_file.exists()
        bdg = json.loads(budget_file.read_text(encoding="utf-8"))
        assert bdg["actual_consumption"]["timeout_triggered"] is True or bdg["actual_consumption"]["decision_iterations"] >= 1

    # -----------------------------------------------------------------------
    # 7. Warm-Start Experience Available
    # -----------------------------------------------------------------------
    def test_condition_7_warm_start_experience_available(self, tmp_path, monkeypatch):
        shared_memory = tmp_path / "shared_meta_memory.jsonl"
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", shared_memory)

        # Seed MetaMemory with prior experience on the same dataset
        meta_memory.record_run(
            meta_features={"num_samples": 100, "num_features": 2, "missing_value_ratio": 0.0},
            best_model="GradientBoostingClassifier",
            best_score=0.92,
            run_id="seed_run",
            dataset_id="int_test_data",
            dataset_fingerprint="fp_int_123",
        )

        run_id = f"int_warm_{uuid.uuid4().hex[:6]}"
        engine = DecisionEngine(
            run_id=run_id,
            history_mode="continual",
            allowed_models=["RandomForestClassifier", "GradientBoostingClassifier"],
        )
        profile = _make_profile(fingerprint="fp_int_123")
        decision = engine.receive_profile(profile)

        # In continual mode with matching dataset, GradientBoostingClassifier should be prioritized
        assert engine.allowed_models[0] == "GradientBoostingClassifier"

    # -----------------------------------------------------------------------
    # 8. Insufficient Warm-Start Experience
    # -----------------------------------------------------------------------
    def test_condition_8_insufficient_warm_start_experience(self, tmp_path, monkeypatch):
        empty_memory = tmp_path / "empty_meta_memory.jsonl"
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", empty_memory)

        run_id = f"int_insuf_{uuid.uuid4().hex[:6]}"
        engine = DecisionEngine(
            run_id=run_id,
            history_mode="independent",
            allowed_models=["RandomForestClassifier", "LogisticRegression"],
        )
        profile = _make_profile()
        decision = engine.receive_profile(profile)

        # Falls back to default bootstrap heuristic cleanly without crashing
        assert decision.action_type in ("switch_model", "apply_preprocessing")
        assert decision.reason.source in ("bootstrap", "rule")
        assert engine.allowed_models[0] in ("RandomForestClassifier", "LogisticRegression")
