"""
test_p1_21_30.py
----------------
Focused verification tests for the six P1 hardening items (Requirements 21 through 30):
  P1-21: Make ablations experiment-level configuration (exposed in CLI/YAML and recorded in manifest.json)
  P1-22: Record coordinator learning activation/state (enabled, active, observation count, threshold)
  P1-24: Truthful decision-source attribution distinguishing mechanism from selection mode, plus fallback participation
  P1-25: Canonical sequential trajectory artifact (outputs/<run_id>/decision_logs/trajectory.jsonl)
  P1-27: Separate execution success (completed|failed) from action outcome (improvement|degradation|failure|neutral)
  P1-28 + P1-30: Exact warm-start provenance, corpus snapshot hash, resolved config, and config hash in manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

from stratml.core.schemas import (
    ActionDecision,
    AgentScore,
    CandidateAction,
    DecisionReason,
    DecisionRecord,
    PreprocessingConfig,
    SecondaryMetrics,
    StateActionContext,
    StateConstraints,
    StateDataset,
    StateGeneralization,
    StateMeta,
    StateMetrics,
    StateModel,
    StateObject,
    StateObjective,
    StateResources,
    StateSearch,
    StateSignals,
    StateTrajectory,
    StateUncertainty,
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
from stratml.decision.agents import coordinator_agent
from stratml.decision.agents.coordinator_agent import RankedAction
from stratml.decision.logging import decision_logger
from stratml.decision.learning.uncertainty import UncertaintyEstimate
from stratml.decision.policy.action_selector import select
from stratml.cli import config as cli_config
from stratml.orchestration.orchestrator import ExecutionOrchestrator


@pytest.fixture(autouse=True)
def disable_external_apis(monkeypatch):
    """Ensure tests run offline without hitting external APIs."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def _make_dummy_profile(name="p1_test_data", problem_type="classification", fingerprint="fp_xyz789"):
    return DataProfile(
        dataset_name=name,
        dataset_type="tabular",
        rows=100,
        columns=3,
        target_column="target",
        problem_type=problem_type,
        numerical_columns=["f1", "f2"],
        categorical_columns=[],
        missing_value_ratio=0.0,
        dataset_fingerprint=fingerprint,
        feature_summary=[
            FeatureInfo(name="f1", dtype="float64", unique_values=100, missing_percentage=0.0, distribution="normal"),
            FeatureInfo(name="f2", dtype="float64", unique_values=100, missing_percentage=0.0, distribution="uniform"),
        ],
        recommended_metrics=["accuracy"] if problem_type == "classification" else ["r2"],
    )


def _make_dummy_state(experiment_id="test_exp", iteration=1):
    return StateObject(
        meta=StateMeta(experiment_id=experiment_id, iteration=iteration, timestamp="2026-01-01T00:00:00Z"),
        objective=StateObjective(primary_metric="accuracy", optimization_goal="maximize"),
        metrics=StateMetrics(primary=0.80, secondary=SecondaryMetrics(accuracy=0.80), train_val_gap=0.02),
        trajectory=StateTrajectory(
            history_length=iteration,
            improvement_rate=0.02,
            slope=0.01,
            volatility=0.01,
            best_score=0.80,
            mean_score=0.78,
            steps_since_improvement=0,
            trend="improving",
        ),
        dataset=StateDataset(num_samples=100, num_features=2, feature_to_sample_ratio=0.02, missing_ratio=0.0),
        model=StateModel(
            model_name="RandomForestClassifier",
            model_type="ml",
            hyperparameters={"n_estimators": 50},
            runtime=1.0,
            convergence_epoch=10,
        ),
        generalization=StateGeneralization(train_loss=0.1, validation_loss=0.12, gap=0.02),
        resources=StateResources(runtime=1.0, gpu_used=False, cpu_time=1.0, remaining_budget=4.0, budget_exhausted=False),
        search=StateSearch(models_tried=["RandomForestClassifier"], unique_models_count=1, repeated_configs=0),
        signals=StateSignals(),
        uncertainty=StateUncertainty(),
        action_context=StateActionContext(),
        constraints=StateConstraints(allowed_models=["RandomForestClassifier", "LogisticRegression"], max_iterations=5),
    )


def _make_dummy_result(experiment_id="test_exp", iteration=1, accuracy=0.85, failed=False):
    return ExperimentResult(
        experiment_id=experiment_id,
        iteration=iteration,
        dataset_name="p1_test_data",
        model_name="RandomForestClassifier",
        model_type="ml",
        hyperparameters={"n_estimators": 50},
        preprocessing_applied=PreprocessingConfig(
            missing_value_strategy="mean", scaling="standard", encoding="onehot", imbalance_strategy="none", feature_selection="none"
        ),
        metrics=ExperimentMetrics(accuracy=accuracy, train_loss=0.1, validation_loss=0.15),
        train_curve=[0.2, 0.1],
        validation_curve=[0.25, 0.15],
        runtime=1.2,
        resource_usage=ResourceUsage(cpu_time_sec=1.2, memory_mb=128.0),
        artifacts=ArtifactRefs(
            model_path="outputs/test/artifacts/model.pkl",
            metrics_file="outputs/test/artifacts/metrics.json",
            tensorboard_logs="",
        ),
        failed=failed,
        status="failed" if failed else "completed",
    )


# ---------------------------------------------------------------------------
# P1-21: Make ablations experiment-level configuration
# ---------------------------------------------------------------------------

class TestP1_21_AblationConfiguration:
    def test_cli_overrides_disable_meta_memory(self):
        args = argparse.Namespace(
            mode=None, max_iter=None, path=None, dl=False, architecture=None,
            epochs=None, lr=None, batch_size=None, tune=False,
            disable_meta_memory=True, disable_value_model=False, ablation=None,
        )
        cfg = cli_config.apply_cli_overrides(cli_config.DEFAULT_CONFIG, args)
        assert cfg["ablations"]["enable_meta_memory"] is False
        assert cfg["ablations"]["enable_value_model"] is True
        assert cfg["ablations"]["condition"] == "MetaMemory OFF"

    def test_cli_overrides_disable_value_model(self):
        args = argparse.Namespace(
            mode=None, max_iter=None, path=None, dl=False, architecture=None,
            epochs=None, lr=None, batch_size=None, tune=False,
            disable_meta_memory=False, disable_value_model=True, ablation=None,
        )
        cfg = cli_config.apply_cli_overrides(cli_config.DEFAULT_CONFIG, args)
        assert cfg["ablations"]["enable_meta_memory"] is True
        assert cfg["ablations"]["enable_value_model"] is False
        assert cfg["ablations"]["condition"] == "Value Model OFF"

    def test_cli_overrides_ablation_string(self):
        args = argparse.Namespace(
            mode=None, max_iter=None, path=None, dl=False, architecture=None,
            epochs=None, lr=None, batch_size=None, tune=False,
            disable_meta_memory=False, disable_value_model=False, ablation="MetaMemory OFF",
        )
        cfg = cli_config.apply_cli_overrides(cli_config.DEFAULT_CONFIG, args)
        assert cfg["ablations"]["enable_meta_memory"] is False
        assert cfg["ablations"]["condition"] == "MetaMemory OFF"

    def test_manifest_records_condition(self, tmp_path):
        run_id = f"test_p1_21_{uuid.uuid4().hex[:6]}"
        engine = DecisionEngine(
            run_id=run_id,
            enable_meta_memory=False,
            enable_value_model=True,
            max_iterations=1,
        )
        profile = _make_dummy_profile()

        # Orchestrator run
        cfg = {
            "mode": "beginner",
            "ablations": {
                "enable_meta_memory": False,
                "enable_value_model": True,
                "condition": "MetaMemory OFF",
            }
        }
        orch = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            max_iterations=1,
            resolved_config=cfg,
        )

        dummy_csv = tmp_path / "test.csv"
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0] * 5, "f2": [0.5, 1.5, 2.5, 3.5] * 5, "target": [0, 1, 0, 1] * 5})
        df.to_csv(dummy_csv, index=False)

        orch.run(str(dummy_csv), "target")

        manifest_path = Path("outputs") / run_id / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["condition"] == "MetaMemory OFF"
        assert manifest["warm_start_mode_corpus"]["condition"] == "MetaMemory OFF"
        assert manifest["warm_start_mode_corpus"]["enable_meta_memory"] is False
        assert manifest["warm_start_mode_corpus"]["enable_value_model"] is True
        assert manifest["resolved_experiment_config"]["ablations"]["condition"] == "MetaMemory OFF"


# ---------------------------------------------------------------------------
# P1-22: Coordinator learning activation/state
# ---------------------------------------------------------------------------

class TestP1_22_CoordinatorLearningState:
    def test_learning_state_structure(self):
        state = coordinator_agent.get_learning_state()
        assert "weight_learning_enabled" in state
        assert "weight_learning_active" in state
        assert "weight_learning_observation_count" in state
        assert "weight_learning_threshold" in state
        assert state["weight_learning_threshold"] == 5

    def test_inactive_under_threshold(self, tmp_path):
        coordinator_agent.reset_weight_update_history()
        log_file = tmp_path / "eval.jsonl"
        # Write 3 records (< 5)
        with open(log_file, "w") as f:
            for i in range(3):
                f.write(json.dumps({"evaluation_id": f"e{i}", "counterfactual_impact": 0.05, "decision_validity": 0.8, "quality_risk": 0.1}) + "\n")

        weights = coordinator_agent._load_agent_weights(log_paths=[log_file])
        l_state = coordinator_agent.get_learning_state()
        assert l_state["weight_learning_active"] is False
        assert l_state["weight_learning_observation_count"] == 3
        # Weights remain defaults
        assert weights == (0.50, 0.25, 0.25)

    def test_active_at_or_above_threshold(self, tmp_path):
        coordinator_agent.reset_weight_update_history()
        log_file = tmp_path / "eval.jsonl"
        # Write 5 records (>= 5)
        with open(log_file, "w") as f:
            for i in range(5):
                f.write(json.dumps({"evaluation_id": f"e{i}", "counterfactual_impact": 0.05, "decision_validity": 0.8, "quality_risk": 0.1}) + "\n")

        weights = coordinator_agent._load_agent_weights(log_paths=[log_file])
        l_state = coordinator_agent.get_learning_state()
        assert l_state["weight_learning_active"] is True
        assert l_state["weight_learning_observation_count"] == 5

    def test_decision_record_persists_learning_state(self, tmp_path):
        run_id = f"test_p1_22_{uuid.uuid4().hex[:6]}"
        engine = DecisionEngine(run_id=run_id, max_iterations=2)
        profile = _make_dummy_profile()
        decision = engine.receive_profile(profile)

        rec_file = Path("outputs") / run_id / "decision_logs" / f"{decision.experiment_id}_{decision.iteration:04d}.json"
        assert rec_file.exists()
        rec_data = json.loads(rec_file.read_text(encoding="utf-8"))
        assert "coordinator_learning_state" in rec_data
        assert rec_data["coordinator_learning_state"]["weight_learning_threshold"] == 5


# ---------------------------------------------------------------------------
# P1-24: Truthful decision-source attribution and fallback participation
# ---------------------------------------------------------------------------

class TestP1_24_SourceAttributionAndFallbackParticipation:
    def test_greedy_selection_mode_and_source_preserved(self):
        state = _make_dummy_state(iteration=1)
        ranked = [
            RankedAction(
                action_type="tune_hyperparameters",
                parameters={"n_estimators": 100},
                predicted_gain=0.1,
                predicted_cost=0.2,
                confidence=0.9,
                agent_scores=AgentScore(performance=0.8, efficiency=0.7, stability=0.9),
                final_score=0.8,
                source="hybrid",
            )
        ]
        decision = select(state, ranked, decision_source="hybrid")
        assert decision.reason.source == "hybrid"
        assert decision.reason.selection_mode == "greedy"
        assert decision.reason.fallback_participation is False

    def test_epsilon_exploration_mode_and_source_preserved(self):
        state = _make_dummy_state(iteration=1)
        ranked = [
            RankedAction(
                action_type="tune_hyperparameters",
                parameters={"n_estimators": 100},
                predicted_gain=0.1,
                predicted_cost=0.2,
                confidence=0.9,
                agent_scores=AgentScore(performance=0.8, efficiency=0.7, stability=0.9),
                final_score=0.8,
                source="rule",
            ),
            RankedAction(
                action_type="switch_model",
                parameters={"model_name": "LogisticRegression"},
                predicted_gain=0.05,
                predicted_cost=0.1,
                confidence=0.8,
                agent_scores=AgentScore(performance=0.6, efficiency=0.9, stability=0.8),
                final_score=0.7,
                source="rule",
            )
        ]
        # Force exploration by mocking rng.random to return 0.0 (less than epsilon)
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.0
        mock_rng.choice.return_value = ranked[1]

        decision = select(state, ranked, rng=mock_rng, decision_source="rule")
        assert decision.reason.source == "rule"
        assert decision.reason.selection_mode == "epsilon_exploration"
        assert decision.action_type == "switch_model"

    def test_bootstrap_iteration_mode_and_source(self):
        state = _make_dummy_state(iteration=0)
        ranked = [
            RankedAction(
                action_type="switch_model",
                parameters={"model_name": "RandomForestClassifier"},
                predicted_gain=0.1,
                predicted_cost=0.2,
                confidence=0.9,
                agent_scores=AgentScore(performance=0.8, efficiency=0.7, stability=0.9),
                final_score=0.8,
                source="rule",
            )
        ]
        decision = select(state, ranked)
        assert decision.reason.source == "bootstrap"
        assert decision.reason.selection_mode == "bootstrap"

    def test_partial_llm_ranking_records_fallback_participation(self):
        estimates = [
            UncertaintyEstimate(action_type="action_a", parameters={}, predicted_gain=0.1, predicted_cost=0.2, confidence=0.8, variance=0.0),
            UncertaintyEstimate(action_type="action_b", parameters={}, predicted_gain=0.05, predicted_cost=0.1, confidence=0.7, variance=0.0),
        ]
        perf = {"action_a": 0.8, "action_b": 0.5}
        eff = {"action_a": 0.7, "action_b": 0.8}
        stab = {"action_a": 0.9, "action_b": 0.6}
        state = _make_dummy_state(iteration=1)

        # Mock LLM output that only returns action_a, omitting action_b
        class MockItem:
            def __init__(self, at, score, rationale):
                self.action_type = at
                self.final_score = score
                self.rationale = rationale

        class MockOutput:
            ranked = [MockItem("action_a", 0.9, "Good action")]

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = MockOutput()

        with patch("langchain_groq.ChatGroq", return_value=mock_llm):
            ranked = coordinator_agent._llm_rank(state, estimates, perf, eff, stab)

        assert ranked is not None
        assert len(ranked) == 2
        # action_a came from LLM
        item_a = next(r for r in ranked if r.action_type == "action_a")
        assert item_a.source == "llm"
        assert item_a.fallback_participation is False

        # action_b was omitted by LLM and scored via fallback
        item_b = next(r for r in ranked if r.action_type == "action_b")
        assert item_b.source == "fallback"
        assert item_b.fallback_participation is True


# ---------------------------------------------------------------------------
# P1-25: Canonical sequential trajectory artifact (trajectory.jsonl)
# ---------------------------------------------------------------------------

class TestP1_25_SequentialTrajectoryArtifact:
    def test_trajectory_jsonl_created_and_sequential(self, tmp_path):
        decision_logger._LOG_DIR = tmp_path / "decision_logs"
        state0 = _make_dummy_state(experiment_id="exp_traj", iteration=0)
        state1 = _make_dummy_state(experiment_id="exp_traj", iteration=1)

        decision0 = ActionDecision(
            experiment_id="exp_traj", iteration=0, action_type="switch_model", parameters={"model_name": "RF"},
            preprocessing=PreprocessingConfig(), expected_gain=0.1, confidence=0.8, reason="init"
        )
        decision1 = ActionDecision(
            experiment_id="exp_traj", iteration=1, action_type="tune_hyperparameters", parameters={"max_depth": 5},
            preprocessing=PreprocessingConfig(), expected_gain=0.05, confidence=0.85, reason="underfitting"
        )

        decision_logger.log(state0, [CandidateAction(action_type="switch_model", parameters={})], decision0)
        decision_logger.log(state1, [CandidateAction(action_type="tune_hyperparameters", parameters={})], decision1)

        traj_file = tmp_path / "decision_logs" / "trajectory.jsonl"
        assert traj_file.exists()

        lines = [json.loads(line) for line in traj_file.read_text(encoding="utf-8").strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["iteration"] == 0
        assert lines[0]["state_id"] == "exp_traj_0"
        assert lines[0]["executed_configuration"] == {"model_name": "RF"}

        assert lines[1]["iteration"] == 1
        assert lines[1]["state_id"] == "exp_traj_1"
        assert lines[1]["executed_configuration"] == {"max_depth": 5}

    def test_update_outcome_syncs_trajectory_jsonl(self, tmp_path):
        decision_logger._LOG_DIR = tmp_path / "decision_logs"
        state = _make_dummy_state(experiment_id="exp_traj2", iteration=0)
        decision = ActionDecision(
            experiment_id="exp_traj2", iteration=0, action_type="switch_model", parameters={"model_name": "RF"},
            preprocessing=PreprocessingConfig(), expected_gain=0.1, confidence=0.8, reason="init"
        )
        decision_logger.log(state, [CandidateAction(action_type="switch_model", parameters={})], decision)

        # Update outcome
        exec_res = {"accuracy": 0.85, "execution_status": "completed", "action_outcome": "improvement"}
        eval_res = {"decision_validity": 0.9, "action_verdict": "correct"}
        decision_logger.update_outcome("exp_traj2", 0, execution_result=exec_res, evaluator_result=eval_res, next_state_id="exp_traj2_1")

        traj_file = tmp_path / "decision_logs" / "trajectory.jsonl"
        lines = [json.loads(line) for line in traj_file.read_text(encoding="utf-8").strip().split("\n")]
        assert len(lines) == 1
        assert lines[0]["execution_outcome"] == exec_res
        assert lines[0]["evaluator_result"] == eval_res
        assert lines[0]["next_state_id"] == "exp_traj2_1"


# ---------------------------------------------------------------------------
# P1-27: Separate execution success from action outcome
# ---------------------------------------------------------------------------

class TestP1_27_ExecutionStatusVsActionOutcome:
    def test_improvement_outcome(self):
        engine = DecisionEngine(run_id=f"test_p1_27_imp_{uuid.uuid4().hex[:6]}", max_iterations=3)
        engine.receive_profile(_make_dummy_profile())

        # Iteration 1: initial score 0.80
        res1 = _make_dummy_result(iteration=1, accuracy=0.80)
        engine.receive_result(res1)

        # Iteration 2: improved score 0.88
        res2 = _make_dummy_result(iteration=2, accuracy=0.88)
        engine.receive_result(res2)

        assert engine._last_execution_status == "completed"
        assert engine._last_action_outcome == "improvement"
        assert engine._last_action_success is True

    def test_degradation_outcome_with_completed_execution(self):
        engine = DecisionEngine(run_id=f"test_p1_27_deg_{uuid.uuid4().hex[:6]}", max_iterations=3)
        engine.receive_profile(_make_dummy_profile())

        # Iteration 1: score 0.85
        res1 = _make_dummy_result(iteration=1, accuracy=0.85)
        engine.receive_result(res1)

        # Iteration 2: degraded score 0.75 (execution completed successfully!)
        res2 = _make_dummy_result(iteration=2, accuracy=0.75)
        engine.receive_result(res2)

        assert engine._last_execution_status == "completed"
        assert engine._last_action_outcome == "degradation"
        assert engine._last_action_success is False

    def test_failure_outcome_with_failed_execution(self):
        engine = DecisionEngine(run_id=f"test_p1_27_fail_{uuid.uuid4().hex[:6]}", max_iterations=3)
        engine.receive_profile(_make_dummy_profile())

        # Iteration 1: score 0.85
        res1 = _make_dummy_result(iteration=1, accuracy=0.85)
        engine.receive_result(res1)

        # Iteration 2: failed execution (crash / timeout / memory error)
        res2 = _make_dummy_result(iteration=2, accuracy=0.0, failed=True)
        engine.receive_result(res2)

        assert engine._last_execution_status == "failed"
        assert engine._last_action_outcome == "failure"
        assert engine._last_action_success is False


# ---------------------------------------------------------------------------
# P1-28 + P1-30: Exact warm-start provenance, corpus snapshot hash & manifest
# ---------------------------------------------------------------------------

class TestP1_28_30_WarmStartProvenanceAndManifest:
    def test_independent_mode_corpus_path_and_hash(self, tmp_path):
        run_id = f"test_p1_28_{uuid.uuid4().hex[:6]}"
        engine = DecisionEngine(run_id=run_id, history_mode="independent", max_iterations=1)
        profile = _make_dummy_profile(fingerprint="test_fp_456")

        dummy_csv = tmp_path / "data.csv"
        df = pd.DataFrame({"f1": [1.0, 2.0] * 10, "f2": [0.1, 0.2] * 10, "target": [0, 1] * 10})
        df.to_csv(dummy_csv, index=False)

        orch = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            max_iterations=1,
            tune=False,
        )
        orch.run(str(dummy_csv), "target")

        manifest_path = Path("outputs") / run_id / "manifest.json"
        assert manifest_path.exists()
        m = json.loads(manifest_path.read_text(encoding="utf-8"))

        warm = m["warm_start_mode_corpus"]
        assert warm["history_mode"] == "independent"
        assert warm["corpus_path"] == str(Path("outputs") / run_id / "decision_logs" / "decision_dataset.csv")
        assert warm["dataset_fingerprint"] == m["dataset_fingerprint"]
        assert warm["dataset_fingerprint"] is not None

        # Check config hash
        assert "config_hash" in m
        assert len(m["config_hash"]) == 64  # sha256 hex string
        assert m["resolved_experiment_config"]["budget"]["tune"] is False
        assert m["resolved_experiment_config"]["seed"] == 42
