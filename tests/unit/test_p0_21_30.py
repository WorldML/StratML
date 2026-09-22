"""
test_p0_21_30.py
----------------
Focused unit & regression tests for P0 requirements 21 through 28, and 30:
  P0-21: Independently disable MetaMemory and Value Model (ablation controls)
  P0-22: Deterministic, bounded, and attributable coordinator weight learning
  P0-23: Persist coordinator weights per iteration in decision records
  P0-24: Correct decision-source classification
  P0-25: Complete state -> action -> outcome trajectory reconstructability
  P0-26: Preserve candidate / runner-up information in decision records
  P0-27: Enforce state-transition integrity (_previous_action_success updated)
  P0-28: Strengthen warm-start provenance/isolation in backfill_last_gain
  P0-30: Canonical machine-readable experiment manifest (outputs/<run_id>/manifest.json)
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np

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
from stratml.decision.agents import coordinator_agent, evaluator_agent
from stratml.decision.logging import decision_logger
from stratml.decision.learning import dataset_builder, meta_memory, value_model
from stratml.decision.learning.uncertainty import UncertaintyEstimate
from stratml.decision.policy.action_selector import select
from stratml.orchestration.orchestrator import ExecutionOrchestrator


@pytest.fixture(autouse=True)
def disable_external_apis(monkeypatch):
    """Ensure tests run offline without hitting external APIs."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def _make_dummy_profile(name="p0_test_data", problem_type="classification", fingerprint="fp_abc123"):
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


def _make_dummy_result(
    experiment_id="test_exp",
    iteration=1,
    accuracy=0.85,
    r2=None,
    model_name="RandomForestClassifier",
    failed=False,
):
    metrics = ExperimentMetrics(
        accuracy=accuracy if r2 is None else None,
        r2=r2,
        precision=accuracy if r2 is None else None,
        recall=accuracy if r2 is None else None,
        f1_score=accuracy if r2 is None else None,
        train_loss=0.15,
        validation_loss=0.20,
    )
    result = ExperimentResult(
        experiment_id=experiment_id,
        iteration=iteration,
        dataset_name="p0_test_data",
        model_name=model_name,
        model_type="ml",
        hyperparameters={"n_estimators": 50},
        preprocessing_applied=PreprocessingConfig(
            missing_value_strategy="mean",
            scaling="standard",
            encoding="onehot",
            imbalance_strategy="none",
            feature_selection="none",
        ),
        metrics=metrics,
        train_curve=[0.5, 0.3, 0.15],
        validation_curve=[0.55, 0.35, 0.20],
        runtime=1.5,
        resource_usage=ResourceUsage(cpu_time_sec=1.5, gpu_used=False),
        artifacts=ArtifactRefs(
            model_path="outputs/test/artifacts/model.pkl",
            metrics_file="outputs/test/artifacts/metrics.json",
            tensorboard_logs="",
        ),
        failed=failed,
        status="failed" if failed else "completed",
    )
    return result


def _make_dummy_state(
    iteration: int = 1,
    primary: float = 0.72,
    underfitting: str = "none",
    well_fitted: str = "strong",
    remaining_budget: float = 15.0,
) -> StateObject:
    return StateObject(
        meta=StateMeta(experiment_id="test_exp", iteration=iteration, timestamp="2026-01-01T00:00:00+00:00"),
        objective=StateObjective(primary_metric="accuracy", optimization_goal="maximize"),
        metrics=StateMetrics(primary=primary, secondary=SecondaryMetrics(), train_val_gap=0.02),
        trajectory=StateTrajectory(
            history_length=3, improvement_rate=0.01, slope=0.005,
            volatility=0.02, best_score=primary, mean_score=primary - 0.02,
            steps_since_improvement=1, trend="improving",
        ),
        dataset=StateDataset(num_samples=500, num_features=10, feature_to_sample_ratio=0.02, missing_ratio=0.0),
        model=StateModel(
            model_name="RandomForest", model_type="ml", hyperparameters={"n_estimators": 100},
            runtime=10.0, convergence_epoch=0,
        ),
        generalization=StateGeneralization(train_loss=0.2, validation_loss=0.22, gap=0.02),
        resources=StateResources(runtime=10.0, gpu_used=False, cpu_time=10.0, remaining_budget=remaining_budget, budget_exhausted=remaining_budget <= 0),
        search=StateSearch(models_tried=["RandomForest"], unique_models_count=1, repeated_configs=0),
        signals=StateSignals(
            underfitting=underfitting, overfitting="none", well_fitted=well_fitted,
            converged="none", stagnating="none", diverging="none",
            plateau_detected="none", diminishing_returns="none",
            unstable_training="none", high_variance="none",
        ),
        uncertainty=StateUncertainty(),
        action_context=StateActionContext(),
        constraints=StateConstraints(
            allowed_models=["LogisticRegression", "RandomForest", "GradientBoosting", "SVC"],
            max_iterations=20,
        ),
    )

_make_state = _make_dummy_state


# ===========================================================================
# P0-21: Independently disable MetaMemory and Value Model
# ===========================================================================
class TestP0_21_IndependentAblationControls:
    def test_metamemory_disabled_bypasses_retrieval(self, tmp_path, monkeypatch):
        run_id = f"test_ablation_mm_{uuid.uuid4().hex[:6]}"
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta_memory.jsonl")

        with patch.object(meta_memory, "retrieve_similar_actions") as mock_retrieve:
            engine = DecisionEngine(
                run_id=run_id,
                enable_meta_memory=False,
                enable_value_model=True,
            )
            profile = _make_dummy_profile()
            decision = engine.receive_profile(profile)

            # MetaMemory retrieval must NOT be invoked when enable_meta_memory=False
            mock_retrieve.assert_not_called()
            assert decision is not None
            assert decision.action_type in ("switch_model", "apply_preprocessing")

    def test_value_model_disabled_uses_neutral_predictions(self, tmp_path, monkeypatch):
        run_id = f"test_ablation_vm_{uuid.uuid4().hex[:6]}"
        with patch("stratml.decision.learning.value_model.predict") as mock_predict:
            engine = DecisionEngine(
                run_id=run_id,
                enable_meta_memory=True,
                enable_value_model=False,
            )
            profile = _make_dummy_profile()
            decision = engine.receive_profile(profile)

            # value_model.predict must NOT be called when enable_value_model=False
            mock_predict.assert_not_called()
            assert decision is not None
            assert decision.action_type in ("switch_model", "apply_preprocessing")

    def test_full_ablation_both_disabled_runs_cleanly(self, tmp_path, monkeypatch):
        run_id = f"test_ablation_both_{uuid.uuid4().hex[:6]}"
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta_memory.jsonl")

        with patch.object(meta_memory, "retrieve_similar_actions") as mock_retrieve, \
             patch("stratml.decision.learning.value_model.predict") as mock_predict:
            engine = DecisionEngine(
                run_id=run_id,
                enable_meta_memory=False,
                enable_value_model=False,
            )
            profile = _make_dummy_profile()
            d0 = engine.receive_profile(profile)
            mock_retrieve.assert_not_called()
            mock_predict.assert_not_called()

            res1 = _make_dummy_result(experiment_id=run_id, iteration=1, accuracy=0.82)
            d1 = engine.receive_result(res1)
            mock_predict.assert_not_called()
            assert d1 is not None


# ===========================================================================
# P0-22: Deterministic, bounded, and attributable coordinator weight learning
# ===========================================================================
# ===========================================================================
# P0-22: Deterministic, bounded, attributable, and persisted coordinator weight learning
# ===========================================================================
class TestP0_22_AttributableCoordinatorWeightLearning:
    def test_identical_inputs_produce_identical_weights_and_history(self, tmp_path):
        eval_log = tmp_path / "eval_log.jsonl"
        records = [
            {"counterfactual_impact": 0.05, "decision_validity": 0.9, "quality_risk": 0.1, "iteration": 1, "experiment_id": "exp1"},
            {"counterfactual_impact": -0.01, "decision_validity": 0.8, "quality_risk": 0.2, "iteration": 2, "experiment_id": "exp1"},
            {"counterfactual_impact": 0.03, "decision_validity": 0.7, "quality_risk": 0.3, "iteration": 3, "experiment_id": "exp1"},
            {"counterfactual_impact": 0.00, "decision_validity": 0.6, "quality_risk": 0.1, "iteration": 4, "experiment_id": "exp1"},
            {"counterfactual_impact": -0.04, "decision_validity": 0.3, "quality_risk": 0.8, "iteration": 5, "experiment_id": "exp1"},
            {"counterfactual_impact": 0.02, "decision_validity": 0.85, "quality_risk": 0.15, "iteration": 6, "experiment_id": "exp1"},
        ]
        with open(eval_log, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        coordinator_agent.reset_weight_update_history()
        weights_1 = coordinator_agent._load_agent_weights(log_paths=[eval_log])
        history_1 = coordinator_agent.get_weight_update_history()

        coordinator_agent.reset_weight_update_history()
        weights_2 = coordinator_agent._load_agent_weights(log_paths=[eval_log])
        history_2 = coordinator_agent.get_weight_update_history()

        # Determinism
        assert weights_1 == weights_2
        assert history_1 == history_2
        assert len(history_1) == len(records)

    def test_weights_remain_bounded_and_normalized(self, tmp_path):
        eval_log = tmp_path / "extreme_eval_log.jsonl"
        # 10 completely failing records designed to collapse performance weight
        records = [
            {"counterfactual_impact": -0.50, "decision_validity": 0.0, "quality_risk": 1.0, "iteration": i, "experiment_id": "exp_fail"}
            for i in range(10)
        ]
        with open(eval_log, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        coordinator_agent.reset_weight_update_history()
        w_p, w_e, w_s = coordinator_agent._load_agent_weights(log_paths=[eval_log])

        # Boundedness: weights must not collapse to 0
        assert w_p >= 1e-4
        assert w_e >= 1e-4
        assert w_s >= 1e-4
        # Normalization
        assert round(w_p + w_e + w_s, 2) == 1.0

    def test_one_evaluation_produces_one_persisted_attributable_update(self, tmp_path):
        eval_log = tmp_path / "eval_log.jsonl"
        update_log = tmp_path / "coordinator_weight_updates.jsonl"
        records = [
            {"counterfactual_impact": 0.02, "decision_validity": 0.8, "quality_risk": 0.2, "iteration": i, "experiment_id": f"eval_{i}", "run_id": "run_test"}
            for i in range(6)
        ]
        with open(eval_log, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        coordinator_agent.reset_weight_update_history()
        coordinator_agent._load_agent_weights(log_paths=[eval_log], update_log_path=update_log, run_id="run_test")

        persisted = coordinator_agent.get_persisted_weight_updates(update_log)
        assert len(persisted) == 6

        # Idempotence: calling again does not create duplicate entries
        coordinator_agent._load_agent_weights(log_paths=[eval_log], update_log_path=update_log, run_id="run_test")
        persisted_again = coordinator_agent.get_persisted_weight_updates(update_log)
        assert len(persisted_again) == 6

    def test_previous_and_new_weights_are_correct(self, tmp_path):
        eval_log = tmp_path / "eval_log.jsonl"
        update_log = tmp_path / "coordinator_weight_updates.jsonl"
        records = [
            {"counterfactual_impact": 0.05, "decision_validity": 0.9, "quality_risk": 0.1, "iteration": 1, "experiment_id": "eval_1", "run_id": "run_test"},
            {"counterfactual_impact": -0.05, "decision_validity": 0.2, "quality_risk": 0.8, "iteration": 2, "experiment_id": "eval_2", "run_id": "run_test"},
            {"counterfactual_impact": 0.01, "decision_validity": 0.7, "quality_risk": 0.3, "iteration": 3, "experiment_id": "eval_3", "run_id": "run_test"},
            {"counterfactual_impact": 0.02, "decision_validity": 0.6, "quality_risk": 0.4, "iteration": 4, "experiment_id": "eval_4", "run_id": "run_test"},
            {"counterfactual_impact": 0.04, "decision_validity": 0.8, "quality_risk": 0.2, "iteration": 5, "experiment_id": "eval_5", "run_id": "run_test"},
        ]
        with open(eval_log, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        coordinator_agent.reset_weight_update_history()
        coordinator_agent._load_agent_weights(log_paths=[eval_log], update_log_path=update_log, run_id="run_test")
        persisted = coordinator_agent.get_persisted_weight_updates(update_log)

        for entry in persisted:
            assert "previous_weights" in entry
            assert "new_weights" in entry
            for k in ("performance", "efficiency", "stability"):
                assert k in entry["previous_weights"]
                assert k in entry["new_weights"]
                assert entry["new_weights"][k] >= 1e-4
            # Normalized sums
            prev_sum = sum(entry["previous_weights"][k] for k in ("performance", "efficiency", "stability"))
            new_sum = sum(entry["new_weights"][k] for k in ("performance", "efficiency", "stability"))
            assert abs(prev_sum - 1.0) < 1e-3
            assert abs(new_sum - 1.0) < 1e-3

    def test_evaluation_iteration_and_run_ids_are_correct(self, tmp_path):
        eval_log = tmp_path / "eval_log.jsonl"
        update_log = tmp_path / "coordinator_weight_updates.jsonl"
        records = [
            {"counterfactual_impact": 0.02, "decision_validity": 0.8, "quality_risk": 0.2, "iteration": i, "experiment_id": f"custom_eval_{i}", "run_id": f"custom_run_{i}"}
            for i in range(6)
        ]
        with open(eval_log, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        coordinator_agent.reset_weight_update_history()
        coordinator_agent._load_agent_weights(log_paths=[eval_log], update_log_path=update_log)
        persisted = coordinator_agent.get_persisted_weight_updates(update_log)

        for i, entry in enumerate(persisted):
            assert entry["evaluation_id"] == f"custom_eval_{i}"
            assert entry["iteration"] == i
            assert entry["run_id"] == f"custom_run_{i}"

    def test_persisted_history_survives_process_completion(self, tmp_path):
        eval_log = tmp_path / "eval_log.jsonl"
        update_log = tmp_path / "coordinator_weight_updates.jsonl"
        records = [
            {"counterfactual_impact": 0.02, "decision_validity": 0.8, "quality_risk": 0.2, "iteration": i, "experiment_id": f"eval_{i}", "run_id": "run_survival"}
            for i in range(6)
        ]
        with open(eval_log, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        coordinator_agent.reset_weight_update_history()
        coordinator_agent._load_agent_weights(log_paths=[eval_log], update_log_path=update_log, run_id="run_survival")

        # Simulate process exit: reset in-memory variables
        coordinator_agent.reset_weight_update_history()
        assert len(coordinator_agent._WEIGHT_UPDATE_HISTORY) == 0

        # Artifact on disk survives
        reloaded = coordinator_agent.get_weight_update_history(log_path=update_log)
        assert len(reloaded) == 6
        assert reloaded[0]["evaluation_id"] == "eval_0"

    def test_replaying_ordered_updates_reconstructs_observed_weight_trajectory(self, tmp_path):
        eval_log = tmp_path / "eval_log.jsonl"
        update_log = tmp_path / "coordinator_weight_updates.jsonl"
        records = [
            {"counterfactual_impact": 0.05, "decision_validity": 0.9, "quality_risk": 0.1, "iteration": 1, "experiment_id": "eval_1", "run_id": "run_traj"},
            {"counterfactual_impact": -0.05, "decision_validity": 0.2, "quality_risk": 0.8, "iteration": 2, "experiment_id": "eval_2", "run_id": "run_traj"},
            {"counterfactual_impact": 0.02, "decision_validity": 0.85, "quality_risk": 0.15, "iteration": 3, "experiment_id": "eval_3", "run_id": "run_traj"},
            {"counterfactual_impact": 0.00, "decision_validity": 0.6, "quality_risk": 0.3, "iteration": 4, "experiment_id": "eval_4", "run_id": "run_traj"},
            {"counterfactual_impact": 0.03, "decision_validity": 0.75, "quality_risk": 0.25, "iteration": 5, "experiment_id": "eval_5", "run_id": "run_traj"},
        ]
        with open(eval_log, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        coordinator_agent.reset_weight_update_history()
        final_weights = coordinator_agent._load_agent_weights(log_paths=[eval_log], update_log_path=update_log, run_id="run_traj")
        persisted = coordinator_agent.get_persisted_weight_updates(update_log)

        # Sequential trajectory continuity
        for i in range(1, len(persisted)):
            prev_entry = persisted[i - 1]
            curr_entry = persisted[i]
            for k in ("performance", "efficiency", "stability"):
                assert abs(curr_entry["previous_weights"][k] - prev_entry["new_weights"][k]) < 1e-4

        # Final weights match active coordinator weights
        last = persisted[-1]
        assert abs(last["new_weights"]["performance"] - final_weights[0]) < 1e-3
        assert abs(last["new_weights"]["efficiency"] - final_weights[1]) < 1e-3
        assert abs(last["new_weights"]["stability"] - final_weights[2]) < 1e-3


# ===========================================================================
# P0-23: Persist coordinator weights per iteration
# ===========================================================================
class TestP0_23_CoordinatorWeightsPersistence:
    def test_coordinator_weights_persisted_in_decision_records(self, tmp_path, monkeypatch):
        run_id = f"test_weights_persist_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        decision = engine.receive_profile(profile)

        # Retrieve written decision record from outputs/<run_id>/decision_logs/
        rec_path = tmp_path / "outputs" / run_id / "decision_logs" / f"{run_id}_0000.json"
        assert rec_path.exists(), f"Expected {rec_path} to exist"

        record_data = json.loads(rec_path.read_text(encoding="utf-8"))
        assert "coordinator_weights" in record_data
        weights = record_data["coordinator_weights"]
        assert "performance_weight" in weights
        assert "efficiency_weight" in weights
        assert "stability_weight" in weights
        assert weights["iteration"] == 0


# ===========================================================================
# P0-24: Mechanism-based decision-source attribution
# ===========================================================================
class TestP0_24_DecisionSourceClassification:
    def test_rule_selected_decision_source_rule(self, tmp_path, monkeypatch):
        """Rule path produces source='rule'."""
        run_id = f"test_source_rule_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id, llm_mode=False, enable_value_model=False)
        profile = _make_dummy_profile()
        engine.receive_profile(profile)

        res = _make_dummy_result(experiment_id=run_id, iteration=1)
        decision = engine.receive_result(res)
        assert decision.reason.source == "rule"

    def test_llm_selected_decision_source_llm(self, tmp_path, monkeypatch):
        """LLM-generated ranking propagates source='llm'."""
        run_id = f"test_source_llm_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        engine.receive_profile(profile)

        with patch("stratml.decision.agents.coordinator_agent.is_llm_enabled", return_value=True), \
             patch("stratml.decision.agents.coordinator_agent._llm_rank") as mock_llm:
            from stratml.decision.agents.coordinator_agent import RankedAction
            mock_llm.return_value = [
                RankedAction(
                    action_type="switch_model",
                    parameters={"model_name": "GradientBoostingClassifier"},
                    predicted_gain=0.08,
                    predicted_cost=0.5,
                    confidence=0.85,
                    agent_scores=AgentScore(performance=0.85, efficiency=0.6, stability=0.7),
                    final_score=0.82,
                    rationale="LLM selected GradientBoostingClassifier based on feature interactions.",
                    source="llm",
                )
            ]
            res = _make_dummy_result(experiment_id=run_id, iteration=1)
            decision = engine.receive_result(res)
            assert decision.reason.source == "llm"

    def test_value_model_selected_decision_source_value_model(self, tmp_path):
        """Pure value-model-driven selection propagates source='value_model'."""
        from stratml.decision.agents.coordinator_agent import RankedAction
        from stratml.decision.policy.action_selector import select
        state = _make_state(iteration=2)
        ranked = [
            RankedAction(
                action_type="switch_model",
                parameters={"model_name": "SVC"},
                predicted_gain=0.12,
                predicted_cost=0.4,
                confidence=0.9,
                agent_scores=AgentScore(performance=0.9, efficiency=0.7, stability=0.8),
                final_score=0.88,
                source="value_model",
            )
        ]
        decision = select(state, ranked)
        assert decision.reason.source == "value_model"

    def test_hybrid_decision_source_hybrid(self, tmp_path, monkeypatch):
        """Combination of learned value model predictions + coordinator weights produces source='hybrid'."""
        from stratml.decision.agents import coordinator_agent
        state = _make_state(iteration=2)
        estimates = [
            UncertaintyEstimate(action_type="switch_model", parameters={}, predicted_gain=0.08, predicted_cost=0.5, confidence=0.8, variance=0.01)
        ]
        ranked = coordinator_agent.rank(
            state, estimates,
            perf_scores={"switch_model": 0.8},
            eff_scores={"switch_model": 0.6},
            stab_scores={"switch_model": 0.7},
            has_learned_model=True,
        )
        assert ranked[0].source == "hybrid"

        from stratml.decision.policy.action_selector import select
        decision = select(state, ranked)
        assert decision.reason.source == "hybrid"

    def test_unavailable_mechanism_with_fallback_source_fallback(self, tmp_path, monkeypatch):
        """When LLM is enabled but fails, explicit fallback source='fallback' is propagated."""
        run_id = f"test_source_fallback_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        engine.receive_profile(profile)

        with patch("stratml.decision.agents.coordinator_agent.is_llm_enabled", return_value=True), \
             patch("stratml.decision.agents.coordinator_agent._llm_rank", return_value=None):
            res = _make_dummy_result(experiment_id=run_id, iteration=1)
            decision = engine.receive_result(res)
            assert decision.reason.source == "fallback"

    def test_bootstrap_decision_source_bootstrap(self, tmp_path, monkeypatch):
        """Initial decision at iteration 0 produces source='bootstrap'."""
        run_id = f"test_source_bootstrap_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        decision = engine.receive_profile(profile)
        assert decision.reason.source == "bootstrap"

    def test_llm_enabled_with_rationale_does_not_imply_llm_unless_llm_determined_decision(self):
        """LLM enabled + non-empty rationale does NOT yield 'llm' if fallback or rule determined the action."""
        from stratml.decision.agents.coordinator_agent import RankedAction
        from stratml.decision.policy.action_selector import select
        state = _make_state(iteration=2)
        # Action has rationale attached, but its mechanism source is 'fallback'
        ranked = [
            RankedAction(
                action_type="switch_model",
                parameters={"model_name": "GradientBoostingClassifier"},
                predicted_gain=0.05,
                predicted_cost=0.5,
                confidence=0.8,
                agent_scores=AgentScore(performance=0.8, efficiency=0.5, stability=0.7),
                final_score=0.75,
                rationale="Candidate rationale string from previous state inspection",
                source="fallback",
            )
        ]
        decision = select(state, ranked)
        assert decision.reason.source == "fallback"
        assert decision.reason.source != "llm"


# ===========================================================================
# P0-25: Complete state -> action -> outcome trajectory reconstructability
# ===========================================================================
class TestP0_25_TrajectoryReconstructability:
    def test_consecutive_iterations_reconstructable_from_persisted_records(self, tmp_path, monkeypatch):
        run_id = f"test_traj_rec_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        d0 = engine.receive_profile(profile)

        # Simulate execution of iteration 1
        r1 = _make_dummy_result(experiment_id=run_id, iteration=1, accuracy=0.80)
        d1 = engine.receive_result(r1)

        # Simulate execution of iteration 2
        r2 = _make_dummy_result(experiment_id=run_id, iteration=2, accuracy=0.85)
        d2 = engine.receive_result(r2)

        # Verify record for iteration 0
        rec0_path = tmp_path / "outputs" / run_id / "decision_logs" / f"{run_id}_0000.json"
        assert rec0_path.exists()
        rec0 = json.loads(rec0_path.read_text(encoding="utf-8"))

        # Verify record for iteration 1
        rec1_path = tmp_path / "outputs" / run_id / "decision_logs" / f"{run_id}_0001.json"
        assert rec1_path.exists()
        rec1 = json.loads(rec1_path.read_text(encoding="utf-8"))

        # Iteration 0 has execution_result linked from iteration 1 outcome
        assert rec0.get("execution_result") is not None
        assert rec0["execution_result"]["iteration"] == 1
        assert rec0["next_state_id"] == f"{run_id}_1"

        # Iteration 1 has execution_result linked from iteration 2 outcome
        assert rec1.get("execution_result") is not None
        assert rec1["execution_result"]["iteration"] == 2
        assert rec1["next_state_id"] == f"{run_id}_2"

        # Both records have state snapshot, candidate actions, selected action
        assert "state_snapshot" in rec0 and "state_snapshot" in rec1
        assert "candidate_actions" in rec0 and "candidate_actions" in rec1
        assert "selected_action" in rec0 and "selected_action" in rec1


# ===========================================================================
# P0-26: Preserve candidate / runner-up ranking information
# ===========================================================================
class TestP0_26_CandidateAndRunnerUpTraceability:
    def test_ranked_candidates_contain_runner_up_information(self, tmp_path, monkeypatch):
        run_id = f"test_runner_up_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        engine.receive_profile(profile)

        rec_path = tmp_path / "outputs" / run_id / "decision_logs" / f"{run_id}_0000.json"
        record = json.loads(rec_path.read_text(encoding="utf-8"))

        assert "ranked_candidates" in record
        ranked = record["ranked_candidates"]
        assert len(ranked) >= 2, "Expected at least 2 candidates to evaluate runner-up"

        # Candidates must be sorted by rank
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2
        # Exactly one candidate is marked selected, matching the selected_action
        selected_candidates = [c for c in ranked if c["selected"]]
        assert len(selected_candidates) == 1
        assert selected_candidates[0]["candidate"]["action_type"] == record["selected_action"]["action_type"]

        # Verify candidate scoring fields exist
        for item in ranked:
            assert "predicted_gain" in item
            assert "predicted_cost" in item
            assert "confidence" in item
            assert "agent_scores" in item
            assert "final_score" in item

        # Runner-up comparison: Rank 1 score >= Rank 2 score
        assert ranked[0]["final_score"] >= ranked[1]["final_score"]


# ===========================================================================
# P0-27: Enforce state-transition integrity
# ===========================================================================
class TestP0_27_StateTransitionIntegrity:
    def test_successful_action_marks_previous_action_success_true(self, tmp_path, monkeypatch):
        run_id = f"test_st_succ_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        engine.receive_profile(profile)

        # Baseline established at 0.70
        r1 = _make_dummy_result(experiment_id=run_id, iteration=1, accuracy=0.70)
        engine.receive_result(r1)

        # Improvement: 0.75 > 0.70 -> success
        r2 = _make_dummy_result(experiment_id=run_id, iteration=2, accuracy=0.75)
        import stratml.decision.engine as engine_mod
        with patch("stratml.decision.engine.build_state", wraps=engine_mod.build_state) as spy_build:
            engine.receive_result(r2)
            # Inspect call arguments to build_state
            kwargs = spy_build.call_args.kwargs
            assert kwargs.get("previous_action_success") is True

    def test_degrading_action_marks_previous_action_success_false(self, tmp_path, monkeypatch):
        run_id = f"test_st_deg_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        engine.receive_profile(profile)

        # Baseline established at 0.80
        r1 = _make_dummy_result(experiment_id=run_id, iteration=1, accuracy=0.80)
        engine.receive_result(r1)

        # Degradation: 0.70 < 0.80 -> failure
        r2 = _make_dummy_result(experiment_id=run_id, iteration=2, accuracy=0.70)
        import stratml.decision.engine as engine_mod
        with patch("stratml.decision.engine.build_state", wraps=engine_mod.build_state) as spy_build:
            engine.receive_result(r2)
            kwargs = spy_build.call_args.kwargs
            assert kwargs.get("previous_action_success") is False

    def test_explicitly_failed_experiment_marks_previous_action_success_false(self, tmp_path, monkeypatch):
        run_id = f"test_st_fail_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        engine = DecisionEngine(run_id=run_id)
        profile = _make_dummy_profile()
        engine.receive_profile(profile)

        r1 = _make_dummy_result(experiment_id=run_id, iteration=1, accuracy=0.70, failed=True)
        import stratml.decision.engine as engine_mod
        with patch("stratml.decision.engine.build_state", wraps=engine_mod.build_state) as spy_build:
            engine.receive_result(r1)
            kwargs = spy_build.call_args.kwargs
            assert kwargs.get("previous_action_success") is False


# ===========================================================================
# P0-28: Strengthen warm-start provenance/isolation in backfill_last_gain
# ===========================================================================
class TestP0_28_InterleavedRunIsolationInBackfill:
    def test_interleaved_runs_do_not_cross_contaminate_pending_observations(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        profile = _make_dummy_profile()
        engine_A = DecisionEngine(run_id="run_A", history_mode="continual")
        engine_B = DecisionEngine(run_id="run_B", history_mode="continual")

        dataset_csv = tmp_path / "decision_dataset.csv"
        monkeypatch.setattr(dataset_builder, "_DATASET_PATH", dataset_csv)
        monkeypatch.setattr(dataset_builder, "_UNIFIED_PATH", dataset_csv)

        # Step 1: Run A iteration 0
        dA0 = engine_A.receive_profile(profile)
        # Step 2: Run B iteration 0 (interleaved before Run A receives result)
        dB0 = engine_B.receive_profile(profile)

        # Both rows should exist with empty observed_gain
        df_pending = pd.read_csv(dataset_csv)
        assert len(df_pending) == 2
        assert df_pending.iloc[0]["run_id"] == "run_A"
        assert df_pending.iloc[1]["run_id"] == "run_B"
        assert pd.isna(df_pending.iloc[0]["observed_gain"]) or str(df_pending.iloc[0]["observed_gain"]).strip() == ""
        assert pd.isna(df_pending.iloc[1]["observed_gain"]) or str(df_pending.iloc[1]["observed_gain"]).strip() == ""

        # Step 3: Run A receives result for iteration 1 (gain = +0.05)
        rA1 = _make_dummy_result(experiment_id="run_A", iteration=1, accuracy=0.85)
        engine_A.receive_result(rA1)

        # Run A's row must be backfilled, but Run B's row MUST REMAIN EMPTY
        df_after_A = pd.read_csv(dataset_csv)
        row_A = df_after_A[df_after_A["run_id"] == "run_A"].iloc[0]
        row_B = df_after_A[df_after_A["run_id"] == "run_B"].iloc[0]

        assert str(row_A["observed_gain"]).strip() != "", "Run A observed_gain should be backfilled"
        assert pd.isna(row_B["observed_gain"]) or str(row_B["observed_gain"]).strip() == "", "Run B must NOT be contaminated by Run A's backfill"

        # Step 4: Run B receives result for iteration 1 (gain = +0.02)
        rB1 = _make_dummy_result(experiment_id="run_B", iteration=1, accuracy=0.72)
        engine_B.receive_result(rB1)

        # Now Run B's row is backfilled
        df_after_B = pd.read_csv(dataset_csv)
        row_B_final = df_after_B[df_after_B["run_id"] == "run_B"].iloc[0]
        assert str(row_B_final["observed_gain"]).strip() != "", "Run B observed_gain should now be backfilled"


# ===========================================================================
# P0-30: Machine-readable experiment manifest
# ===========================================================================
class TestP0_30_ExperimentManifest:
    def test_completed_run_produces_valid_manifest(self, tmp_path, monkeypatch):
        run_id = f"test_manifest_{uuid.uuid4().hex[:6]}"
        monkeypatch.chdir(tmp_path)

        csv_path = tmp_path / "synthetic_dataset.csv"
        df = pd.DataFrame({
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "target": [0, 1] * 25,
        })
        df.to_csv(csv_path, index=False)

        engine = DecisionEngine(run_id=run_id, max_iterations=2)
        orchestrator = ExecutionOrchestrator(
            send_profile=engine.receive_profile,
            send_result=engine.receive_result,
            run_id=run_id,
            max_iterations=2,
            split_config=SplitConfig(method="stratified", random_seed=42),
        )

        orchestrator.run(str(csv_path), "target")

        manifest_path = tmp_path / "outputs" / run_id / "manifest.json"
        assert manifest_path.exists(), f"Expected manifest file at {manifest_path}"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        required_keys = [
            "run_id",
            "dataset",
            "dataset_fingerprint",
            "task",
            "seed",
            "budget",
            "model_space",
            "action_space",
            "hyperparameter_mutation_space",
            "llm_configuration",
            "warm_start_mode_corpus",
            "stratml_version",
            "evaluation_configuration",
        ]
        for key in required_keys:
            assert key in manifest, f"Manifest is missing required key: {key}"

        # Value consistency
        assert manifest["run_id"] == run_id
        assert manifest["seed"] == 42
        assert manifest["task"] == "classification"
        assert manifest["dataset"]["target_column"] == "target"
        assert isinstance(manifest["model_space"], list)
        assert isinstance(manifest["action_space"], list)
        assert isinstance(manifest["hyperparameter_mutation_space"], dict)
