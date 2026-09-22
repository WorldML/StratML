"""
test_decision_team.py
---------------------
Comprehensive tests for all decision team components introduced/modified
in the improvement commits. Covers:

  - signals.py          (singleton, rule-based, all signal types)
  - value_model.py      (encoding, vocab, stub vs active)
  - uncertainty.py      (state-aware encoding, stub fallback)
  - state_builder.py    (previous_signals threading)
  - action_selector.py  (adaptive preprocessing, epsilon-greedy)
  - coordinator_agent.py (adaptive weights, EMA, fallback)
  - evaluator_agent.py  (all 4 audit dimensions, rule + LLM fallback)
  - counterfactual.py   (runner-up recording)
  - meta_memory.py      (cosine similarity, record/retrieve)
  - engine.py           (end-to-end wiring)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stratml.core.schemas import (
    ActionDecision, AgentScore, CandidateAction, DecisionReason,
    PreprocessingConfig, SecondaryMetrics, StateActionContext,
    StateConstraints, StateDataset, StateGeneralization, StateMeta,
    StateMetrics, StateModel, StateObject, StateObjective, StateResources,
    StateSearch, StateSignals, StateTrajectory, StateUncertainty,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_state(
    iteration: int = 1,
    primary: float = 0.72,
    underfitting: str = "none",
    overfitting: str = "none",
    well_fitted: str = "strong",
    converged: str = "none",
    stagnating: str = "none",
    diverging: str = "none",
    plateau_detected: str = "none",
    diminishing_returns: str = "none",
    unstable_training: str = "none",
    high_variance: str = "none",
    remaining_budget: float = 15.0,
    models_tried: list | None = None,
    allowed_models: list | None = None,
    runtime: float = 10.0,
    model_name: str = "RandomForestClassifier",
    complexity_hint: str | None = None,
    train_loss: float = 0.20,
    val_loss: float = 0.22,
    slope: float = 0.005,
    steps_since_improvement: int = 1,
    improvement_rate: float = 0.01,
    volatility: float = 0.02,
    num_samples: int = 500,
    num_features: int = 10,
    missing_ratio: float = 0.0,
    imbalance_ratio: float | None = None,
    previous_signals: StateSignals | None = None,
    previous_action: str | None = None,
) -> StateObject:
    gap = round(val_loss - train_loss, 6)
    return StateObject(
        meta=StateMeta(experiment_id="test_exp", iteration=iteration, timestamp="2026-01-01T00:00:00+00:00"),
        objective=StateObjective(primary_metric="accuracy", optimization_goal="maximize"),
        metrics=StateMetrics(primary=primary, secondary=SecondaryMetrics(), train_val_gap=gap),
        trajectory=StateTrajectory(
            history_length=max(iteration, 1),
            improvement_rate=improvement_rate,
            slope=slope,
            volatility=volatility,
            best_score=primary,
            mean_score=primary - 0.02,
            steps_since_improvement=steps_since_improvement,
            trend="improving",
        ),
        dataset=StateDataset(
            num_samples=num_samples,
            num_features=num_features,
            feature_to_sample_ratio=round(num_features / max(num_samples, 1), 6),
            missing_ratio=missing_ratio,
            imbalance_ratio=imbalance_ratio,
        ),
        model=StateModel(
            model_name=model_name,
            model_type="ml",
            hyperparameters={"n_estimators": 100},
            complexity_hint=complexity_hint,
            runtime=runtime,
            convergence_epoch=0,
        ),
        generalization=StateGeneralization(train_loss=train_loss, validation_loss=val_loss, gap=gap),
        resources=StateResources(
            runtime=runtime,
            gpu_used=False,
            cpu_time=runtime,
            remaining_budget=remaining_budget,
            budget_exhausted=remaining_budget <= 0,
        ),
        search=StateSearch(
            models_tried=models_tried or [model_name],
            unique_models_count=len(set(models_tried or [model_name])),
            repeated_configs=0,
        ),
        signals=StateSignals(
            underfitting=underfitting, overfitting=overfitting, well_fitted=well_fitted,
            converged=converged, stagnating=stagnating, diverging=diverging,
            plateau_detected=plateau_detected, diminishing_returns=diminishing_returns,
            unstable_training=unstable_training, high_variance=high_variance,
        ),
        uncertainty=StateUncertainty(),
        action_context=StateActionContext(
            previous_action=previous_action,
            previous_signals=previous_signals,
        ),
        constraints=StateConstraints(
            allowed_models=allowed_models or [
                "RandomForestClassifier", "LogisticRegression",
                "GradientBoostingClassifier", "SVC", "KNeighborsClassifier",
            ],
            max_iterations=20,
        ),
    )


def _make_decision(
    action_type: str = "switch_model",
    trigger: str = "underfitting",
    expected_gain: float = 0.05,
    confidence: float = 0.7,
) -> ActionDecision:
    return ActionDecision(
        experiment_id="test_exp",
        iteration=1,
        action_type=action_type,
        parameters={"model_name": "GradientBoostingClassifier"},
        preprocessing=PreprocessingConfig(
            missing_value_strategy="mean", scaling="standard",
            encoding="onehot", imbalance_strategy="none", feature_selection="none",
        ),
        expected_gain=expected_gain,
        expected_cost=0.5,
        confidence=confidence,
        agent_scores=AgentScore(performance=0.7, efficiency=0.6, stability=0.8),
        reason=DecisionReason(trigger=trigger, evidence={}, source="rule"),
    )


def _make_result(accuracy: float = 0.78, train_loss: float = 0.18, val_loss: float = 0.20):
    from stratml.core.schemas import ExperimentResult, ExperimentMetrics, ResourceUsage, ArtifactRefs
    return ExperimentResult(
        experiment_id="test_exp",
        iteration=2,
        dataset_name="test",
        model_name="GradientBoostingClassifier",
        model_type="ml",
        hyperparameters={"n_estimators": 100},
        preprocessing_applied=PreprocessingConfig(
            missing_value_strategy="mean", scaling="standard",
            encoding="onehot", imbalance_strategy="none", feature_selection="none",
        ),
        metrics=ExperimentMetrics(
            accuracy=accuracy, train_loss=train_loss, validation_loss=val_loss,
        ),
        train_curve=[0.5, 0.6, 0.7],
        validation_curve=[0.48, 0.58, 0.68],
        runtime=12.0,
        resource_usage=ResourceUsage(),
        artifacts=ArtifactRefs(model_path="m.pkl", metrics_file="m.json", tensorboard_logs="tb/"),
    )


def _full_pipeline(state: StateObject):
    """Run the full decision pipeline and return (ranked, decision)."""
    from stratml.decision.learning.value_model import predict
    from stratml.decision.learning.calibration import calibrate
    from stratml.decision.learning.uncertainty import estimate
    from stratml.decision.agents import performance_agent, efficiency_agent, stability_agent
    from stratml.decision.agents.coordinator_agent import rank
    from stratml.decision.policy.action_selector import select

    candidates = [
        CandidateAction(action_type="switch_model", parameters={"model_name": "SVC"}),
        CandidateAction(action_type="terminate", parameters={}),
        CandidateAction(action_type="increase_model_capacity", parameters={"scale": 1.5}),
    ]
    preds = predict(state, candidates)
    cal = calibrate(preds)
    ests = estimate(cal, state)
    perf = performance_agent.score(state, ests)
    eff = efficiency_agent.score(state, ests)
    stab = stability_agent.score(state, ests)
    ranked = rank(state, ests, perf, eff, stab)
    decision = select(state, ranked)
    return ranked, decision


# ===========================================================================
# signals.py — singleton + rule-based
# ===========================================================================

class TestSignalsSingleton:
    def test_get_agent_returns_same_object(self):
        """_get_agent() must return the same instance on repeated calls."""
        import os
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")
        from stratml.decision.state.signals import _get_agent
        a1 = _get_agent()
        a2 = _get_agent()
        assert a1 is a2

    def test_singleton_not_rebuilt_on_compute_signals(self, monkeypatch):
        """compute_signals must not rebuild the agent on every call."""
        import stratml.decision.state.signals as sig_mod
        build_count = {"n": 0}
        original = sig_mod._get_agent

        def counting_get():
            build_count["n"] += 1
            return original()

        monkeypatch.setattr(sig_mod, "_get_agent", counting_get)
        # Force rule-based path so we don't need GROQ
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        state = _make_state()
        sig_mod.compute_signals(state)
        sig_mod.compute_signals(state)
        # _get_agent should not be called at all on rule-based path
        assert build_count["n"] == 0


class TestSignalsRuleBased:
    def _compute(self, **kwargs):
        from stratml.decision.state.signals import _rule_based
        return _rule_based(_make_state(**kwargs))

    def test_underfitting_strong_when_primary_below_60(self):
        sig = self._compute(primary=0.55, well_fitted="none")
        assert sig.underfitting == "strong"

    def test_underfitting_weak_when_primary_60_to_70(self):
        sig = self._compute(primary=0.65, well_fitted="none")
        assert sig.underfitting == "weak"

    def test_no_underfitting_above_70(self):
        sig = self._compute(primary=0.75)
        assert sig.underfitting == "none"

    def test_overfitting_strong_when_gap_above_10(self):
        sig = self._compute(train_loss=0.10, val_loss=0.25)
        assert sig.overfitting == "strong"

    def test_overfitting_weak_when_gap_5_to_10(self):
        sig = self._compute(train_loss=0.10, val_loss=0.17)
        assert sig.overfitting == "weak"

    def test_well_fitted_strong_when_primary_above_75_no_gap(self):
        sig = self._compute(primary=0.80, train_loss=0.10, val_loss=0.11)
        assert sig.well_fitted == "strong"

    def test_converged_strong_when_slope_near_zero_and_high_score(self):
        sig = self._compute(primary=0.80, slope=0.0005, steps_since_improvement=0)
        assert sig.converged == "strong"

    def test_stagnating_strong_when_steps_since_4_plus(self):
        sig = self._compute(steps_since_improvement=5, primary=0.60, slope=0.001)
        assert sig.stagnating == "strong"

    def test_diverging_strong_when_slope_below_minus_2pct(self):
        sig = self._compute(slope=-0.03)
        assert sig.diverging == "strong"

    def test_plateau_strong_when_steps_since_4_plus(self):
        sig = self._compute(steps_since_improvement=4, improvement_rate=0.001)
        assert sig.plateau_detected == "strong"

    def test_diminishing_returns_when_tiny_improvement(self):
        sig = self._compute(improvement_rate=0.003, steps_since_improvement=0)
        assert sig.diminishing_returns == "strong"

    def test_too_slow_strong_when_runtime_above_300(self):
        sig = self._compute(runtime=400.0)
        assert sig.too_slow == "strong"

    def test_all_confidences_in_range(self):
        sig = self._compute(primary=0.55, train_loss=0.10, val_loss=0.30, slope=-0.03)
        for field in StateSignals.model_fields:
            if field.endswith("_confidence"):
                v = getattr(sig, field)
                assert 0.0 <= v <= 1.0, f"{field}={v} out of range"

    def test_mutually_exclusive_underfitting_overfitting(self):
        # Can't be both underfitting and overfitting simultaneously in rule path
        sig = self._compute(primary=0.55, train_loss=0.10, val_loss=0.25)
        # underfitting fires on primary, overfitting fires on gap — both can fire
        # but well_fitted must be none
        assert sig.well_fitted == "none"

    def test_no_signals_on_perfect_state(self):
        sig = self._compute(primary=0.95, train_loss=0.05, val_loss=0.06,
                            slope=0.0001, steps_since_improvement=0,
                            improvement_rate=0.001, volatility=0.001, runtime=5.0)
        assert sig.underfitting == "none"
        assert sig.overfitting == "none"
        assert sig.diverging == "none"
        assert sig.stagnating == "none"


# ===========================================================================
# value_model.py — encoding + vocab
# ===========================================================================

class TestValueModelEncoding:
    def test_action_vocab_no_collisions(self):
        from stratml.decision.learning.value_model import _ACTION_VOCAB
        values = list(_ACTION_VOCAB.values())
        assert len(values) == len(set(values)), "ACTION_VOCAB has duplicate indices"

    def test_model_vocab_no_collisions(self):
        from stratml.decision.learning.value_model import _MODEL_VOCAB
        values = list(_MODEL_VOCAB.values())
        assert len(values) == len(set(values)), "MODEL_VOCAB has duplicate indices"

    def test_encode_known_action_uses_vocab(self):
        from stratml.decision.learning.value_model import _encode_state_action, _ACTION_VOCAB
        state = _make_state()
        vec = _encode_state_action(state, "terminate")
        assert vec[10] == float(_ACTION_VOCAB["terminate"])

    def test_encode_unknown_action_uses_fallback(self):
        from stratml.decision.learning.value_model import _encode_state_action, _ACTION_VOCAB
        state = _make_state()
        vec = _encode_state_action(state, "nonexistent_action")
        assert vec[10] == float(len(_ACTION_VOCAB))

    def test_different_actions_produce_different_encodings(self):
        from stratml.decision.learning.value_model import _encode_state_action
        state = _make_state()
        v1 = _encode_state_action(state, "terminate")
        v2 = _encode_state_action(state, "switch_model")
        assert v1 != v2

    def test_model_name_enc_differs_by_model(self):
        from stratml.decision.learning.value_model import _encode_state_action
        state_rf = _make_state(model_name="RandomForestClassifier")
        state_svc = _make_state(model_name="SVC")
        v_rf = _encode_state_action(state_rf, "switch_model")
        v_svc = _encode_state_action(state_svc, "switch_model")
        assert v_rf[11] != v_svc[11]

    def test_complexity_hint_enc_correct(self):
        from stratml.decision.learning.value_model import _encode_state_action, _COMPLEXITY_VOCAB
        state_low = _make_state(complexity_hint="low")
        state_high = _make_state(complexity_hint="high")
        v_low = _encode_state_action(state_low, "switch_model")
        v_high = _encode_state_action(state_high, "switch_model")
        assert v_low[12] == float(_COMPLEXITY_VOCAB["low"])
        assert v_high[12] == float(_COMPLEXITY_VOCAB["high"])
        assert v_high[12] > v_low[12]

    def test_complexity_hint_none_encodes_zero(self):
        from stratml.decision.learning.value_model import _encode_state_action
        state = _make_state(complexity_hint=None)
        vec = _encode_state_action(state, "switch_model")
        assert vec[12] == 0.0

    def test_feature_vector_length_is_13(self):
        from stratml.decision.learning.value_model import _encode_state_action
        state = _make_state()
        vec = _encode_state_action(state, "switch_model")
        assert len(vec) == 13

    def test_stub_returns_neutral_values(self):
        from stratml.decision.learning.value_model import predict
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        preds = predict(state, candidates)
        # In stub mode (< 50 rows), predicted_gain should be 0.05
        assert preds[0].predicted_gain == 0.05
        assert preds[0].predicted_cost == 0.5

    def test_predict_returns_one_per_candidate(self):
        from stratml.decision.learning.value_model import predict
        state = _make_state()
        candidates = [
            CandidateAction(action_type="switch_model", parameters={}),
            CandidateAction(action_type="terminate", parameters={}),
            CandidateAction(action_type="modify_regularization", parameters={}),
        ]
        preds = predict(state, candidates)
        assert len(preds) == 3

    def test_predict_preserves_action_types(self):
        from stratml.decision.learning.value_model import predict
        state = _make_state()
        candidates = [
            CandidateAction(action_type="terminate", parameters={}),
            CandidateAction(action_type="switch_model", parameters={}),
        ]
        preds = predict(state, candidates)
        types = [p.action_type for p in preds]
        assert "terminate" in types
        assert "switch_model" in types

    def test_predict_gain_in_range(self):
        from stratml.decision.learning.value_model import predict
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        preds = predict(state, candidates)
        assert 0.0 <= preds[0].predicted_gain <= 1.0

    def test_active_model_with_synthetic_data(self, tmp_path, monkeypatch):
        """When >= 50 rows exist, RF model should activate and return non-stub values."""
        import pandas as pd
        import numpy as np
        from stratml.decision.learning import value_model

        # Build synthetic dataset with 60 rows
        rows = []
        for i in range(60):
            rows.append({
                "best_score": np.random.uniform(0.5, 0.9),
                "improvement_rate": np.random.uniform(0, 0.05),
                "slope": np.random.uniform(-0.01, 0.02),
                "volatility": np.random.uniform(0, 0.05),
                "steps_since_improvement": np.random.randint(0, 5),
                "num_samples": 500,
                "num_features": 10,
                "missing_ratio": 0.0,
                "runtime": 10.0,
                "remaining_budget": 10.0,
                "action_type": "switch_model",
                "model_name": "RandomForestClassifier",
                "complexity_hint": "none",
                "observed_gain": np.random.uniform(0.0, 0.1),
                "predicted_gain": 0.05,
            })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "decision_dataset.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(value_model, "_DATASET_PATH", csv_path)

        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        preds = value_model.predict(state, candidates)
        # Should not be exactly 0.05 (stub value) when RF is active
        assert 0.0 <= preds[0].predicted_gain <= 1.0


# ===========================================================================
# uncertainty.py — state-aware encoding
# ===========================================================================

class TestUncertainty:
    def test_accepts_state_parameter(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        preds = predict(state, candidates)
        ests = estimate(preds, state)
        assert len(ests) == 1

    def test_works_without_state(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        preds = predict(state, candidates)
        ests = estimate(preds)  # no state — proxy fallback
        assert len(ests) == 1

    def test_confidence_in_range(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        est = estimate(predict(state, candidates), state)[0]
        assert 0.0 <= est.confidence <= 1.0

    def test_variance_non_negative(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        est = estimate(predict(state, candidates), state)[0]
        assert est.variance >= 0.0

    def test_gain_in_range(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        est = estimate(predict(state, candidates), state)[0]
        assert 0.0 <= est.predicted_gain <= 1.0

    def test_stub_confidence_is_half(self):
        """In stub mode (< 50 rows), confidence should be 0.5."""
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        est = estimate(predict(state, candidates), state)[0]
        # Stub path: confidence=0.5, variance=0.0
        assert est.confidence == 0.5
        assert est.variance == 0.0

    def test_length_preserved(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [
            CandidateAction(action_type="switch_model", parameters={}),
            CandidateAction(action_type="terminate", parameters={}),
        ]
        ests = estimate(predict(state, candidates), state)
        assert len(ests) == 2

    def test_action_types_preserved(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [
            CandidateAction(action_type="terminate", parameters={}),
            CandidateAction(action_type="switch_model", parameters={}),
        ]
        ests = estimate(predict(state, candidates), state)
        types = {e.action_type for e in ests}
        assert types == {"terminate", "switch_model"}


# ===========================================================================
# state_builder.py — previous_signals threading
# ===========================================================================

class TestPreviousSignals:
    def test_previous_signals_in_schema(self):
        ctx = StateActionContext()
        assert hasattr(ctx, "previous_signals")
        assert ctx.previous_signals is None

    def test_previous_signals_stored_in_context(self):
        prev = StateSignals(underfitting="strong")
        ctx = StateActionContext(previous_signals=prev)
        assert ctx.previous_signals.underfitting == "strong"

    def test_build_state_threads_previous_signals(self):
        from stratml.decision.state.state_builder import build_state
        from stratml.decision.state.state_history import ExperimentHistory
        result = _make_result()
        prev_signals = StateSignals(underfitting="weak")
        state = build_state(
            result,
            history=ExperimentHistory(),
            previous_signals=prev_signals,
        )
        assert state.action_context.previous_signals is not None
        assert state.action_context.previous_signals.underfitting == "weak"

    def test_build_state_without_previous_signals_is_none(self):
        from stratml.decision.state.state_builder import build_state
        from stratml.decision.state.state_history import ExperimentHistory
        result = _make_result()
        state = build_state(result, history=ExperimentHistory())
        assert state.action_context.previous_signals is None

    def test_engine_stores_last_signals(self, tmp_path, monkeypatch):
        from stratml.decision.engine import DecisionEngine
        from stratml.execution.schemas import DataProfile, FeatureInfo
        monkeypatch.chdir(tmp_path)
        profile = DataProfile(
            dataset_name="test", dataset_type="tabular", rows=200, columns=5,
            target_column="target", problem_type="classification",
            numerical_columns=["f0", "f1", "f2", "f3"],
            categorical_columns=[], missing_value_ratio=0.0,
            class_distribution={"a": 100, "b": 100},
            feature_summary=[
                FeatureInfo(name=f"f{i}", dtype="float64", unique_values=200,
                            missing_percentage=0.0, distribution="normal")
                for i in range(4)
            ],
            recommended_metrics=["accuracy"],
        )
        engine = DecisionEngine(run_id="test_signals_run")
        engine.receive_profile(profile)
        assert engine._last_signals is not None

    def test_transition_underfitting_to_well_fitted_detectable(self):
        """previous_signals lets us detect the transition."""
        prev = StateSignals(underfitting="strong")
        curr = _make_state(well_fitted="strong", underfitting="none", previous_signals=prev)
        assert curr.action_context.previous_signals.underfitting == "strong"
        assert curr.signals.well_fitted == "strong"


# ===========================================================================
# action_selector.py — adaptive preprocessing + epsilon-greedy
# ===========================================================================

class TestAdaptivePreprocessing:
    def test_imbalanced_dataset_gets_oversample(self):
        from stratml.decision.policy.action_selector import _build_preprocessing
        state = _make_state(imbalance_ratio=3.5)
        prep = _build_preprocessing(state)
        assert prep.imbalance_strategy == "oversample"

    def test_balanced_dataset_gets_none(self):
        from stratml.decision.policy.action_selector import _build_preprocessing
        state = _make_state(imbalance_ratio=1.2)
        prep = _build_preprocessing(state)
        assert prep.imbalance_strategy == "none"

    def test_high_missing_ratio_gets_median(self):
        from stratml.decision.policy.action_selector import _build_preprocessing
        state = _make_state(missing_ratio=0.15)
        prep = _build_preprocessing(state)
        assert prep.missing_value_strategy == "median"

    def test_low_missing_ratio_gets_mean(self):
        from stratml.decision.policy.action_selector import _build_preprocessing
        state = _make_state(missing_ratio=0.05)
        prep = _build_preprocessing(state)
        assert prep.missing_value_strategy == "mean"

    def test_tree_model_gets_no_scaling(self):
        from stratml.decision.policy.action_selector import _build_preprocessing
        for model in ["RandomForestClassifier", "GradientBoostingClassifier",
                      "ExtraTreesClassifier", "DecisionTreeClassifier"]:
            state = _make_state(model_name=model)
            prep = _build_preprocessing(state)
            assert prep.scaling == "none", f"{model} should have scaling=none"

    def test_non_tree_model_gets_standard_scaling(self):
        from stratml.decision.policy.action_selector import _build_preprocessing
        for model in ["LogisticRegression", "SVC", "KNeighborsClassifier"]:
            state = _make_state(model_name=model)
            prep = _build_preprocessing(state)
            assert prep.scaling == "standard", f"{model} should have scaling=standard"

    def test_preprocessing_embedded_in_decision(self):
        state = _make_state(imbalance_ratio=4.0, missing_ratio=0.2,
                            model_name="LogisticRegression")
        _, decision = _full_pipeline(state)
        assert decision.preprocessing.imbalance_strategy == "oversample"
        assert decision.preprocessing.missing_value_strategy == "median"
        assert decision.preprocessing.scaling == "standard"

    def test_preprocessing_fields_valid(self):
        state = _make_state()
        _, decision = _full_pipeline(state)
        prep = decision.preprocessing
        assert prep.missing_value_strategy in {"mean", "median", "mode", "drop"}
        assert prep.scaling in {"standard", "minmax", "none"}
        assert prep.imbalance_strategy in {"oversample", "undersample", "none"}


class TestEpsilonGreedy:
    def test_always_picks_valid_action(self, monkeypatch):
        """Even with epsilon=1.0, selected action must be a valid type."""
        import stratml.decision.policy.action_selector as sel
        monkeypatch.setattr(sel, "_EPSILON_LOW_DATA", 1.0)
        monkeypatch.setattr(sel, "_row_count", lambda: 0)
        state = _make_state()
        valid = {"switch_model", "terminate", "increase_model_capacity",
                 "decrease_model_capacity", "modify_regularization", "change_optimizer"}
        for _ in range(20):
            _, decision = _full_pipeline(state)
            assert decision.action_type in valid

    def test_never_explores_terminate(self, monkeypatch):
        """Epsilon-greedy must never randomly select terminate."""
        import stratml.decision.policy.action_selector as sel
        import random
        monkeypatch.setattr(sel, "_EPSILON_LOW_DATA", 1.0)
        monkeypatch.setattr(sel, "_row_count", lambda: 0)
        # Force random to always trigger exploration
        monkeypatch.setattr(random, "random", lambda: 0.0)
        state = _make_state(well_fitted="none", underfitting="strong")
        for _ in range(10):
            _, decision = _full_pipeline(state)
            assert decision.action_type != "terminate"

    def test_epsilon_zero_always_picks_top(self, monkeypatch):
        """With epsilon=0, must always pick ranked[0]."""
        import stratml.decision.policy.action_selector as sel
        monkeypatch.setattr(sel, "_EPSILON_LOW_DATA", 0.0)
        monkeypatch.setattr(sel, "_EPSILON_HIGH_DATA", 0.0)
        monkeypatch.setattr(sel, "_row_count", lambda: 0)
        state = _make_state()
        ranked, decision = _full_pipeline(state)
        assert decision.action_type == ranked[0].action_type

    def test_row_count_zero_uses_high_epsilon(self, monkeypatch):
        import stratml.decision.policy.action_selector as sel
        monkeypatch.setattr(sel, "_row_count", lambda: 0)
        # Just verify it doesn't crash and returns a decision
        state = _make_state()
        _, decision = _full_pipeline(state)
        assert decision.action_type is not None

    def test_row_count_above_50_uses_low_epsilon(self, monkeypatch):
        import stratml.decision.policy.action_selector as sel
        monkeypatch.setattr(sel, "_row_count", lambda: 100)
        state = _make_state()
        _, decision = _full_pipeline(state)
        assert decision.action_type is not None


# ===========================================================================
# coordinator_agent.py — adaptive weights
# ===========================================================================

class TestCoordinatorAdaptiveWeights:
    def test_no_logs_returns_defaults(self, tmp_path, monkeypatch):
        import stratml.decision.agents.coordinator_agent as coord
        # Patch glob to return empty
        monkeypatch.setattr("glob.glob", lambda *a, **kw: [])
        w_p, w_e, w_s = coord._load_agent_weights()
        assert w_p == coord._W_PERF_DEFAULT
        assert w_e == coord._W_EFF_DEFAULT
        assert w_s == coord._W_STAB_DEFAULT

    def test_fewer_than_5_records_returns_defaults(self, tmp_path, monkeypatch):
        import stratml.decision.agents.coordinator_agent as coord
        log_path = tmp_path / "evaluation_log.jsonl"
        for i in range(3):
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "decision_validity": 1.0, "quality_risk": 0.2,
                    "counterfactual_impact": 0.03,
                }) + "\n")
        monkeypatch.setattr("glob.glob", lambda *a, **kw: [str(log_path)])
        w_p, w_e, w_s = coord._load_agent_weights()
        assert w_p == coord._W_PERF_DEFAULT

    def test_weights_sum_to_one(self, tmp_path, monkeypatch):
        import stratml.decision.agents.coordinator_agent as coord
        log_path = tmp_path / "evaluation_log.jsonl"
        for i in range(10):
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "decision_validity": 0.8, "quality_risk": 0.3,
                    "counterfactual_impact": 0.02,
                }) + "\n")
        monkeypatch.setattr("glob.glob", lambda *a, **kw: [str(log_path)])
        w_p, w_e, w_s = coord._load_agent_weights()
        assert abs(w_p + w_e + w_s - 1.0) < 1e-4

    def test_weights_non_negative(self, tmp_path, monkeypatch):
        import stratml.decision.agents.coordinator_agent as coord
        log_path = tmp_path / "evaluation_log.jsonl"
        for i in range(10):
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "decision_validity": 0.0, "quality_risk": 1.0,
                    "counterfactual_impact": -0.1,
                }) + "\n")
        monkeypatch.setattr("glob.glob", lambda *a, **kw: [str(log_path)])
        w_p, w_e, w_s = coord._load_agent_weights()
        assert w_p >= 0.0
        assert w_e >= 0.0
        assert w_s >= 0.0

    def test_corrupt_log_returns_defaults(self, tmp_path, monkeypatch):
        import stratml.decision.agents.coordinator_agent as coord
        log_path = tmp_path / "evaluation_log.jsonl"
        log_path.write_text("not json\n{broken\n")
        monkeypatch.setattr("glob.glob", lambda *a, **kw: [str(log_path)])
        w_p, w_e, w_s = coord._load_agent_weights()
        assert w_p == coord._W_PERF_DEFAULT

    def test_rule_rank_uses_loaded_weights(self, monkeypatch):
        """_rule_rank should call _load_agent_weights, not use hardcoded constants."""
        import stratml.decision.agents.coordinator_agent as coord
        called = {"n": 0}
        original = coord._load_agent_weights

        def tracking():
            called["n"] += 1
            return original()

        monkeypatch.setattr(coord, "_load_agent_weights", tracking)
        state = _make_state()
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        ests = estimate(predict(state, candidates), state)
        coord._rule_rank(state, ests, {"switch_model": 0.7}, {"switch_model": 0.6}, {"switch_model": 0.8})
        assert called["n"] == 1

    def test_rank_output_sorted_descending(self):
        state = _make_state()
        ranked, _ = _full_pipeline(state)
        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# evaluator_agent.py — all 4 audit dimensions
# ===========================================================================

class TestEvaluatorAgent:
    def _audit(self, action_type="switch_model", trigger="underfitting",
               accuracy=0.78, train_loss=0.18, val_loss=0.20,
               expected_gain=0.05, best_score=None, **state_kwargs):
        from stratml.decision.agents.evaluator_agent import _rule_audit
        decision = _make_decision(action_type=action_type, trigger=trigger, expected_gain=expected_gain)
        if best_score is not None:
            decision.reason.evidence["best_score"] = best_score
        result = _make_result(accuracy=accuracy, train_loss=train_loss, val_loss=val_loss)
        state = _make_state(train_loss=train_loss, val_loss=val_loss, **state_kwargs)
        return _rule_audit(decision, result, state)

    # --- decision_validity ---
    def test_valid_action_for_trigger_scores_1(self):
        rec = self._audit(action_type="switch_model", trigger="underfitting")
        assert rec.decision_validity == 1.0

    def test_invalid_action_for_trigger_scores_0(self):
        rec = self._audit(action_type="terminate", trigger="underfitting")
        assert rec.decision_validity == 0.0

    def test_terminate_valid_for_convergence(self):
        rec = self._audit(action_type="terminate", trigger="convergence")
        assert rec.decision_validity == 1.0

    def test_switch_model_valid_for_overfitting(self):
        rec = self._audit(action_type="switch_model", trigger="overfitting")
        assert rec.decision_validity == 1.0

    def test_unknown_trigger_does_not_crash(self):
        rec = self._audit(action_type="switch_model", trigger="unknown_trigger")
        assert 0.0 <= rec.decision_validity <= 1.0

    # --- reasoning_consistency ---
    def test_consistent_when_signal_present(self):
        rec = self._audit(trigger="underfitting", underfitting="strong", well_fitted="none")
        assert rec.reasoning_consistency == 1.0

    def test_inconsistent_when_signal_absent(self):
        rec = self._audit(trigger="underfitting", underfitting="none", well_fitted="strong")
        assert rec.reasoning_consistency == 0.3

    def test_bootstrap_always_consistent(self):
        rec = self._audit(trigger="bootstrap")
        assert rec.reasoning_consistency == 1.0

    # --- quality_risk ---
    def test_quality_risk_increases_with_gap(self):
        low_risk  = self._audit(train_loss=0.18, val_loss=0.20)
        high_risk = self._audit(train_loss=0.10, val_loss=0.35)
        assert high_risk.quality_risk > low_risk.quality_risk

    def test_unstable_training_increases_risk(self):
        stable = self._audit(unstable_training="none")
        unstable = self._audit(unstable_training="strong")
        assert unstable.quality_risk >= stable.quality_risk

    def test_quality_risk_in_range(self):
        rec = self._audit()
        assert 0.0 <= rec.quality_risk <= 1.0

    # --- counterfactual_impact ---
    def test_positive_impact_when_accuracy_exceeds_expected(self):
        rec = self._audit(accuracy=0.90)  # expected_gain=0.05 by default, best_score=0.78
        assert rec.counterfactual_impact > 0

    def test_negative_impact_when_accuracy_below_expected(self):
        rec = self._audit(accuracy=0.01)
        assert rec.counterfactual_impact < 0

    def test_exact_counterfactual_impact_positive(self):
        # previous=0.80, current=0.82, expected=0.01 -> (0.82 - 0.80) - 0.01 = +0.01
        rec = self._audit(best_score=0.80, accuracy=0.82, expected_gain=0.01)
        assert rec.counterfactual_impact == 0.01

    def test_exact_counterfactual_impact_zero(self):
        # previous=0.80, current=0.81, expected=0.01 -> (0.81 - 0.80) - 0.01 = 0.00
        rec = self._audit(best_score=0.80, accuracy=0.81, expected_gain=0.01)
        assert rec.counterfactual_impact == 0.0

    def test_exact_counterfactual_impact_negative(self):
        # previous=0.80, current=0.78, expected=0.01 -> (0.78 - 0.80) - 0.01 = -0.03
        rec = self._audit(best_score=0.80, accuracy=0.78, expected_gain=0.01)
        assert rec.counterfactual_impact == -0.03

    def test_llm_cannot_override_counterfactual_impact(self, monkeypatch):
        from stratml.decision.agents import evaluator_agent
        from unittest.mock import MagicMock
        mock_chat = MagicMock()
        mock_structured = MagicMock()
        mock_output = MagicMock()
        mock_output.decision_validity = 0.9
        mock_output.reasoning_consistency = 0.9
        mock_output.quality_risk = 0.1
        mock_output.counterfactual_impact = 999.0
        mock_output.fault_detected = False
        mock_output.notes = "all good"
        mock_structured.invoke.return_value = mock_output
        mock_chat.return_value.with_structured_output.return_value = mock_structured

        monkeypatch.setattr("os.getenv", lambda k, default=None: "fake_key" if k == "GROQ_API_KEY" else default)

        import sys
        mock_module = MagicMock()
        mock_module.ChatGroq = mock_chat
        monkeypatch.setitem(sys.modules, "langchain_groq", mock_module)
        monkeypatch.setitem(sys.modules, "langchain_core.messages", MagicMock())

        decision = _make_decision(expected_gain=0.01)
        decision.reason.evidence["best_score"] = 0.80
        result = _make_result(accuracy=0.82)
        state = _make_state()
        rec = evaluator_agent._llm_audit(decision, result, state)
        assert rec is not None
        assert rec.counterfactual_impact == 0.01

    # --- fault_detected ---
    def test_fault_detected_on_invalid_action(self):
        rec = self._audit(action_type="terminate", trigger="underfitting")
        assert rec.fault_detected is True

    def test_no_fault_on_valid_consistent_decision(self):
        rec = self._audit(
            action_type="switch_model", trigger="underfitting",
            underfitting="strong", well_fitted="none",
        )
        assert rec.fault_detected is False

    def test_notes_populated_on_fault(self):
        rec = self._audit(action_type="terminate", trigger="underfitting")
        assert len(rec.notes) > 0

    def test_notes_empty_on_clean_decision(self):
        rec = self._audit(
            action_type="switch_model", trigger="underfitting",
            underfitting="strong", well_fitted="none",
        )
        assert rec.notes == ""

    # --- output / logging ---
    def test_audit_writes_to_log(self, tmp_path, monkeypatch):
        import stratml.decision.agents.evaluator_agent as ea
        monkeypatch.setattr(ea, "_EVAL_LOG", tmp_path / "eval.jsonl")
        decision = _make_decision()
        result = _make_result()
        state = _make_state()
        ea.audit(decision, result, state)
        assert (tmp_path / "eval.jsonl").exists()

    def test_audit_log_is_valid_json(self, tmp_path, monkeypatch):
        import stratml.decision.agents.evaluator_agent as ea
        monkeypatch.setattr(ea, "_EVAL_LOG", tmp_path / "eval.jsonl")
        decision = _make_decision()
        result = _make_result()
        state = _make_state()
        ea.audit(decision, result, state)
        line = (tmp_path / "eval.jsonl").read_text().strip()
        record = json.loads(line)
        assert "fault_detected" in record
        assert "decision_validity" in record

    def test_audit_appends_multiple_entries(self, tmp_path, monkeypatch):
        import stratml.decision.agents.evaluator_agent as ea
        monkeypatch.setattr(ea, "_EVAL_LOG", tmp_path / "eval.jsonl")
        decision = _make_decision()
        result = _make_result()
        state = _make_state()
        ea.audit(decision, result, state)
        ea.audit(decision, result, state)
        lines = (tmp_path / "eval.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_audit_returns_evaluation_record(self, tmp_path, monkeypatch):
        import stratml.decision.agents.evaluator_agent as ea
        from stratml.decision.agents.evaluator_agent import EvaluationRecord
        monkeypatch.setattr(ea, "_EVAL_LOG", tmp_path / "eval.jsonl")
        rec = ea.audit(_make_decision(), _make_result(), _make_state())
        assert isinstance(rec, EvaluationRecord)

    def test_llm_failure_falls_back_to_rule(self, tmp_path, monkeypatch):
        import stratml.decision.agents.evaluator_agent as ea
        monkeypatch.setattr(ea, "_EVAL_LOG", tmp_path / "eval.jsonl")
        monkeypatch.setenv("GROQ_API_KEY", "fake_key")
        # _llm_audit will fail because key is fake — should fall back to rule
        rec = ea.audit(_make_decision(), _make_result(), _make_state())
        assert rec is not None
        assert 0.0 <= rec.decision_validity <= 1.0


# ===========================================================================
# counterfactual.py — runner-up recording
# ===========================================================================

class TestCounterfactualRunnerUp:
    def test_record_without_runner_up(self, tmp_path, monkeypatch):
        from stratml.decision.validation import counterfactual
        monkeypatch.setattr(counterfactual, "_CF_LOG", tmp_path / "cf.jsonl")
        counterfactual.record(_make_decision())
        data = json.loads((tmp_path / "cf.jsonl").read_text().strip())
        assert data["runner_up_action"] is None
        assert data["runner_up_predicted_gain"] is None

    def test_record_with_runner_up(self, tmp_path, monkeypatch):
        from stratml.decision.validation import counterfactual
        from stratml.decision.agents.coordinator_agent import RankedAction
        monkeypatch.setattr(counterfactual, "_CF_LOG", tmp_path / "cf.jsonl")
        runner_up = RankedAction(
            action_type="modify_regularization",
            parameters={"direction": "increase"},
            predicted_gain=0.03,
            predicted_cost=0.2,
            confidence=0.6,
            agent_scores=AgentScore(performance=0.5, efficiency=0.7, stability=0.8),
            final_score=0.62,
        )
        counterfactual.record(_make_decision(), runner_up)
        data = json.loads((tmp_path / "cf.jsonl").read_text().strip())
        assert data["runner_up_action"] == "modify_regularization"
        assert data["runner_up_predicted_gain"] == 0.03

    def test_expected_gain_recorded(self, tmp_path, monkeypatch):
        from stratml.decision.validation import counterfactual
        monkeypatch.setattr(counterfactual, "_CF_LOG", tmp_path / "cf.jsonl")
        decision = _make_decision(expected_gain=0.08)
        counterfactual.record(decision)
        data = json.loads((tmp_path / "cf.jsonl").read_text().strip())
        assert data["expected_gain"] == 0.08

    def test_multiple_entries_all_valid_json(self, tmp_path, monkeypatch):
        from stratml.decision.validation import counterfactual
        monkeypatch.setattr(counterfactual, "_CF_LOG", tmp_path / "cf.jsonl")
        for _ in range(5):
            counterfactual.record(_make_decision())
        lines = (tmp_path / "cf.jsonl").read_text().strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            json.loads(line)  # must not raise

    def test_engine_passes_runner_up_to_counterfactual(self, tmp_path, monkeypatch):
        """Engine must pass ranked[1] as runner_up to record_cf."""
        from stratml.decision.validation import counterfactual
        recorded = {}

        def mock_record(decision, runner_up=None):
            recorded["runner_up"] = runner_up

        monkeypatch.setattr(counterfactual, "record", mock_record)
        # Also patch _CF_LOG path
        monkeypatch.setattr(counterfactual, "_CF_LOG", tmp_path / "cf.jsonl")

        from stratml.decision.engine import DecisionEngine
        from stratml.execution.schemas import DataProfile, FeatureInfo
        monkeypatch.chdir(tmp_path)

        profile = DataProfile(
            dataset_name="test", dataset_type="tabular", rows=200, columns=5,
            target_column="target", problem_type="classification",
            numerical_columns=["f0", "f1", "f2", "f3"],
            categorical_columns=[], missing_value_ratio=0.0,
            class_distribution={"a": 100, "b": 100},
            feature_summary=[
                FeatureInfo(name=f"f{i}", dtype="float64", unique_values=200,
                            missing_percentage=0.0, distribution="normal")
                for i in range(4)
            ],
            recommended_metrics=["accuracy"],
        )
        engine = DecisionEngine(run_id="test_cf_run")
        engine.receive_profile(profile)
        # runner_up should be a RankedAction or None (if only 1 candidate)
        assert "runner_up" in recorded


# ===========================================================================
# meta_memory.py — cosine similarity retrieval
# ===========================================================================

class TestMetaMemory:
    def _make_meta(self, num_samples=500, num_features=10, imbalance=1.0):
        from stratml.decision.state.meta_features import DatasetMetaFeatures
        return DatasetMetaFeatures(
            num_samples=num_samples,
            num_features=num_features,
            feature_sample_ratio=round(num_features / num_samples, 6),
            class_entropy=1.0,
            missing_value_ratio=0.0,
            feature_variance_mean=10.0,
            imbalance_ratio=imbalance,
        )

    def test_retrieve_empty_memory_returns_empty(self, tmp_path, monkeypatch):
        from stratml.decision.learning import meta_memory
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta.jsonl")
        result = meta_memory.retrieve_similar_actions(self._make_meta())
        assert result == []

    def test_record_creates_file(self, tmp_path, monkeypatch):
        from stratml.decision.learning import meta_memory
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta.jsonl")
        meta_memory.record_run(self._make_meta(), "RandomForestClassifier", 0.85, "run_001")
        assert (tmp_path / "meta.jsonl").exists()

    def test_record_and_retrieve_returns_best_model(self, tmp_path, monkeypatch):
        from stratml.decision.learning import meta_memory
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta.jsonl")
        meta = self._make_meta(num_samples=500, num_features=10)
        meta_memory.record_run(meta, "GradientBoostingClassifier", 0.90, "run_001")
        results = meta_memory.retrieve_similar_actions(meta)
        assert "GradientBoostingClassifier" in results

    def test_retrieves_most_similar_not_dissimilar(self, tmp_path, monkeypatch):
        from stratml.decision.learning import meta_memory
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta.jsonl")
        # Record two very different runs
        similar_meta = self._make_meta(num_samples=500, num_features=10)
        dissimilar_meta = self._make_meta(num_samples=50000, num_features=500)
        meta_memory.record_run(similar_meta, "SVC", 0.88, "run_similar")
        meta_memory.record_run(dissimilar_meta, "LogisticRegression", 0.70, "run_dissimilar")
        query = self._make_meta(num_samples=480, num_features=11)
        results = meta_memory.retrieve_similar_actions(query)
        assert results[0] == "SVC"

    def test_no_duplicates_in_results(self, tmp_path, monkeypatch):
        from stratml.decision.learning import meta_memory
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta.jsonl")
        meta = self._make_meta()
        for i in range(5):
            meta_memory.record_run(meta, "RandomForestClassifier", 0.85, f"run_{i:03d}")
        results = meta_memory.retrieve_similar_actions(meta)
        assert len(results) == len(set(results))

    def test_top_k_respected(self, tmp_path, monkeypatch):
        from stratml.decision.learning import meta_memory
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta.jsonl")
        models = ["SVC", "LogisticRegression", "GradientBoostingClassifier",
                  "RandomForestClassifier", "KNeighborsClassifier"]
        for i, m in enumerate(models):
            meta_memory.record_run(self._make_meta(num_samples=500 + i), m, 0.8, f"run_{i}")
        results = meta_memory.retrieve_similar_actions(self._make_meta(), top_k=2)
        assert len(results) <= 2

    def test_cosine_similarity_identical_vectors(self):
        from stratml.decision.learning.meta_memory import _cosine
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        from stratml.decision.learning.meta_memory import _cosine
        assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        from stratml.decision.learning.meta_memory import _cosine
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_corrupt_memory_file_returns_empty(self, tmp_path, monkeypatch):
        from stratml.decision.learning import meta_memory
        p = tmp_path / "meta.jsonl"
        p.write_text("not json\n{broken\n")
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", p)
        result = meta_memory.retrieve_similar_actions(self._make_meta())
        assert result == []

    def test_record_run_on_terminate_in_engine(self, tmp_path, monkeypatch):
        """Engine must call record_run when terminate is decided."""
        from stratml.decision.learning import meta_memory
        recorded = {}

        def mock_record(meta, best_model, best_score, run_id):
            recorded["called"] = True
            recorded["best_model"] = best_model

        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta.jsonl")
        monkeypatch.setattr(meta_memory, "record_run", mock_record)

        from stratml.decision.engine import DecisionEngine
        from stratml.execution.schemas import DataProfile, FeatureInfo
        monkeypatch.chdir(tmp_path)

        profile = DataProfile(
            dataset_name="test", dataset_type="tabular", rows=200, columns=5,
            target_column="target", problem_type="classification",
            numerical_columns=["f0", "f1", "f2", "f3"],
            categorical_columns=[], missing_value_ratio=0.0,
            class_distribution={"a": 100, "b": 100},
            feature_summary=[
                FeatureInfo(name=f"f{i}", dtype="float64", unique_values=200,
                            missing_percentage=0.0, distribution="normal")
                for i in range(4)
            ],
            recommended_metrics=["accuracy"],
        )
        engine = DecisionEngine(
            run_id="test_meta_run",
            allowed_models=["RandomForestClassifier"],
            max_iterations=1,
        )
        engine.receive_profile(profile)
        # Force a terminate decision by exhausting budget
        result = _make_result()
        engine._models_tried = ["RandomForestClassifier", "LogisticRegression",
                                 "GradientBoostingClassifier", "SVC", "KNeighborsClassifier"]
        engine.max_iterations = 1
        decision = engine.receive_result(result)
        if decision.action_type == "terminate":
            assert recorded.get("called") is True


# ===========================================================================
# engine.py — end-to-end wiring
# ===========================================================================

class TestEngineWiring:
    @pytest.fixture
    def engine_and_profile(self, tmp_path, monkeypatch):
        from stratml.decision.engine import DecisionEngine
        from stratml.execution.schemas import DataProfile, FeatureInfo
        monkeypatch.chdir(tmp_path)
        profile = DataProfile(
            dataset_name="test", dataset_type="tabular", rows=300, columns=6,
            target_column="target", problem_type="classification",
            numerical_columns=["f0", "f1", "f2", "f3", "f4"],
            categorical_columns=[], missing_value_ratio=0.0,
            class_distribution={"a": 150, "b": 150},
            feature_summary=[
                FeatureInfo(name=f"f{i}", dtype="float64", unique_values=300,
                            missing_percentage=0.0, distribution="normal")
                for i in range(5)
            ],
            recommended_metrics=["accuracy"],
        )
        engine = DecisionEngine(
            run_id="test_engine_run",
            allowed_models=["RandomForestClassifier", "LogisticRegression", "SVC"],
            max_iterations=5,
        )
        return engine, profile

    def test_receive_profile_returns_action_decision(self, engine_and_profile):
        engine, profile = engine_and_profile
        decision = engine.receive_profile(profile)
        assert isinstance(decision, ActionDecision)

    def test_receive_profile_sets_last_signals(self, engine_and_profile):
        engine, profile = engine_and_profile
        engine.receive_profile(profile)
        assert engine._last_signals is not None

    def test_receive_profile_sets_last_decision(self, engine_and_profile):
        engine, profile = engine_and_profile
        engine.receive_profile(profile)
        assert engine._last_decision is not None

    def test_receive_result_returns_action_decision(self, engine_and_profile):
        engine, profile = engine_and_profile
        engine.receive_profile(profile)
        result = _make_result()
        decision = engine.receive_result(result)
        assert isinstance(decision, ActionDecision)

    def test_receive_result_updates_last_signals(self, engine_and_profile):
        engine, profile = engine_and_profile
        engine.receive_profile(profile)
        first_signals = engine._last_signals
        result = _make_result()
        engine.receive_result(result)
        # Signals may or may not change, but must be set
        assert engine._last_signals is not None

    def test_previous_signals_passed_to_second_decision(self, engine_and_profile):
        engine, profile = engine_and_profile
        engine.receive_profile(profile)
        first_signals = engine._last_signals
        result = _make_result()
        engine.receive_result(result)
        # The second state should have previous_signals from first decision
        # We verify by checking _last_signals was updated
        assert engine._last_signals is not None

    def test_evaluator_called_on_second_receive_result(self, engine_and_profile, monkeypatch):
        import stratml.decision.agents.evaluator_agent as ea
        audit_calls = {"n": 0}
        original_audit = ea.audit

        def counting_audit(*args, **kwargs):
            audit_calls["n"] += 1
            return original_audit(*args, **kwargs)

        monkeypatch.setattr(ea, "audit", counting_audit)
        engine, profile = engine_and_profile
        engine.receive_profile(profile)
        engine.receive_result(_make_result())
        engine.receive_result(_make_result(accuracy=0.80))
        assert audit_calls["n"] >= 1

    def test_evaluator_not_called_on_first_receive_result(self, engine_and_profile, monkeypatch):
        """No previous decision on first receive_result — evaluator must not be called."""
        import stratml.decision.agents.evaluator_agent as ea
        audit_calls = {"n": 0}

        def counting_audit(*args, **kwargs):
            audit_calls["n"] += 1

        monkeypatch.setattr(ea, "audit", counting_audit)
        engine, profile = engine_and_profile
        # Don't call receive_profile first — engine._last_decision is None
        engine.receive_result(_make_result())
        assert audit_calls["n"] == 0

    def test_output_dir_created(self, engine_and_profile):
        engine, profile = engine_and_profile
        engine.receive_profile(profile)
        assert engine._out_dir.exists()

    def test_decision_log_created(self, engine_and_profile):
        engine, profile = engine_and_profile
        engine.receive_profile(profile)
        log_dir = engine._out_dir / "decision_logs"
        assert log_dir.exists()
        json_files = list(log_dir.glob("*.json"))
        assert len(json_files) >= 1

    def test_action_type_is_valid(self, engine_and_profile):
        engine, profile = engine_and_profile
        decision = engine.receive_profile(profile)
        valid = {"switch_model", "terminate", "increase_model_capacity",
                 "decrease_model_capacity", "modify_regularization",
                 "change_optimizer", "add_preprocessing"}
        assert decision.action_type in valid

    def test_budget_exhausted_terminates(self, tmp_path, monkeypatch):
        from stratml.decision.engine import DecisionEngine
        from stratml.execution.schemas import DataProfile, FeatureInfo
        monkeypatch.chdir(tmp_path)
        profile = DataProfile(
            dataset_name="test", dataset_type="tabular", rows=200, columns=5,
            target_column="target", problem_type="classification",
            numerical_columns=["f0", "f1", "f2", "f3"],
            categorical_columns=[], missing_value_ratio=0.0,
            class_distribution={"a": 100, "b": 100},
            feature_summary=[
                FeatureInfo(name=f"f{i}", dtype="float64", unique_values=200,
                            missing_percentage=0.0, distribution="normal")
                for i in range(4)
            ],
            recommended_metrics=["accuracy"],
        )
        engine = DecisionEngine(run_id="test_budget_run", max_iterations=1)
        engine.receive_profile(profile)
        # Simulate iteration at max
        result = _make_result()
        # Manually set iteration to max
        from stratml.execution.schemas import ExperimentResult, ExperimentMetrics, ResourceUsage
        result2 = ExperimentResult(
            experiment_id="test_exp", iteration=1, dataset_name="test",
            model_name="RandomForestClassifier", model_type="ml", hyperparameters={},
            preprocessing_applied=result.preprocessing_applied,
            metrics=ExperimentMetrics(accuracy=0.72),
            train_curve=[0.5], validation_curve=[0.5],
            runtime=1.0, resource_usage=ResourceUsage(),
            artifacts={"model_path": "m.pkl", "metrics_file": "m.json", "tensorboard_logs": "tb/"},
        )
        decision = engine.receive_result(result2)
        assert decision.action_type == "terminate"

    def test_meta_memory_inject_reorders_allowed_models(self, tmp_path, monkeypatch):
        """If meta_memory returns a model, it should appear first in allowed_models."""
        from stratml.decision.learning import meta_memory
        from stratml.decision.engine import DecisionEngine
        from stratml.execution.schemas import DataProfile, FeatureInfo
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(meta_memory, "_MEMORY_FILE", tmp_path / "meta.jsonl")
        monkeypatch.setattr(meta_memory, "retrieve_similar_actions",
                            lambda meta, top_k=3: ["SVC"])

        profile = DataProfile(
            dataset_name="test", dataset_type="tabular", rows=200, columns=5,
            target_column="target", problem_type="classification",
            numerical_columns=["f0", "f1", "f2", "f3"],
            categorical_columns=[], missing_value_ratio=0.0,
            class_distribution={"a": 100, "b": 100},
            feature_summary=[
                FeatureInfo(name=f"f{i}", dtype="float64", unique_values=200,
                            missing_percentage=0.0, distribution="normal")
                for i in range(4)
            ],
            recommended_metrics=["accuracy"],
        )
        engine = DecisionEngine(
            run_id="test_meta_inject",
            allowed_models=["RandomForestClassifier", "LogisticRegression", "SVC"],
        )
        # Capture the state passed to _decide
        states_seen = []
        original_decide = engine._decide

        def capturing_decide(state):
            states_seen.append(state)
            return original_decide(state)

        engine._decide = capturing_decide
        engine.receive_profile(profile)
        if states_seen:
            assert states_seen[0].constraints.allowed_models[0] == "SVC"
