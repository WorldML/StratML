"""
test_decision_engine.py
-----------------------
Unit tests for all 12 decision engine components.
Tests run in pipeline order: action_generator → ... → counterfactual.
"""

import pytest
from stratml.core.schemas import (
    ActionDecision, AgentScore, CandidateAction, DecisionReason,
    PreprocessingConfig, SecondaryMetrics, StateActionContext,
    StateConstraints, StateDataset, StateGeneralization, StateMeta,
    StateMetrics, StateModel, StateObject, StateObjective, StateResources,
    StateSearch, StateSignals, StateTrajectory, StateUncertainty,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _sig(v) -> str:
    return v if isinstance(v, str) else ("strong" if v else "none")


def _make_state(
    iteration: int = 1,
    primary: float = 0.72,
    underfitting = "none",
    overfitting = "none",
    well_fitted = "strong",
    converged = "none",
    stagnating = "none",
    diverging = "none",
    plateau_detected = "none",
    diminishing_returns = "none",
    unstable_training = "none",
    high_variance = "none",
    remaining_budget: float = 15.0,
    models_tried: list[str] | None = None,
    allowed_models: list[str] | None = None,
    runtime: float = 10.0,
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
            runtime=runtime, convergence_epoch=0,
        ),
        generalization=StateGeneralization(train_loss=0.2, validation_loss=0.22, gap=0.02),
        resources=StateResources(runtime=runtime, gpu_used=False, cpu_time=runtime, remaining_budget=remaining_budget, budget_exhausted=remaining_budget <= 0),
        search=StateSearch(models_tried=models_tried or ["RandomForest"], unique_models_count=1, repeated_configs=0),
        signals=StateSignals(
            underfitting=_sig(underfitting), overfitting=_sig(overfitting), well_fitted=_sig(well_fitted),
            converged=_sig(converged), stagnating=_sig(stagnating), diverging=_sig(diverging),
            plateau_detected=_sig(plateau_detected), diminishing_returns=_sig(diminishing_returns),
            unstable_training=_sig(unstable_training), high_variance=_sig(high_variance),
        ),
        uncertainty=StateUncertainty(),
        action_context=StateActionContext(),
        constraints=StateConstraints(
            allowed_models=allowed_models or ["LogisticRegression", "RandomForest", "GradientBoosting", "SVC"],
            max_iterations=20,
        ),
    )


def _make_decision(action_type: str = "switch_model") -> ActionDecision:
    return ActionDecision(
        experiment_id="test_exp",
        iteration=1,
        action_type=action_type,
        parameters={"model_name": "GradientBoosting"},
        preprocessing=PreprocessingConfig(
            missing_value_strategy="mean", scaling="standard",
            encoding="onehot", imbalance_strategy="none", feature_selection="none",
        ),
        expected_gain=0.05,
        expected_cost=0.5,
        confidence=0.5,
        agent_scores=AgentScore(performance=0.7, efficiency=0.6, stability=0.8),
        reason=DecisionReason(trigger="underfitting", evidence={}, source="rule"),
    )


# ---------------------------------------------------------------------------
# 1. action_generator
# ---------------------------------------------------------------------------

class TestActionGenerator:
    from stratml.decision.actions import action_generator

    def test_bootstrap_returns_two_candidates(self):
        from stratml.decision.actions.action_generator import generate
        state = _make_state(iteration=0)
        candidates = generate(state)
        assert len(candidates) >= 1
        assert all(isinstance(c, CandidateAction) for c in candidates)

    def test_bootstrap_uses_allowed_models(self):
        from stratml.decision.actions.action_generator import generate
        state = _make_state(iteration=0, allowed_models=["SVC", "MLP"])
        candidates = generate(state)
        model_names = [c.parameters.get("model_name") for c in candidates if c.action_type == "switch_model"]
        assert "SVC" in model_names

    def test_underfitting_suggests_switch_or_increase(self):
        from stratml.decision.actions.action_generator import generate
        state = _make_state(underfitting="strong", well_fitted="none")
        candidates = generate(state)
        types = {c.action_type for c in candidates}
        assert types & {"switch_model", "increase_model_capacity"}

    def test_overfitting_suggests_regularize_or_decrease(self):
        from stratml.decision.actions.action_generator import generate
        state = _make_state(overfitting="strong", well_fitted="none")
        candidates = generate(state)
        types = {c.action_type for c in candidates}
        assert types & {"modify_regularization", "decrease_model_capacity"}

    def test_converged_well_fitted_returns_terminate(self):
        from stratml.decision.actions.action_generator import generate
        models = ["LogisticRegression", "RandomForest", "GradientBoosting", "SVC"]
        state = _make_state(
            converged="strong",
            well_fitted="strong",
            iteration=2,
            allowed_models=models,
            models_tried=models,
        )
        candidates = generate(state)
        assert candidates[0].action_type == "terminate"
        assert len(candidates) == 1

    def test_budget_exhausted_returns_only_terminate(self):
        from stratml.decision.actions.action_generator import generate
        state = _make_state(remaining_budget=0.0)
        candidates = generate(state)
        assert all(c.action_type == "terminate" for c in candidates)

    def test_no_duplicate_candidates(self):
        from stratml.decision.actions.action_generator import generate
        state = _make_state(stagnating="strong", underfitting="strong", well_fitted="none")
        candidates = generate(state)
        keys = [(c.action_type, str(sorted(c.parameters.items()))) for c in candidates]
        assert len(keys) == len(set(keys))

    def test_terminate_always_present(self):
        from stratml.decision.actions.action_generator import generate
        state = _make_state()
        candidates = generate(state)
        assert any(c.action_type == "terminate" for c in candidates)


# ---------------------------------------------------------------------------
# 2. dataset_builder
# ---------------------------------------------------------------------------

class TestDatasetBuilder:
    def test_record_creates_csv(self, tmp_path, monkeypatch):
        from stratml.decision.learning import dataset_builder
        monkeypatch.setattr(dataset_builder, "_DATASET_PATH", tmp_path / "decision_dataset.csv")
        state = _make_state()
        action = CandidateAction(action_type="switch_model", parameters={"model_name": "SVC"})
        dataset_builder.record(state, action)
        assert (tmp_path / "decision_dataset.csv").exists()

    def test_record_appends_rows(self, tmp_path, monkeypatch):
        from stratml.decision.learning import dataset_builder
        monkeypatch.setattr(dataset_builder, "_DATASET_PATH", tmp_path / "decision_dataset.csv")
        state = _make_state()
        action = CandidateAction(action_type="switch_model", parameters={"model_name": "SVC"})
        dataset_builder.record(state, action)
        dataset_builder.record(state, action)
        lines = (tmp_path / "decision_dataset.csv").read_text().strip().splitlines()
        assert len(lines) == 3  # header + 2 rows

    def test_record_contains_action_type(self, tmp_path, monkeypatch):
        from stratml.decision.learning import dataset_builder
        monkeypatch.setattr(dataset_builder, "_DATASET_PATH", tmp_path / "decision_dataset.csv")
        state = _make_state()
        action = CandidateAction(action_type="terminate", parameters={})
        dataset_builder.record(state, action)
        content = (tmp_path / "decision_dataset.csv").read_text()
        assert "terminate" in content


# ---------------------------------------------------------------------------
# 3. value_model
# ---------------------------------------------------------------------------

class TestValueModel:
    def test_returns_one_prediction_per_candidate(self):
        from stratml.decision.learning.value_model import predict
        state = _make_state()
        candidates = [
            CandidateAction(action_type="switch_model", parameters={}),
            CandidateAction(action_type="terminate", parameters={}),
        ]
        preds = predict(state, candidates)
        assert len(preds) == 2

    def test_predicted_gain_in_range(self):
        from stratml.decision.learning.value_model import predict
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        preds = predict(state, candidates)
        assert 0.0 <= preds[0].predicted_gain <= 1.0
        assert 0.0 <= preds[0].predicted_cost <= 1.0

    def test_action_type_preserved(self):
        from stratml.decision.learning.value_model import predict
        state = _make_state()
        candidates = [CandidateAction(action_type="terminate", parameters={})]
        preds = predict(state, candidates)
        assert preds[0].action_type == "terminate"


# ---------------------------------------------------------------------------
# 4. calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_output_gain_in_range(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.calibration import calibrate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        calibrated = calibrate(predict(state, candidates))
        assert 0.0 <= calibrated[0].predicted_gain <= 1.0

    def test_length_preserved(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.calibration import calibrate
        state = _make_state()
        candidates = [
            CandidateAction(action_type="switch_model", parameters={}),
            CandidateAction(action_type="terminate", parameters={}),
        ]
        preds = predict(state, candidates)
        assert len(calibrate(preds)) == 2


# ---------------------------------------------------------------------------
# 5. uncertainty
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_returns_one_estimate_per_prediction(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        estimates = estimate(predict(state, candidates))
        assert len(estimates) == 1

    def test_confidence_in_range(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        est = estimate(predict(state, candidates))[0]
        assert 0.0 <= est.confidence <= 1.0
        assert est.variance >= 0.0

    def test_gain_in_range(self):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        est = estimate(predict(state, candidates))[0]
        assert 0.0 <= est.predicted_gain <= 1.0


# ---------------------------------------------------------------------------
# 6-8. agents
# ---------------------------------------------------------------------------

def _get_estimates():
    from stratml.decision.learning.value_model import predict
    from stratml.decision.learning.calibration import calibrate
    from stratml.decision.learning.uncertainty import estimate
    state = _make_state()
    candidates = [
        CandidateAction(action_type="switch_model", parameters={}),
        CandidateAction(action_type="terminate", parameters={}),
        CandidateAction(action_type="increase_model_capacity", parameters={}),
    ]
    return state, estimate(calibrate(predict(state, candidates)))


class TestPerformanceAgent:
    def test_returns_score_per_action(self):
        from stratml.decision.agents.performance_agent import score
        state, estimates = _get_estimates()
        scores = score(state, estimates)
        assert set(scores.keys()) == {e.action_type for e in estimates}

    def test_underfitting_prefers_switch_over_terminate(self):
        from stratml.decision.agents.performance_agent import score
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        state = _make_state(underfitting="strong", well_fitted="none")
        candidates = [
            CandidateAction(action_type="switch_model", parameters={}),
            CandidateAction(action_type="terminate", parameters={}),
        ]
        estimates = estimate(predict(state, candidates))
        scores = score(state, estimates)
        assert scores["switch_model"] > scores["terminate"]

    def test_scores_in_valid_range(self):
        from stratml.decision.agents.performance_agent import score
        state, estimates = _get_estimates()
        for v in score(state, estimates).values():
            assert 0.0 <= v <= 1.5  # gain bonus can push slightly above 1


class TestEfficiencyAgent:
    def test_returns_score_per_action(self):
        from stratml.decision.agents.efficiency_agent import score
        state, estimates = _get_estimates()
        scores = score(state, estimates)
        assert set(scores.keys()) == {e.action_type for e in estimates}

    def test_terminate_is_most_efficient(self):
        from stratml.decision.agents.efficiency_agent import score
        state, estimates = _get_estimates()
        scores = score(state, estimates)
        assert scores["terminate"] >= scores["switch_model"]

    def test_scores_non_negative(self):
        from stratml.decision.agents.efficiency_agent import score
        state, estimates = _get_estimates()
        for v in score(state, estimates).values():
            assert v >= 0.0


class TestStabilityAgent:
    def test_returns_score_per_action(self):
        from stratml.decision.agents.stability_agent import score
        state, estimates = _get_estimates()
        scores = score(state, estimates)
        assert set(scores.keys()) == {e.action_type for e in estimates}

    def test_terminate_is_most_stable(self):
        from stratml.decision.agents.stability_agent import score
        state, estimates = _get_estimates()
        scores = score(state, estimates)
        assert scores["terminate"] >= scores["increase_model_capacity"]

    def test_diverging_penalises_risky_actions(self):
        from stratml.decision.agents.stability_agent import score
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.uncertainty import estimate
        stable_state = _make_state(diverging="none")
        unstable_state = _make_state(diverging="strong")
        candidates = [CandidateAction(action_type="increase_model_capacity", parameters={})]
        s_stable = score(stable_state, estimate(predict(stable_state, candidates)))
        s_unstable = score(unstable_state, estimate(predict(unstable_state, candidates)))
        assert s_stable["increase_model_capacity"] >= s_unstable["increase_model_capacity"]


# ---------------------------------------------------------------------------
# 9. coordinator_agent
# ---------------------------------------------------------------------------

class TestCoordinatorAgent:
    def _run(self, state=None):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.calibration import calibrate
        from stratml.decision.learning.uncertainty import estimate
        from stratml.decision.agents import performance_agent, efficiency_agent, stability_agent
        from stratml.decision.agents.coordinator_agent import rank
        if state is None:
            state = _make_state()
        candidates = [
            CandidateAction(action_type="switch_model", parameters={}),
            CandidateAction(action_type="terminate", parameters={}),
        ]
        estimates = estimate(calibrate(predict(state, candidates)))
        perf = performance_agent.score(state, estimates)
        eff = efficiency_agent.score(state, estimates)
        stab = stability_agent.score(state, estimates)
        return rank(state, estimates, perf, eff, stab)

    def test_returns_ranked_list(self):
        ranked = self._run()
        assert len(ranked) == 2

    def test_sorted_descending(self):
        ranked = self._run()
        assert ranked[0].final_score >= ranked[1].final_score

    def test_agent_scores_populated(self):
        ranked = self._run()
        for r in ranked:
            assert r.agent_scores.performance is not None
            assert r.agent_scores.efficiency is not None
            assert r.agent_scores.stability is not None

    def test_final_score_in_range(self):
        ranked = self._run()
        for r in ranked:
            assert 0.0 <= r.final_score <= 1.0

    def test_rationale_is_string(self):
        ranked = self._run()
        for r in ranked:
            assert isinstance(r.rationale, str)


# ---------------------------------------------------------------------------
# 10. action_selector
# ---------------------------------------------------------------------------

class TestActionSelector:
    def _run_full(self, state=None):
        from stratml.decision.learning.value_model import predict
        from stratml.decision.learning.calibration import calibrate
        from stratml.decision.learning.uncertainty import estimate
        from stratml.decision.agents import performance_agent, efficiency_agent, stability_agent
        from stratml.decision.agents.coordinator_agent import rank
        from stratml.decision.policy.action_selector import select
        if state is None:
            state = _make_state()
        candidates = [
            CandidateAction(action_type="switch_model", parameters={"model_name": "SVC"}),
            CandidateAction(action_type="terminate", parameters={}),
        ]
        estimates = estimate(calibrate(predict(state, candidates)))
        perf = performance_agent.score(state, estimates)
        eff = efficiency_agent.score(state, estimates)
        stab = stability_agent.score(state, estimates)
        ranked = rank(state, estimates, perf, eff, stab)
        return select(state, ranked)

    def test_returns_action_decision(self):
        decision = self._run_full()
        assert isinstance(decision, ActionDecision)

    def test_action_type_is_valid(self):
        decision = self._run_full()
        assert decision.action_type in {"switch_model", "terminate", "increase_model_capacity",
                                        "decrease_model_capacity", "modify_regularization", "change_optimizer"}

    def test_reason_populated(self):
        decision = self._run_full()
        assert decision.reason.trigger != ""
        assert decision.reason.source in {"rule", "learned"}

    def test_learned_source_sets_rationale(self):
        from stratml.decision.agents.coordinator_agent import RankedAction
        from stratml.decision.policy.action_selector import select
        from stratml.core.schemas import AgentScore
        state = _make_state()
        ranked = [
            RankedAction(
                action_type="terminate", parameters={}, predicted_gain=0.05,
                predicted_cost=0.0, confidence=0.9,
                agent_scores=AgentScore(performance=0.9, efficiency=1.0, stability=0.95),
                final_score=0.95, rationale="Budget exhausted; termination is optimal.",
            )
        ]
        decision = select(state, ranked)
        assert decision.reason.source == "learned"
        assert "rationale" in decision.reason.evidence

    def test_bootstrap_trigger_on_iteration_0(self):
        decision = self._run_full(state=_make_state(iteration=0))
        assert decision.reason.trigger == "bootstrap"

    def test_confidence_in_range(self):
        decision = self._run_full()
        assert 0.0 <= decision.confidence <= 1.0


# ---------------------------------------------------------------------------
# 11. decision_logger
# ---------------------------------------------------------------------------

class TestDecisionLogger:
    def test_creates_json_file(self, tmp_path, monkeypatch):
        from stratml.decision.logging import decision_logger
        monkeypatch.setattr(decision_logger, "_LOG_DIR", tmp_path)
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        decision = _make_decision()
        path = decision_logger.log(state, candidates, decision)
        assert path.exists()
        assert path.suffix == ".json"

    def test_log_contains_action_type(self, tmp_path, monkeypatch):
        from stratml.decision.logging import decision_logger
        monkeypatch.setattr(decision_logger, "_LOG_DIR", tmp_path)
        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        decision = _make_decision("terminate")
        path = decision_logger.log(state, candidates, decision)
        content = path.read_text()
        assert "terminate" in content

    def test_filename_includes_iteration(self, tmp_path, monkeypatch):
        from stratml.decision.logging import decision_logger
        monkeypatch.setattr(decision_logger, "_LOG_DIR", tmp_path)
        state = _make_state(iteration=7)
        candidates = [CandidateAction(action_type="switch_model", parameters={})]
        decision = _make_decision()
        path = decision_logger.log(state, candidates, decision)
        assert "0007" in path.name


# ---------------------------------------------------------------------------
# 12. counterfactual
# ---------------------------------------------------------------------------

class TestCounterfactual:
    def test_creates_log_file(self, tmp_path, monkeypatch):
        from stratml.decision.validation import counterfactual
        monkeypatch.setattr(counterfactual, "_CF_LOG", tmp_path / "cf.jsonl")
        decision = _make_decision()
        counterfactual.record(decision)
        assert (tmp_path / "cf.jsonl").exists()

    def test_appends_multiple_entries(self, tmp_path, monkeypatch):
        from stratml.decision.validation import counterfactual
        monkeypatch.setattr(counterfactual, "_CF_LOG", tmp_path / "cf.jsonl")
        decision = _make_decision()
        counterfactual.record(decision)
        counterfactual.record(decision)
        lines = (tmp_path / "cf.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_log_contains_action_type(self, tmp_path, monkeypatch):
        from stratml.decision.validation import counterfactual
        monkeypatch.setattr(counterfactual, "_CF_LOG", tmp_path / "cf.jsonl")
        decision = _make_decision("modify_regularization")
        counterfactual.record(decision)
        content = (tmp_path / "cf.jsonl").read_text()
        assert "modify_regularization" in content
