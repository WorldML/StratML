"""
engine.py
---------
Decision Engine — public entry point for the orchestrator.
All outputs (artifacts, logs, report, model) go to outputs/<run_id>/
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from stratml.execution.schemas import DataProfile, ExperimentResult
from stratml.core.schemas import ActionDecision, CandidateAction

from stratml.decision.state.state_builder import build_state_from_profile, build_state
from stratml.decision.state.state_history import ExperimentHistory
from stratml.decision.actions.action_generator import generate
from stratml.decision.learning.dataset_builder import record as record_dataset, backfill_last_gain
from stratml.decision.learning.value_model import predict
from stratml.decision.learning.calibration import calibrate
from stratml.decision.learning.uncertainty import estimate
from stratml.decision.agents import performance_agent, efficiency_agent, stability_agent
from stratml.decision.agents.coordinator_agent import rank
from stratml.decision.policy.action_selector import select
from stratml.decision.logging import decision_logger
from stratml.decision.validation import counterfactual
from stratml.decision.agents import evaluator_agent
from stratml.decision.learning import dataset_builder
from stratml.decision.learning import meta_memory as _meta_memory
from stratml.decision.state.meta_features import extract as _extract_meta
from stratml.decision.learning import value_model as _value_model


class DecisionEngine:
    def __init__(
        self,
        primary_metric: Optional[str] = None,
        optimization_goal: str = "maximize",
        allowed_models: Optional[list[str]] = None,
        max_iterations: int = 20,
        time_budget: Optional[float] = None,
        run_id: Optional[str] = None,
        dl_hyperparams: Optional[dict] = None,
    ) -> None:
        self.primary_metric    = primary_metric
        self.optimization_goal = optimization_goal
        self._metric_explicit  = primary_metric is not None
        self.allowed_models    = allowed_models
        self.max_iterations    = max_iterations
        self.time_budget       = time_budget
        self.dl_hyperparams    = dl_hyperparams or {}
        self.run_id            = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

        self._history               = ExperimentHistory()
        self._profile               = None
        self._models_tried: list[str] = []
        self._repeated_configs: int   = 0
        self._last_action: Optional[str]   = None
        self._last_action_success: Optional[bool] = None
        self._last_best_score: Optional[float]    = None  # for observed_gain backfill
        self._last_signals = None
        self._last_decision = None

        # Redirect all outputs under outputs/<run_id>/
        self._out_dir = Path("outputs") / self.run_id
        self._out_dir.mkdir(parents=True, exist_ok=True)

        decision_logger._LOG_DIR      = self._out_dir / "decision_logs"
        counterfactual._CF_LOG        = self._out_dir / "decision_logs" / "counterfactual_log.jsonl"
        dataset_builder._DATASET_PATH = self._out_dir / "decision_logs" / "decision_dataset.csv"
        dataset_builder._UNIFIED_PATH = Path("runs/decision_logs/decision_dataset.csv")
        # value_model and uncertainty read from unified path so 50-row threshold
        # counts across all runs, not just the current one
        _value_model._DATASET_PATH    = Path("runs/decision_logs/decision_dataset.csv")

    def receive_profile(self, profile: DataProfile) -> ActionDecision:
        self._profile = profile
        if not self._metric_explicit or self.primary_metric is None:
            if profile.problem_type == "regression":
                self.primary_metric = "r2"
                self.optimization_goal = "maximize"
            else:
                self.primary_metric = "accuracy"
                self.optimization_goal = "maximize"

        meta = _extract_meta(profile)
        similar_models = _meta_memory.retrieve_similar_actions(meta)
        allowed = self.allowed_models
        if similar_models and allowed:
            # Prioritise similar models by moving them to front
            reordered = [m for m in similar_models if m in allowed]
            rest = [m for m in allowed if m not in reordered]
            allowed = reordered + rest
        state = build_state_from_profile(
            profile,
            run_id=self.run_id,
            primary_metric=self.primary_metric,
            optimization_goal=self.optimization_goal,
            allowed_models=allowed,
            max_iterations=self.max_iterations,
            time_budget=self.time_budget,
        )
        return self._decide(state)

    def receive_result(self, result: ExperimentResult) -> ActionDecision:
        # Backfill observed_gain for the previous decision row
        metric_name = self.primary_metric or (
            "r2" if self._profile and self._profile.problem_type == "regression" else "accuracy"
        )
        if self._last_action is not None and self._last_best_score is not None:
            current_score = getattr(result.metrics, metric_name, None)
            if current_score is None:
                current_score = (
                    getattr(result.metrics, "r2", None)
                    if metric_name == "r2"
                    else getattr(result.metrics, "accuracy", 0.0)
                )
            current_score = current_score or 0.0
            gain = current_score - self._last_best_score
            backfill_last_gain(gain)

        if result.model_name not in self._models_tried:
            self._models_tried.append(result.model_name)
        else:
            self._repeated_configs += 1

        remaining = max(0.0, self.max_iterations - result.iteration)
        state = build_state(
            result,
            history=self._history,
            profile=self._profile,
            primary_metric=metric_name,
            optimization_goal=self.optimization_goal,
            allowed_models=self.allowed_models,
            max_iterations=self.max_iterations,
            time_budget=self.time_budget,
            previous_action=self._last_action,
            previous_action_success=self._last_action_success,
            models_tried=self._models_tried,
            repeated_configs=self._repeated_configs,
            remaining_budget=remaining,
            previous_signals=self._last_signals,
        )
        if self._last_decision is not None:
            eval_log = self._out_dir / "decision_logs" / "evaluation_log.jsonl"
            evaluator_agent._EVAL_LOG = eval_log
            evaluator_agent.audit(self._last_decision, result, state)
        return self._decide(state)

    def _decide(self, state) -> ActionDecision:
        candidates: list[CandidateAction] = generate(state)

        predictions = predict(state, candidates)
        calibrated  = calibrate(predictions)
        estimates   = estimate(calibrated, state)

        perf_scores = performance_agent.score(state, estimates)
        eff_scores  = efficiency_agent.score(state, estimates)
        stab_scores = stability_agent.score(state, estimates)

        ranked   = rank(state, estimates, perf_scores, eff_scores, stab_scores)
        decision = select(state, ranked)

        # Inject DL hyperparams when running in DL mode
        if self.dl_hyperparams and decision.action_type != "terminate":
            decision.parameters.update(self.dl_hyperparams)
            
        # Find predicted_gain for the selected action
        selected_pred = next((p for p in predictions if p.action_type == decision.action_type), None)
        predicted_gain = selected_pred.predicted_gain if selected_pred else 0.0

        record_dataset(
            state,
            CandidateAction(action_type=decision.action_type, parameters=decision.parameters),
            predicted_gain=predicted_gain,
        )
        decision_logger.log(state, candidates, decision)
        runner_up = ranked[1] if len(ranked) > 1 else None
        counterfactual.record(decision, runner_up)

        self._last_action         = decision.action_type
        self._last_action_success = None
        self._last_best_score     = state.trajectory.best_score
        self._last_signals        = state.signals
        self._last_decision       = decision
        if decision.action_type == "terminate":
            # Backfill the terminate row immediately — no further result will arrive.
            # gain=0.0 because termination produces no score change.
            backfill_last_gain(0.0)
            if self._profile is not None:
                meta = _extract_meta(self._profile)
                best_model = state.search.models_tried[-1] if state.search.models_tried else "unknown"
                _meta_memory.record_run(meta, best_model, state.trajectory.best_score, self.run_id)
        return decision
