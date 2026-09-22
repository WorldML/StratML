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

import json
import random
import uuid

from stratml.execution.schemas import DataProfile, ExperimentResult
from stratml.core.schemas import ActionDecision, CandidateAction

from stratml.decision.state.state_builder import build_state_from_profile, build_state
from stratml.decision.state.state_history import ExperimentHistory
from stratml.decision.actions.action_generator import generate
from stratml.decision.learning.dataset_builder import record as record_dataset, backfill_last_gain
from stratml.decision.learning.value_model import predict
from stratml.decision.learning.calibration import calibrate
from stratml.decision.learning.uncertainty import estimate
from stratml.decision.agents import performance_agent, efficiency_agent, stability_agent, coordinator_agent
from stratml.decision.agents.coordinator_agent import rank
from stratml.decision.policy.action_selector import select
from stratml.decision.logging import decision_logger
from stratml.decision.validation import counterfactual
from stratml.decision.agents import evaluator_agent
from stratml.decision.learning import dataset_builder
from stratml.decision.learning import meta_memory as _meta_memory
from stratml.decision.state.meta_features import extract as _extract_meta
from stratml.decision.learning import value_model as _value_model
from stratml.decision.llm_control import set_llm_mode, is_llm_enabled
import os


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
        seed: int = 42,
        history_mode: str = "independent",
        llm_mode: Optional[bool] = None,
        enable_meta_memory: bool = True,
        enable_value_model: bool = True,
    ) -> None:
        self.primary_metric    = primary_metric
        self.optimization_goal = optimization_goal
        self._metric_explicit  = primary_metric is not None
        self.allowed_models    = allowed_models
        self.max_iterations    = max_iterations
        self.time_budget       = time_budget
        self.dl_hyperparams    = dl_hyperparams or {}
        self.seed              = seed
        self._rng              = random.Random(seed)
        self.history_mode      = history_mode
        self.enable_meta_memory = enable_meta_memory
        self.enable_value_model = enable_value_model

        if not run_id:
            now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            salt = uuid.uuid4().hex[:6]
            self.run_id = f"run_{now_str}_{salt}"
        else:
            self.run_id = run_id

        self._history               = ExperimentHistory()
        self._profile               = None
        self._models_tried: list[str] = []
        self._repeated_configs: int   = 0
        self._last_action: Optional[str]   = None
        self._last_action_success: Optional[bool] = None
        self._last_best_score: Optional[float]    = None  # for observed_gain backfill
        self._dataset_fingerprint: Optional[str]  = None
        self._last_signals = None
        self._last_decision = None

        # Actual best validation model provenance
        self._best_model: Optional[str] = None
        self._best_val_score: Optional[float] = None

        # Redirect all outputs under outputs/<run_id>/
        self._out_dir = Path("outputs") / self.run_id
        self._out_dir.mkdir(parents=True, exist_ok=True)

        decision_logger._LOG_DIR      = self._out_dir / "decision_logs"
        counterfactual._CF_LOG        = self._out_dir / "decision_logs" / "counterfactual_log.jsonl"
        dataset_builder._DATASET_PATH = self._out_dir / "decision_logs" / "decision_dataset.csv"
        dataset_builder._UNIFIED_PATH = Path("runs/decision_logs/decision_dataset.csv")

        # Configure value model history mode and corpus path
        _value_model._HISTORY_MODE = self.history_mode
        if self.history_mode == "independent":
            _value_model._DATASET_PATH = self._out_dir / "decision_logs" / "decision_dataset.csv"
        else:
            _value_model._DATASET_PATH = Path("runs/decision_logs/decision_dataset.csv")

        # MetaMemory isolation
        if self.history_mode == "independent":
            _meta_memory._MEMORY_FILE = self._out_dir / "decision_logs" / "meta_memory.jsonl"
        elif "outputs" in str(_meta_memory._MEMORY_FILE):
            _meta_memory._MEMORY_FILE = Path("runs/decision_logs/meta_memory.jsonl")

        # Record resolved LLM configuration (P0-7 & P0-3)
        set_llm_mode(llm_mode)
        self.llm_mode = llm_mode
        llm_active = is_llm_enabled()
        self.llm_config = {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "llm_mode_configured": llm_mode,
            "llm_mode_enabled": llm_active,
            "temperature": {
                "action_generator": 0.2,
                "specialist_agents": 0.0,
                "coordinator_agent": 0.0,
                "evaluator_agent": 0.0,
            },
            "structured_output_schema": {
                "action_generator": "_CandidateList",
                "coordinator": "_CoordinatorOutput",
                "agents": "_Scores",
                "evaluator": "_AuditOutput",
            },
            "prompt_version": "v1_paper_frozen",
            "fallback_behavior": "rule_based",
        }
        art_dir = self._out_dir / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "llm_config.json").write_text(json.dumps(self.llm_config, indent=2), encoding="utf-8")

    def receive_profile(self, profile: DataProfile) -> ActionDecision:
        self._profile = profile
        self._dataset_fingerprint = getattr(profile, "dataset_fingerprint", None)
        if not self._metric_explicit or self.primary_metric is None:
            if profile.problem_type == "regression":
                self.primary_metric = "r2"
                self.optimization_goal = "maximize"
            else:
                self.primary_metric = "accuracy"
                self.optimization_goal = "maximize"

        meta = _extract_meta(profile)
        if self.enable_meta_memory:
            try:
                similar_models = _meta_memory.retrieve_similar_actions(
                    meta,
                    history_mode=self.history_mode,
                    current_run_id=self.run_id,
                    current_dataset_id=profile.dataset_name,
                    current_dataset_fingerprint=self._dataset_fingerprint,
                )
            except TypeError:
                similar_models = _meta_memory.retrieve_similar_actions(meta)
        else:
            similar_models = []
        allowed = self.allowed_models
        if similar_models and allowed:
            # Prioritise similar models by moving them to front
            reordered = [m for m in similar_models if m in allowed]
            rest = [m for m in allowed if m not in reordered]
            allowed = reordered + rest
            self.allowed_models = allowed
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
        current_score = getattr(result.metrics, metric_name, None)
        if current_score is None:
            current_score = (
                getattr(result.metrics, "r2", None)
                if metric_name == "r2"
                else getattr(result.metrics, "accuracy", 0.0)
            )
        current_score = float(current_score or 0.0)

        # Determine action success/failure for S_(t+1) (P0-27)
        is_failed = getattr(result, "failed", False) or getattr(result, "status", "") == "failed"
        if is_failed:
            action_success = False
            gain = (current_score - self._last_best_score) if self._last_best_score is not None else 0.0
        elif self._last_best_score is not None:
            gain = current_score - self._last_best_score
            if self.optimization_goal == "minimize":
                action_success = (gain <= 0.0)
            else:
                action_success = (gain >= 0.0)
        else:
            gain = 0.0
            action_success = True

        self._last_action_success = action_success

        if self._last_action is not None and self._last_decision is not None:
            backfill_last_gain(
                gain,
                run_id=self.run_id,
                experiment_id=self._last_decision.experiment_id,
                iteration=self._last_decision.iteration,
                dataset_id=self._profile.dataset_name if self._profile else None,
                dataset_fingerprint=self._dataset_fingerprint,
            )

            if self._best_val_score is None:
                self._best_val_score = current_score
                self._best_model = result.model_name
            elif self.optimization_goal == "maximize" and current_score > self._best_val_score:
                self._best_val_score = current_score
                self._best_model = result.model_name
            elif self.optimization_goal == "minimize" and current_score < self._best_val_score:
                self._best_val_score = current_score
                self._best_model = result.model_name

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
            eval_record = evaluator_agent.audit(self._last_decision, result, state)

            # Reconstructable trajectory (P0-25): Link execution result and evaluator result to previous decision record
            next_state_id = f"{result.experiment_id}_{result.iteration}"
            exec_summary = {
                "experiment_id": result.experiment_id,
                "iteration": result.iteration,
                "model_name": result.model_name,
                "metrics": result.metrics.model_dump(),
                "runtime": result.runtime,
                "action_success": self._last_action_success,
                "gain": gain,
            }
            eval_summary = {
                "decision_validity": getattr(eval_record, "decision_validity", None),
                "quality_risk": getattr(eval_record, "quality_risk", None),
                "counterfactual_impact": getattr(eval_record, "counterfactual_impact", None),
                "action_verdict": getattr(eval_record, "action_verdict", None),
                "rationale": getattr(eval_record, "rationale", None),
            } if eval_record else None

            decision_logger.update_outcome(
                experiment_id=self._last_decision.experiment_id,
                iteration=self._last_decision.iteration,
                execution_result=exec_summary,
                evaluator_result=eval_summary,
                next_state_id=next_state_id,
            )
        return self._decide(state)

    def _decide(self, state) -> ActionDecision:
        candidates: list[CandidateAction] = generate(state)

        # P0-21: Value Model ablation control
        if self.enable_value_model:
            predictions = predict(state, candidates, seed=self.seed)
            has_learned_model = False
            try:
                import pandas as pd
                if _value_model._DATASET_PATH.exists():
                    df = pd.read_csv(_value_model._DATASET_PATH)
                    has_learned_model = (df.get("observed_gain") is not None and df["observed_gain"].notna().sum() >= 50)
            except Exception:
                has_learned_model = False
        else:
            from stratml.decision.learning.value_model import ValuePrediction
            predictions = [
                ValuePrediction(
                    action_type=c.action_type,
                    parameters=c.parameters,
                    predicted_gain=0.05,
                    predicted_cost=0.5,
                )
                for c in candidates
            ]
            has_learned_model = False

        calibrated  = calibrate(predictions)
        estimates   = estimate(calibrated, state)

        perf_scores = performance_agent.score(state, estimates)
        eff_scores  = efficiency_agent.score(state, estimates)
        stab_scores = stability_agent.score(state, estimates)

        ranked = coordinator_agent.rank(state, estimates, perf_scores, eff_scores, stab_scores)

        # Determine decision source (P0-24)
        if is_llm_enabled():
            llm_decided = any(bool(r.rationale) for r in ranked)
            decision_source = "llm" if llm_decided else "fallback"
        else:
            decision_source = "hybrid" if (self.enable_value_model and has_learned_model) else "rule"

        decision = select(state, ranked, rng=self._rng, decision_source=decision_source)

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
            run_id=self.run_id,
            dataset_id=self._profile.dataset_name if self._profile else None,
            dataset_fingerprint=self._dataset_fingerprint,
            seed=self.seed,
        )

        # Coordinator weights used for this decision (P0-23)
        coordinator_weights = coordinator_agent.get_current_weights()
        coordinator_weights["iteration"] = state.meta.iteration

        # Ranked candidates information for runner-up traceability (P0-26)
        ranked_candidates = [
            {
                "candidate": {
                    "action_type": r.action_type,
                    "parameters": r.parameters,
                },
                "predicted_gain": r.predicted_gain,
                "predicted_cost": r.predicted_cost,
                "confidence": r.confidence,
                "agent_scores": {
                    "performance": r.agent_scores.performance,
                    "efficiency": r.agent_scores.efficiency,
                    "stability": r.agent_scores.stability,
                },
                "final_score": r.final_score,
                "rank": i + 1,
                "selected": (r.action_type == decision.action_type and r.parameters == decision.parameters),
                "rationale": r.rationale,
            }
            for i, r in enumerate(ranked)
        ]

        decision_logger.log(
            state,
            candidates,
            decision,
            coordinator_weights=coordinator_weights,
            ranked_candidates=ranked_candidates,
        )

        runner_up = ranked[1] if len(ranked) > 1 else None
        counterfactual.record(decision, runner_up)

        self._last_action         = decision.action_type
        self._last_best_score     = state.trajectory.best_score
        self._last_signals        = state.signals
        self._last_decision       = decision

        if decision.action_type == "terminate":
            # Backfill the terminate row immediately — no further result will arrive.
            backfill_last_gain(
                0.0,
                run_id=self.run_id,
                experiment_id=decision.experiment_id,
                iteration=decision.iteration,
                dataset_id=self._profile.dataset_name if self._profile else None,
                dataset_fingerprint=self._dataset_fingerprint,
            )
            if self.enable_meta_memory and self._profile is not None:
                meta = _extract_meta(self._profile)
                best_model = self._best_model or (state.search.models_tried[-1] if state.search.models_tried else "unknown")
                try:
                    _meta_memory.record_run(
                        meta,
                        best_model,
                        state.trajectory.best_score,
                        self.run_id,
                        dataset_id=self._profile.dataset_name,
                        dataset_fingerprint=self._dataset_fingerprint,
                    )
                except TypeError:
                    _meta_memory.record_run(
                        meta,
                        best_model,
                        state.trajectory.best_score,
                        self.run_id,
                    )
        return decision
