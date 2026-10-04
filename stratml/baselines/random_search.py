"""
stratml/baselines/random_search.py
-----------------------------------
Configuration-level Random Search baseline for AutoML empirical evaluation.

Frozen protocol requirements:
1. Uses the exact same classical ML model space and parameter options as StratML.
2. Independent random candidate sampling per iteration without adaptive ranking,
   state reasoning, LLM deliberation, MetaMemory, or value modeling.
3. Fully deterministic given the random seed.
4. Integrates seamlessly with ExecutionOrchestrator and Phase 2 budget semantics.
"""

from __future__ import annotations

import copy
import json
import random
import uuid
from typing import Any, Optional

from stratml.core.schemas import (
    ActionDecision,
    DecisionReason,
    ExperimentResult,
    PreprocessingConfig,
)
from stratml.execution.pipelines.ml_pipeline import (
    PAPER_CLASSIFICATION_MODELS,
    PAPER_REGRESSION_MODELS,
)
from stratml.execution.schemas import DataProfile

RANDOM_SEARCH_CLASSIFICATION_MODELS: list[str] = list(PAPER_CLASSIFICATION_MODELS)
RANDOM_SEARCH_REGRESSION_MODELS: list[str] = list(PAPER_REGRESSION_MODELS)

# Shared classical-ML parameter spaces derived from paper mutation space and param grids
RANDOM_SEARCH_PARAM_SPACES: dict[str, dict[str, list[Any]]] = {
    # ── Classification Models (8) ─────────────────────────────────────────────
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
    },
    "GradientBoostingClassifier": {
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
    },
    "ExtraTreesClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    },
    "SVC": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 11],
    },
    "GaussianNB": {
        "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },

    # ── Regression Models (8) ─────────────────────────────────────────────────
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "GradientBoostingRegressor": {
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
    },
    "ExtraTreesRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "Lasso": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "ElasticNet": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.7],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 11],
    },
}


class RandomSearchSampler:
    """
    Deterministic, independent configuration sampler for classical ML.
    Samples uniformly across model families and discrete parameter options.
    """

    def __init__(
        self,
        seed: int = 42,
        param_spaces: Optional[dict[str, dict[str, list[Any]]]] = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.param_spaces = (
            copy.deepcopy(param_spaces)
            if param_spaces is not None
            else copy.deepcopy(RANDOM_SEARCH_PARAM_SPACES)
        )

    def sample_candidate(
        self,
        problem_type: str,
        allowed_models: Optional[list[str]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Sample a model family and hyperparameters uniformly at random.
        Sorting keys guarantees strict order-independent reproducibility.
        """
        if allowed_models:
            models = sorted(list(allowed_models))
        elif problem_type == "regression":
            models = sorted(list(RANDOM_SEARCH_REGRESSION_MODELS))
        else:
            models = sorted(list(RANDOM_SEARCH_CLASSIFICATION_MODELS))

        model_name = self.rng.choice(models)
        param_grid = self.param_spaces.get(model_name, {})
        hyperparameters: dict[str, Any] = {}
        for param in sorted(param_grid.keys()):
            choices = param_grid[param]
            hyperparameters[param] = self.rng.choice(choices)

        return model_name, hyperparameters


class RandomSearchEngine:
    """
    Baseline Decision Engine implementing pure configuration-level Random Search.
    Participates in the ExecutionOrchestrator protocol via receive_profile and receive_result.
    No state reasoning, no LLM, no MetaMemory, no value model, no adaptive ranking.
    """

    def __init__(
        self,
        evaluation_budget: int = 20,
        seed: int = 42,
        run_id: Optional[str] = None,
        time_budget: Optional[float] = None,
        allowed_models: Optional[list[str]] = None,
        param_spaces: Optional[dict[str, dict[str, list[Any]]]] = None,
    ):
        self.evaluation_budget = evaluation_budget
        self.seed = seed
        self.run_id = run_id or f"rs_{uuid.uuid4().hex[:8]}"
        self.time_budget = time_budget
        self.allowed_models = allowed_models
        self.sampler = RandomSearchSampler(seed=seed, param_spaces=param_spaces)

        self._iteration = 0
        self._profile: Optional[DataProfile] = None
        self._results_history: list[ExperimentResult] = []
        self._seen_configs: set[str] = set()
        self.repeated_configs: int = 0
        self.trajectory: list[dict[str, Any]] = []

        self.best_val_score: Optional[float] = None
        self.primary_metric: Optional[str] = None
        self.optimization_goal: Optional[str] = None

    @property
    def configured_budget(self) -> int:
        return self.evaluation_budget

    @property
    def actual_evaluations(self) -> int:
        return len(self._results_history)

    def receive_profile(self, profile: DataProfile) -> ActionDecision:
        """Receive initial DataProfile and emit first random configuration."""
        self._profile = profile
        from stratml.core.metrics import resolve_canonical_metric

        n_classes = len(profile.class_distribution) if profile.class_distribution else None
        self.primary_metric, self.optimization_goal, _ = resolve_canonical_metric(
            profile.problem_type, n_classes
        )
        return self._sample_next_action()

    def receive_result(self, result: ExperimentResult) -> ActionDecision:
        """Record evaluation result, track duplicates and trajectory, emit next random configuration."""
        self._results_history.append(result)

        # Check duplicate configuration accounting (Phase 2 & 3 compliance)
        config_sig = f"{result.model_name}:{json.dumps(result.hyperparameters, sort_keys=True)}"
        if config_sig in self._seen_configs:
            self.repeated_configs += 1
            is_repeated = True
        else:
            self._seen_configs.add(config_sig)
            is_repeated = False

        primary_score = (
            getattr(result.metrics, self.primary_metric, None)
            if self.primary_metric
            else None
        )
        if primary_score is not None:
            from stratml.core.metrics import is_better_score

            if self.best_val_score is None or is_better_score(
                primary_score, self.best_val_score, self.optimization_goal
            ):
                self.best_val_score = primary_score

        self.trajectory.append(
            {
                "evaluation": len(self._results_history),
                "iteration": result.iteration,
                "model_name": result.model_name,
                "hyperparameters": result.hyperparameters,
                "validation_metrics": result.metrics.model_dump(exclude_none=True),
                "primary_metric": self.primary_metric,
                "primary_score": primary_score,
                "runtime": result.runtime,
                "status": result.status or ("failed" if result.failed else "completed"),
                "repeated_config": is_repeated,
                "cumulative_evaluations": len(self._results_history),
                "cumulative_fits": sum(
                    getattr(r, "fit_count", 1) for r in self._results_history
                ),
                "decision_source": "random_search",
            }
        )

        # Random Search does not terminate based on internal signals;
        # hard evaluation ceiling is enforced by ExecutionOrchestrator.
        return self._sample_next_action()

    def _sample_next_action(self) -> ActionDecision:
        self._iteration += 1
        problem_type = self._profile.problem_type if self._profile else "classification"
        model_name, hp = self.sampler.sample_candidate(problem_type, self.allowed_models)

        # Standard baseline preprocessing (matching StratML default)
        prep = PreprocessingConfig(
            missing_value_strategy="mean",
            scaling="standard",
            encoding="onehot",
            imbalance_strategy="none",
            feature_selection="none",
        )

        return ActionDecision(
            experiment_id=f"rs_{self.run_id}_{self._iteration}",
            action_type="switch_model",
            parameters={"model_name": model_name, **hp},
            preprocessing=prep,
            reason=DecisionReason(
                trigger="random_selection",
                evidence={"sampled_model": model_name, "sampled_hyperparameters": hp},
                source="random_search",
                selection_mode="random",
            ),
            expected_gain=0.0,
            expected_cost=1.0,
            confidence=1.0,
        )
