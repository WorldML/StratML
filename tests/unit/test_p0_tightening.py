"""
test_p0_tightening.py
---------------------
Unit & regression tests for confirmed P0 tightening items:
P0-1: Canonical paper model space consistency
P0-2: Classical paper action space containment
P0-3: Warm-start corpus provenance & status tracking
P0-4: MetaMemory best-model validation score provenance
P0-5: Transfer vs continual vs independent learning distinction
P0-6: Warm-start activation data integrity & model encoding
P0-7: Frozen resolved LLM configuration recording in run artifacts
P0-8: Experiment seed propagation across split, estimators, samplers, action selector
P0-9: Independent repetition isolation (no cross-run contamination)
P0-10: Collision-safe run ID generation
"""

import json
import os
import shutil
import uuid
from pathlib import Path
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
from stratml.decision.actions.action_generator import (
    PAPER_CLASSICAL_ACTIONS,
    _DEFAULT_MODELS,
    _DEFAULT_REGRESSION_MODELS,
    generate,
)
from stratml.decision.engine import DecisionEngine
from stratml.decision.learning import dataset_builder, meta_memory, value_model
_meta_memory = meta_memory
from stratml.decision.learning.uncertainty import UncertaintyEstimate
from stratml.decision.state.state_builder import build_state_from_profile
from stratml.decision.learning.value_model import (
    _ACTION_VOCAB,
    _MODEL_VOCAB,
    _encode_state_action,
    _load_training_data,
)
from stratml.decision.policy.action_selector import select
from stratml.execution.config.experiment_config_builder import build_experiment_config
from stratml.execution.pipelines.ml_pipeline import (
    MODEL_REGISTRY,
    PAPER_CLASSICAL_MODELS,
    PAPER_CLASSIFICATION_MODELS,
    PAPER_REGRESSION_MODELS,
    run_ml_pipeline,
)
from stratml.execution.preprocessing.preprocessor import apply_preprocessing
from stratml.execution.preprocessing.splitter import split_dataset
from stratml.execution.schemas import DataProfile, DataSplit, Dataset, FeatureInfo, SplitConfig
from tests.unit.test_decision_team import _make_result, _make_state


@pytest.fixture(autouse=True)
def disable_external_apis(monkeypatch):
    """Ensure tests run offline without hitting external APIs."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def _make_dummy_profile(problem_type="classification"):
    return DataProfile(
        dataset_name="test_data",
        dataset_type="tabular",
        rows=100,
        columns=3,
        target_column="target",
        problem_type=problem_type,
        numerical_columns=["a"],
        categorical_columns=["cat"],
        missing_value_ratio=0.1,
        feature_summary=[
            FeatureInfo(name="a", dtype="float64", unique_values=4, missing_percentage=0.1, distribution="uniform"),
            FeatureInfo(name="cat", dtype="object", unique_values=2, missing_percentage=0.0, distribution="uniform"),
        ],
        recommended_metrics=["accuracy"] if problem_type == "classification" else ["r2"],
    )


def _make_clean_split(is_regression=False):
    np.random.seed(42)
    X_tr = pd.DataFrame({"x1": np.random.randn(50), "x2": np.random.randn(50)})
    X_va = pd.DataFrame({"x1": np.random.randn(20), "x2": np.random.randn(20)})
    X_te = pd.DataFrame({"x1": np.random.randn(20), "x2": np.random.randn(20)})
    if is_regression:
        y_tr = pd.Series(np.random.randn(50), name="target")
        y_va = pd.Series(np.random.randn(20), name="target")
        y_te = pd.Series(np.random.randn(20), name="target")
    else:
        y_tr = pd.Series(np.random.randint(0, 2, size=50), name="target")
        y_va = pd.Series(np.random.randint(0, 2, size=20), name="target")
        y_te = pd.Series(np.random.randint(0, 2, size=20), name="target")
    return DataSplit(X_train=X_tr, X_val=X_va, X_test=X_te, y_train=y_tr, y_val=y_va, y_test=y_te)


# ===========================================================================
# P0-1: Freeze the Paper Model Space
# ===========================================================================

class TestP0_1_FreezePaperModelSpace:
    def test_every_paper_model_exists_in_model_registry(self):
        for model_name in PAPER_CLASSICAL_MODELS:
            assert model_name in MODEL_REGISTRY, f"Paper model '{model_name}' missing from MODEL_REGISTRY"

    def test_candidate_generator_models_match_paper_subsets(self):
        assert set(_DEFAULT_MODELS) == set(PAPER_CLASSIFICATION_MODELS)
        assert set(_DEFAULT_REGRESSION_MODELS) == set(PAPER_REGRESSION_MODELS)

    def test_every_paper_model_can_be_encoded_by_value_model(self):
        for model_name in PAPER_CLASSICAL_MODELS:
            assert model_name in _MODEL_VOCAB, f"Paper model '{model_name}' missing from _MODEL_VOCAB"
            state = _make_state(model_name=model_name)
            vec = _encode_state_action(state, "switch_model")
            # model_name is encoded at index 11
            assert vec[11] == float(_MODEL_VOCAB[model_name])

    @pytest.mark.parametrize("model_name", PAPER_CLASSICAL_MODELS)
    def test_every_paper_model_can_be_executed(self, model_name):
        is_reg = model_name in PAPER_REGRESSION_MODELS
        split = _make_clean_split(is_regression=is_reg)
        decision = ActionDecision(
            experiment_id=f"exp_{model_name}",
            action_type="switch_model",
            parameters={"model_name": model_name},
            preprocessing=PreprocessingConfig(
                missing_value_strategy="mean",
                scaling="none",
                encoding="none",
                imbalance_strategy="none",
                feature_selection="none",
            ),
            reason="test_p0_1",
        )
        config = build_experiment_config(decision, tune=False)
        result = run_ml_pipeline(config, split)
        assert result.model is not None
        assert len(result.y_val_pred) == len(split.X_val)


# ===========================================================================
# P0-2: Freeze the Paper Action Space
# ===========================================================================

class TestP0_2_FreezePaperActionSpace:
    def test_paper_action_set_contains_only_executable_classical_actions(self):
        dl_actions = {"change_optimizer", "unfreeze_backbone", "switch_architecture", "early_stop"}
        for dl_act in dl_actions:
            assert dl_act not in PAPER_CLASSICAL_ACTIONS, f"DL action '{dl_act}' must not be in classical action space"
        assert "terminate" in PAPER_CLASSICAL_ACTIONS
        assert "switch_model" in PAPER_CLASSICAL_ACTIONS

    @pytest.mark.parametrize("signal_kwargs", [
        {"underfitting": "moderate"},
        {"overfitting": "high"},
        {"stagnating": "mild"},
        {"diverging": "severe"},
        {"diminishing_returns": "moderate"},
        {"converged": "strong", "well_fitted": "strong"},
        {},  # pure exploration
    ])
    def test_generated_classical_actions_are_subset_of_paper_action_set(self, signal_kwargs):
        state = _make_state(**signal_kwargs)
        candidates = generate(state)
        assert len(candidates) > 0
        for c in candidates:
            assert c.action_type in PAPER_CLASSICAL_ACTIONS, (
                f"Generated action '{c.action_type}' escaped PAPER_CLASSICAL_ACTIONS"
            )


# ===========================================================================
# P0-3: Warm-Start Corpus Provenance
# ===========================================================================

class TestP0_3_CorpusProvenance:
    def test_two_datasets_and_runs_remain_distinguishable_in_corpus(self, tmp_path):
        csv_path = tmp_path / "decision_dataset.csv"
        dataset_builder._DATASET_PATH = csv_path
        dataset_builder._UNIFIED_PATH = csv_path

        state1 = _make_state()
        state1.meta.experiment_id = "exp_run1_it0"
        state1.meta.iteration = 0
        state1.model.model_name = "RandomForestClassifier"

        act1 = CandidateAction(action_type="switch_model", parameters={"model_name": "RandomForestClassifier"})
        dataset_builder.record(state1, act1, predicted_gain=0.05, run_id="run_alpha", dataset_id="dataset_iris", seed=42)
        dataset_builder.backfill_last_gain(0.04, status="completed")

        state2 = _make_state()
        state2.meta.experiment_id = "exp_run2_it0"
        state2.meta.iteration = 0
        state2.model.model_name = "GradientBoostingClassifier"

        act2 = CandidateAction(action_type="switch_model", parameters={"model_name": "GradientBoostingClassifier"})
        dataset_builder.record(state2, act2, predicted_gain=0.02, run_id="run_beta", dataset_id="dataset_wine", seed=99)
        dataset_builder.backfill_last_gain(-0.01, status="completed")

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df["dataset_id"]) == ["dataset_iris", "dataset_wine"]
        assert list(df["run_id"]) == ["run_alpha", "run_beta"]
        assert list(df["seed"].astype(int)) == [42, 99]
        assert list(df["model_name"]) == ["RandomForestClassifier", "GradientBoostingClassifier"]
        assert list(df["status"]) == ["completed", "completed"]

    def test_failed_and_incomplete_observations_are_distinguishable(self, tmp_path):
        csv_path = tmp_path / "provenance_status.csv"
        dataset_builder._DATASET_PATH = csv_path
        dataset_builder._UNIFIED_PATH = csv_path

        state = _make_state()
        act = CandidateAction(action_type="switch_model", parameters={})

        # 1. Completed
        dataset_builder.record(state, act, run_id="run_1", dataset_id="ds_1")
        dataset_builder.backfill_last_gain(0.05, status="completed")

        # 2. Incomplete / pending
        dataset_builder.record(state, act, run_id="run_1", dataset_id="ds_1")

        # 3. Failed
        dataset_builder.record(state, act, run_id="run_1", dataset_id="ds_1")
        dataset_builder.backfill_last_gain(0.0, status="failed")

        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert list(df["status"]) == ["completed", "pending", "failed"]


# ===========================================================================
# P0-4: MetaMemory Best-Model Provenance
# ===========================================================================

class TestP0_4_MetaMemoryBestModelProvenance:
    def test_metamemory_records_actual_best_model_not_last_tried(self, monkeypatch):
        recorded_calls = []

        def mock_record_run(meta_features, best_model, best_score, run_id, **kwargs):
            recorded_calls.append({
                "best_model": best_model,
                "best_score": best_score,
                "run_id": run_id,
                **kwargs,
            })

        monkeypatch.setattr(meta_memory, "record_run", mock_record_run)

        run_id = f"test_best_model_{uuid.uuid4().hex[:6]}"
        out_dir = Path("outputs") / run_id
        try:
            engine = DecisionEngine(run_id=run_id, max_iterations=3, optimization_goal="maximize")
            profile = _make_dummy_profile()

            # Bootstrap
            engine.receive_profile(profile)

            # Iteration 1: Model A gets high score (0.92)
            res1 = _make_result(accuracy=0.92)
            res1.iteration = 1
            res1.model_name = "RandomForestClassifier"
            engine.receive_result(res1)
            assert engine._best_model == "RandomForestClassifier"
            assert engine._best_val_score == 0.92

            # Iteration 2: Model B gets worse score (0.70)
            res2 = _make_result(accuracy=0.70)
            res2.iteration = 2
            res2.model_name = "DecisionTreeClassifier"
            act_term = engine.receive_result(res2)

            # Iteration 3: Exhaust budget -> terminates
            res3 = _make_result(accuracy=0.65)
            res3.iteration = 3
            res3.model_name = "GaussianNB"
            engine.receive_result(res3)

            assert len(recorded_calls) >= 1
            last_record = recorded_calls[-1]
            # Must record Model A (RandomForestClassifier, 0.92), NOT Model B or Model C
            assert last_record["best_model"] == "RandomForestClassifier"
            assert last_record["best_score"] == 0.92
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


# ===========================================================================
# P0-5 & P0-9: Transfer vs Continual vs Independent Distinguishability
# ===========================================================================

class TestP0_5_And_P0_9_TransferContinualIndependent:
    def _create_synthetic_corpus(self, csv_path: Path):
        rows = []
        # 30 rows from previous run on same dataset (ds_A, run_past_1)
        for i in range(30):
            rows.append({
                "dataset_id": "ds_A", "run_id": "run_past_1", "seed": 42,
                "experiment_id": f"exp_past1_{i}", "iteration": i,
                "primary_metric": "accuracy", "best_score": 0.8, "improvement_rate": 0.01,
                "slope": 0.005, "volatility": 0.01, "steps_since_improvement": 1,
                "num_samples": 150, "num_features": 4, "missing_ratio": 0.0,
                "runtime": 0.1, "remaining_budget": 10.0, "model_name": "RandomForestClassifier",
                "action_type": "switch_model", "observed_gain": 0.02, "status": "completed",
            })
        # 30 rows from another dataset (ds_B, run_transfer_1)
        for i in range(30):
            rows.append({
                "dataset_id": "ds_B", "run_id": "run_transfer_1", "seed": 42,
                "experiment_id": f"exp_trans1_{i}", "iteration": i,
                "primary_metric": "accuracy", "best_score": 0.7, "improvement_rate": 0.01,
                "slope": 0.005, "volatility": 0.01, "steps_since_improvement": 1,
                "num_samples": 500, "num_features": 10, "missing_ratio": 0.0,
                "runtime": 0.2, "remaining_budget": 10.0, "model_name": "SVC",
                "action_type": "switch_model", "observed_gain": 0.03, "status": "completed",
            })
        # 10 rows from current run (ds_A, run_curr)
        for i in range(10):
            rows.append({
                "dataset_id": "ds_A", "run_id": "run_curr", "seed": 42,
                "experiment_id": f"exp_curr_{i}", "iteration": i,
                "primary_metric": "accuracy", "best_score": 0.85, "improvement_rate": 0.01,
                "slope": 0.005, "volatility": 0.01, "steps_since_improvement": 1,
                "num_samples": 150, "num_features": 4, "missing_ratio": 0.0,
                "runtime": 0.1, "remaining_budget": 10.0, "model_name": "LogisticRegression",
                "action_type": "switch_model", "observed_gain": 0.01, "status": "completed",
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    def test_independent_mode_prevents_cross_run_contamination(self, tmp_path):
        csv_path = tmp_path / "decision_dataset.csv"
        self._create_synthetic_corpus(csv_path)

        # In independent mode with current_run_id="run_curr", only 10 rows match -> (< 50) -> None
        X, y = _load_training_data(csv_path, current_run_id="run_curr", current_dataset_id="ds_A", history_mode="independent")
        assert X is None
        assert y is None

    def test_continual_mode_filters_to_same_dataset(self, tmp_path):
        csv_path = tmp_path / "decision_dataset.csv"
        self._create_synthetic_corpus(csv_path)

        # In continual mode with ds_A, 30 + 10 = 40 rows (< 50) -> None
        X, y = _load_training_data(csv_path, current_run_id="run_curr", current_dataset_id="ds_A", history_mode="continual")
        assert X is None

        # If another 15 rows exist on ds_A (total 55 on ds_A), continual mode activates
        df = pd.read_csv(csv_path)
        extra = df[df["dataset_id"] == "ds_A"].iloc[:15].copy()
        extra["experiment_id"] = [f"extra_{i}" for i in range(15)]
        pd.concat([df, extra]).to_csv(csv_path, index=False)

        X, y = _load_training_data(csv_path, current_run_id="run_curr", current_dataset_id="ds_A", history_mode="continual")
        assert X is not None
        assert len(X) == 55

    def test_transfer_mode_filters_to_other_datasets(self, tmp_path):
        csv_path = tmp_path / "decision_dataset.csv"
        self._create_synthetic_corpus(csv_path)

        # In transfer mode with ds_A, only ds_B rows match (30 rows < 50) -> None
        X, y = _load_training_data(csv_path, current_run_id="run_curr", current_dataset_id="ds_A", history_mode="transfer")
        assert X is None

        # With 30 more rows on ds_B (total 60 transfer rows) -> activates
        df = pd.read_csv(csv_path)
        extra_b = df[df["dataset_id"] == "ds_B"].copy()
        pd.concat([df, extra_b]).to_csv(csv_path, index=False)

        X, y = _load_training_data(csv_path, current_run_id="run_curr", current_dataset_id="ds_A", history_mode="transfer")
        assert X is not None
        assert len(X) == 60


# ===========================================================================
# P0-6: Warm-Start Activation Data Integrity
# ===========================================================================

class TestP0_6_WarmStartActivationDataIntegrity:
    def test_insufficient_valid_observations_does_not_activate(self, tmp_path):
        csv_path = tmp_path / "insufficient.csv"
        rows = [
            {
                "best_score": 0.8, "improvement_rate": 0.01, "slope": 0.005, "volatility": 0.01,
                "steps_since_improvement": 1, "num_samples": 100, "num_features": 4, "missing_ratio": 0.0,
                "runtime": 0.1, "remaining_budget": 10.0, "model_name": "LogisticRegression",
                "action_type": "switch_model", "observed_gain": 0.01, "status": "completed",
            }
            for _ in range(49)  # 49 rows < 50
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        X, y = _load_training_data(csv_path)
        assert X is None
        assert y is None

    def test_invalid_and_failed_rows_not_counted(self, tmp_path):
        csv_path = tmp_path / "with_failed.csv"
        rows = [
            {
                "best_score": 0.8, "improvement_rate": 0.01, "slope": 0.005, "volatility": 0.01,
                "steps_since_improvement": 1, "num_samples": 100, "num_features": 4, "missing_ratio": 0.0,
                "runtime": 0.1, "remaining_budget": 10.0, "model_name": "LogisticRegression",
                "action_type": "switch_model", "observed_gain": 0.01, "status": "completed",
            }
            for _ in range(40)
        ] + [
            {
                "best_score": 0.8, "improvement_rate": 0.01, "slope": 0.005, "volatility": 0.01,
                "steps_since_improvement": 1, "num_samples": 100, "num_features": 4, "missing_ratio": 0.0,
                "runtime": 0.1, "remaining_budget": 10.0, "model_name": "LogisticRegression",
                "action_type": "switch_model", "observed_gain": "", "status": "pending",
            }
            for _ in range(20)
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        X, y = _load_training_data(csv_path)
        assert X is None  # only 40 completed rows, 20 pending ignored

    def test_sufficient_valid_observations_activates(self, tmp_path):
        csv_path = tmp_path / "sufficient.csv"
        rows = [
            {
                "best_score": 0.8, "improvement_rate": 0.01, "slope": 0.005, "volatility": 0.01,
                "steps_since_improvement": 1, "num_samples": 100, "num_features": 4, "missing_ratio": 0.0,
                "runtime": 0.1, "remaining_budget": 10.0, "model_name": "LogisticRegression",
                "action_type": "switch_model", "observed_gain": 0.01, "status": "completed",
            }
            for _ in range(50)  # Exactly 50 rows
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        X, y = _load_training_data(csv_path)
        assert X is not None
        assert len(X) == 50

    def test_model_name_encoded_correctly_in_feature_matrix(self, tmp_path):
        csv_path = tmp_path / "model_enc.csv"
        rows = [
            {
                "best_score": 0.8, "improvement_rate": 0.01, "slope": 0.005, "volatility": 0.01,
                "steps_since_improvement": 1, "num_samples": 100, "num_features": 4, "missing_ratio": 0.0,
                "runtime": 0.1, "remaining_budget": 10.0, "model_name": "RandomForestClassifier",
                "action_type": "switch_model", "observed_gain": 0.01, "status": "completed",
            }
            for _ in range(50)
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        X, y = _load_training_data(csv_path)
        # model_name_enc is at index 11
        expected_rf_enc = float(_MODEL_VOCAB["RandomForestClassifier"])
        assert (X[:, 11] == expected_rf_enc).all()


# ===========================================================================
# P0-7: Freeze / Record LLM Configuration
# ===========================================================================

class TestP0_7_RecordLLMConfiguration:
    def test_llm_configuration_recorded_in_artifacts(self):
        run_id = f"test_llm_cfg_{uuid.uuid4().hex[:6]}"
        out_dir = Path("outputs") / run_id
        try:
            engine = DecisionEngine(run_id=run_id, llm_mode=False)
            cfg_file = out_dir / "artifacts" / "llm_config.json"
            assert cfg_file.exists(), "llm_config.json must be written to run artifacts"

            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            assert cfg["provider"] == "groq"
            assert cfg["model"] == "llama-3.3-70b-versatile"
            assert cfg["llm_mode_enabled"] is False
            assert "temperature" in cfg
            assert "structured_output_schema" in cfg
            assert cfg["fallback_behavior"] == "rule_based"
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


# ===========================================================================
# P0-8: Experiment Seed Propagation
# ===========================================================================

class TestP0_8_ExperimentSeedPropagation:
    def test_split_dataset_same_seed_reproducible_different_seed_differs(self):
        from stratml.execution.data.validator import build_dataset

        df = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})
        ds = build_dataset(df, "test", "y")

        s1 = split_dataset(ds, SplitConfig(method="stratified", random_seed=42), "classification")
        s2 = split_dataset(ds, SplitConfig(method="stratified", random_seed=42), "classification")
        s3 = split_dataset(ds, SplitConfig(method="stratified", random_seed=99), "classification")

        assert s1.X_train["x"].tolist() == s2.X_train["x"].tolist()
        assert s1.X_train["x"].tolist() != s3.X_train["x"].tolist()

    def test_ml_pipeline_propagates_seed_to_estimator_random_state(self):
        split = _make_clean_split(is_regression=False)
        decision = ActionDecision(
            experiment_id="exp_seed_test",
            action_type="switch_model",
            parameters={"model_name": "RandomForestClassifier"},
            preprocessing=PreprocessingConfig(
                missing_value_strategy="mean", scaling="none",
                encoding="none", imbalance_strategy="none", feature_selection="none",
            ),
            reason="seed_test",
        )
        config = build_experiment_config(decision, tune=False, seed=777)
        res = run_ml_pipeline(config, split)
        assert res.model.random_state == 777

    def test_action_selector_seeded_rng_is_reproducible(self):
        from stratml.decision.agents.coordinator_agent import RankedAction
        from stratml.core.schemas import AgentScore
        state = _make_state()
        ranked = [
            RankedAction(action_type="switch_model", parameters={}, predicted_gain=0.1, predicted_cost=0.5, confidence=0.8, agent_scores=AgentScore(), final_score=0.9),
            RankedAction(action_type="increase_model_capacity", parameters={}, predicted_gain=0.08, predicted_cost=0.5, confidence=0.8, agent_scores=AgentScore(), final_score=0.8),
            RankedAction(action_type="modify_regularization", parameters={}, predicted_gain=0.05, predicted_cost=0.5, confidence=0.8, agent_scores=AgentScore(), final_score=0.7),
        ]
        # With identical seed, choices under exploration are deterministic
        sel1 = select(state, ranked, seed=123)
        sel2 = select(state, ranked, seed=123)
        assert sel1.action_type == sel2.action_type


# ===========================================================================
# P0-10: Collision-Safe Run IDs
# ===========================================================================

class TestP0_10_CollisionSafeRunIds:
    def test_rapid_successive_runs_generate_unique_run_ids_and_dirs(self):
        run_ids = set()
        created_dirs = []
        try:
            for _ in range(50):
                engine = DecisionEngine()
                assert engine.run_id not in run_ids, f"Collision detected for run_id: {engine.run_id}"
                run_ids.add(engine.run_id)
                created_dirs.append(engine._out_dir)

            assert len(run_ids) == 50
            # All directories are distinct
            assert len({str(d) for d in created_dirs}) == 50
        finally:
            for d in created_dirs:
                shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# Round 2 P0 Audits: P0-1 through P0-5
# ===========================================================================

class TestRound2_P0_1_DatasetFingerprintIdentity:
    def test_same_dataset_content_produces_identical_fingerprint(self):
        from stratml.execution.data.validator import build_dataset, compute_dataset_fingerprint

        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30], "target": [0, 1, 0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30], "target": [0, 1, 0]})

        fp1 = compute_dataset_fingerprint(df1)
        fp2 = compute_dataset_fingerprint(df2)
        assert fp1 == fp2
        assert len(fp1) == 16

        ds1 = build_dataset(df1, "ds_test", "target")
        ds2 = build_dataset(df2, "ds_test", "target")
        assert ds1.dataset_fingerprint == fp1
        assert ds1.dataset_fingerprint == ds2.dataset_fingerprint

    def test_different_dataset_content_or_version_produces_different_fingerprint(self):
        from stratml.execution.data.validator import compute_dataset_fingerprint

        df_base = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30], "target": [0, 1, 0]})
        df_modified_val = pd.DataFrame({"a": [1.0, 2.0, 99.0], "b": [10, 20, 30], "target": [0, 1, 0]})
        df_extra_row = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10, 20, 30, 40], "target": [0, 1, 0, 1]})
        df_rename_col = pd.DataFrame({"c": [1.0, 2.0, 3.0], "b": [10, 20, 30], "target": [0, 1, 0]})

        fp_base = compute_dataset_fingerprint(df_base)
        assert compute_dataset_fingerprint(df_modified_val) != fp_base
        assert compute_dataset_fingerprint(df_extra_row) != fp_base
        assert compute_dataset_fingerprint(df_rename_col) != fp_base

    def test_dataset_fingerprint_propagates_to_profile_state_and_corpus(self, tmp_path):
        from stratml.execution.data.validator import build_dataset
        from stratml.execution.data.profiler import build_profile

        df = pd.DataFrame({"feat": range(20), "target": [0, 1] * 10})
        ds = build_dataset(df, "fp_test", "target")
        prof = build_profile(ds)

        assert prof.dataset_fingerprint == ds.dataset_fingerprint
        assert prof.dataset_fingerprint is not None

        state = build_state_from_profile(prof, run_id="run_fp_check")
        assert state.dataset.dataset_fingerprint == ds.dataset_fingerprint

        csv_path = tmp_path / "decision_dataset.csv"
        dataset_builder._DATASET_PATH = csv_path
        dataset_builder._UNIFIED_PATH = csv_path

        act = CandidateAction(action_type="switch_model", parameters={"model_name": "RandomForestClassifier"})
        dataset_builder.record(state, act, predicted_gain=0.05, run_id="run_fp_check", dataset_id="fp_test")
        dataset_builder.backfill_last_gain(0.04, status="completed")

        recorded_df = pd.read_csv(csv_path)
        assert "dataset_fingerprint" in recorded_df.columns
        assert recorded_df["dataset_fingerprint"].iloc[0] == ds.dataset_fingerprint


class TestRound2_P0_2_And_P0_5_MetaMemoryHistoryModeAndIsolation:
    def test_independent_runs_do_not_influence_each_other_via_meta_memory(self, tmp_path, monkeypatch):
        global_mem = tmp_path / "global_meta_memory.jsonl"
        monkeypatch.setattr(_meta_memory, "_MEMORY_FILE", global_mem)

        profile = DataProfile(
            dataset_name="dataset_iso",
            dataset_type="tabular",
            rows=100,
            columns=5,
            target_column="target",
            problem_type="classification",
            numerical_columns=["f1", "f2"],
            categorical_columns=[],
            missing_value_ratio=0.0,
            class_distribution={"0": 50, "1": 50},
            feature_summary=[],
            recommended_metrics=["accuracy"],
            dataset_fingerprint="fingerprint_iso_1",
        )

        # Run A: independent run that finishes with best model GradientBoostingClassifier
        engine_a = DecisionEngine(
            history_mode="independent",
            allowed_models=["RandomForestClassifier", "GradientBoostingClassifier"],
            llm_mode=False,
        )
        engine_a.receive_profile(profile)
        monkeypatch.setattr(_meta_memory, "_MEMORY_FILE", global_mem)
        _meta_memory.record_run(
            profile,
            best_model="GradientBoostingClassifier",
            best_score=0.95,
            run_id=engine_a.run_id,
            dataset_id="dataset_iso",
            dataset_fingerprint="fingerprint_iso_1",
        )

        # Run B: independent run on same dataset
        engine_b = DecisionEngine(
            history_mode="independent",
            allowed_models=["RandomForestClassifier", "GradientBoostingClassifier"],
            llm_mode=False,
        )
        dec_b = engine_b.receive_profile(profile)

        # In independent mode, previous runs MUST NOT influence allowed_models or decision
        assert engine_b.allowed_models == ["RandomForestClassifier", "GradientBoostingClassifier"]
        assert dec_b.parameters.get("model_name") == "RandomForestClassifier"

    def test_continual_mode_reuses_relevant_experience_on_same_dataset(self, tmp_path, monkeypatch):
        global_mem = tmp_path / "global_meta_memory.jsonl"
        monkeypatch.setattr(_meta_memory, "_MEMORY_FILE", global_mem)

        profile_same = DataProfile(
            dataset_name="dataset_cont",
            dataset_type="tabular",
            rows=100,
            columns=5,
            target_column="target",
            problem_type="classification",
            numerical_columns=["f1", "f2"],
            categorical_columns=[],
            missing_value_ratio=0.0,
            class_distribution={"0": 50, "1": 50},
            feature_summary=[],
            recommended_metrics=["accuracy"],
            dataset_fingerprint="fingerprint_cont_1",
        )

        _meta_memory.record_run(
            profile_same,
            best_model="GradientBoostingClassifier",
            best_score=0.92,
            run_id="run_past_cont",
            dataset_id="dataset_cont",
            dataset_fingerprint="fingerprint_cont_1",
        )

        engine_cont = DecisionEngine(
            history_mode="continual",
            allowed_models=["RandomForestClassifier", "GradientBoostingClassifier"],
            llm_mode=False,
        )
        dec_cont = engine_cont.receive_profile(profile_same)

        assert engine_cont.allowed_models[0] == "GradientBoostingClassifier"
        assert dec_cont.parameters.get("model_name") == "GradientBoostingClassifier"

    def test_transfer_mode_reuses_cross_dataset_experience_only(self, tmp_path, monkeypatch):
        global_mem = tmp_path / "global_meta_memory.jsonl"
        monkeypatch.setattr(_meta_memory, "_MEMORY_FILE", global_mem)

        profile_a = DataProfile(
            dataset_name="dataset_A",
            dataset_type="tabular",
            rows=100,
            columns=5,
            target_column="target",
            problem_type="classification",
            numerical_columns=["f1", "f2"],
            categorical_columns=[],
            missing_value_ratio=0.0,
            class_distribution={"0": 50, "1": 50},
            feature_summary=[],
            recommended_metrics=["accuracy"],
            dataset_fingerprint="fingerprint_A",
        )
        profile_b = DataProfile(
            dataset_name="dataset_B",
            dataset_type="tabular",
            rows=100,
            columns=5,
            target_column="target",
            problem_type="classification",
            numerical_columns=["f1", "f2"],
            categorical_columns=[],
            missing_value_ratio=0.0,
            class_distribution={"0": 50, "1": 50},
            feature_summary=[],
            recommended_metrics=["accuracy"],
            dataset_fingerprint="fingerprint_B",
        )

        _meta_memory.record_run(
            profile_a,
            best_model="GradientBoostingClassifier",
            best_score=0.92,
            run_id="run_past_A",
            dataset_id="dataset_A",
            dataset_fingerprint="fingerprint_A",
        )

        # Transfer run on SAME dataset: must NOT reuse dataset_A experience
        engine_same = DecisionEngine(
            history_mode="transfer",
            allowed_models=["RandomForestClassifier", "GradientBoostingClassifier"],
            llm_mode=False,
        )
        engine_same.receive_profile(profile_a)
        assert engine_same.allowed_models == ["RandomForestClassifier", "GradientBoostingClassifier"]

        # Transfer run on DIFFERENT dataset: MUST reuse cross-dataset experience
        engine_diff = DecisionEngine(
            history_mode="transfer",
            allowed_models=["RandomForestClassifier", "GradientBoostingClassifier"],
            llm_mode=False,
        )
        dec_diff = engine_diff.receive_profile(profile_b)
        assert engine_diff.allowed_models[0] == "GradientBoostingClassifier"
        assert dec_diff.parameters.get("model_name") == "GradientBoostingClassifier"


class TestRound2_P0_3_AuthoritativeLLMMode:
    def test_llm_mode_false_never_calls_llm_even_with_api_key(self, monkeypatch):
        from stratml.decision.llm_control import is_llm_enabled
        from stratml.decision.actions import action_generator
        from stratml.decision.agents import coordinator_agent, evaluator_agent, performance_agent, stability_agent, efficiency_agent

        monkeypatch.setenv("GROQ_API_KEY", "gsk_live_api_key_for_testing")

        engine = DecisionEngine(llm_mode=False)
        assert engine.llm_config["llm_mode_configured"] is False
        assert engine.llm_config["llm_mode_enabled"] is False
        assert is_llm_enabled() is False

        # Verify artifacts/llm_config.json reflects this
        cfg_file = engine._out_dir / "artifacts" / "llm_config.json"
        assert cfg_file.exists()
        saved = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert saved["llm_mode_configured"] is False
        assert saved["llm_mode_enabled"] is False

        # Verify LLM helpers are never called
        called = {"llm": False}
        def fail_if_called(*args, **kwargs):
            called["llm"] = True
            raise RuntimeError("LLM was invoked despite llm_mode=False!")

        monkeypatch.setattr(action_generator, "_llm_candidates", fail_if_called)
        monkeypatch.setattr(coordinator_agent, "_llm_rank", fail_if_called)
        monkeypatch.setattr(evaluator_agent, "_llm_audit", fail_if_called)
        monkeypatch.setattr(performance_agent, "_llm_score", fail_if_called)
        monkeypatch.setattr(stability_agent, "_llm_score", fail_if_called)
        monkeypatch.setattr(efficiency_agent, "_llm_score", fail_if_called)

        state = _make_state()
        state.meta.iteration = 1  # non-bootstrap

        # Action generator returns rule candidates
        cands = action_generator.generate(state)
        assert len(cands) > 0
        assert not called["llm"]

        # Coordinator rank returns rule rank
        estimates = [
            UncertaintyEstimate(
                action_type=c.action_type,
                parameters=c.parameters,
                predicted_gain=0.05,
                predicted_cost=0.5,
                confidence=0.8,
                variance=0.01,
            )
            for c in cands
        ]
        p_scores = performance_agent.score(state, estimates)
        e_scores = efficiency_agent.score(state, estimates)
        s_scores = stability_agent.score(state, estimates)
        ranked = coordinator_agent.rank(state, estimates, p_scores, e_scores, s_scores)
        assert len(ranked) > 0
        assert not called["llm"]


class TestRound2_P0_4_ExperimentSeedPropagationToRemainingComponents:
    def test_profiler_uses_experiment_seed_for_sampling(self, monkeypatch):
        from stratml.execution.data.validator import build_dataset
        from stratml.execution.data.profiler import build_profile
        import stratml.execution.data.profiler as profiler_mod

        df = pd.DataFrame({"x": range(1000), "y": [0, 1] * 500})
        ds = build_dataset(df, "test", "y")

        captured_seeds = []
        orig_describe = profiler_mod._describe_feature

        def tracking_describe(series, random_seed=42):
            captured_seeds.append(random_seed)
            return orig_describe(series, random_seed=random_seed)

        monkeypatch.setattr(profiler_mod, "_describe_feature", tracking_describe)

        build_profile(ds, random_seed=777)
        assert 777 in captured_seeds

    def test_value_model_uses_experiment_seed_for_random_forest(self, monkeypatch):
        from stratml.decision.learning import value_model
        from sklearn.ensemble import RandomForestRegressor

        state = _make_state()
        candidates = [CandidateAction(action_type="switch_model", parameters={"model_name": "RandomForestClassifier"})]

        captured_seed = []
        orig_rf_init = RandomForestRegressor.__init__

        def tracking_rf_init(self, *args, **kwargs):
            captured_seed.append(kwargs.get("random_state"))
            orig_rf_init(self, *args, **kwargs)

        monkeypatch.setattr(RandomForestRegressor, "__init__", tracking_rf_init)

        # Mock X_train, y_train to trigger RF training
        monkeypatch.setattr(value_model, "_load_training_data", lambda *a, **kw: (np.ones((60, 13)), np.ones(60)))

        value_model.predict(state, candidates, seed=999)
        assert 999 in captured_seed
