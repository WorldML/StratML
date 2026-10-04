# Random Search Baseline Implementation Report

## 1. Repository Audit

A comprehensive pre-implementation audit was conducted across the codebase to identify reusable components and prevent redundant mechanisms:

- **Model Families & Search Utilities**:
  - [`stratml/execution/pipelines/ml_pipeline.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/execution/pipelines/ml_pipeline.py): Defines the authoritative `PAPER_CLASSIFICATION_MODELS` (8 models), `PAPER_REGRESSION_MODELS` (8 models), and `_PARAM_GRIDS`.
  - [`stratml/execution/config/ml_mutations.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/execution/config/ml_mutations.py): Authoritative specification of the classical hyperparameter mutation space (`PAPER_MUTATION_SPACE`).
- **Data & Preprocessing**:
  - `stratml/execution/data/loader.py`, `validator.py`, and `profiler.py`: Standard dataset loading, schema verification, and metadata profiling.
  - `stratml/execution/preprocessing/splitter.py`: Split logic implementing stratified/random splits into Train/Val/Test partitions with test-set isolation.
  - `stratml/execution/preprocessing/preprocessor.py`: Deterministic preprocessing transform applying scalers, encoders, and imputation.
- **Budget Accounting & Orchestrator**:
  - [`stratml/orchestration/orchestrator.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/orchestration/orchestrator.py): Phase 2 hard evaluation ceiling enforcement (`self.actual_evaluations >= self.evaluation_budget`), independent fit and evaluation counters, and manifest emission.
- **Metric Resolution**:
  - [`stratml/core/metrics.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/core/metrics.py): Phase 1 canonical metric resolver (`resolve_canonical_metric`, `is_better_score`, `resolve_task_type`).
- **CLI & Dispatch Architecture**:
  - `stratml/cli/config.py`, `main.py`, and `commands/run.py`: Config merging, command line argument parsing, and orchestrator execution.

No duplicate pipeline, preprocessing, or splitting logic was created. Random Search reuses the exact same execution engine that StratML uses.

---

## 2. Implementation Summary

Random Search is implemented as a standalone baseline engine in [`stratml/baselines/random_search.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/baselines/random_search.py):

```text
Experiment Runner (stratml/runner.py / CLI)
      │
      ├── StratML (DecisionEngine: Multi-Agent LLM / MetaMemory / ValueModel)
      │
      ├── Random Search (RandomSearchEngine: Independent Uniform Sampling)
      │
      └── Auto-sklearn (Future baseline)
             │
             ↓
      ExecutionOrchestrator (Phase 2 Budget & Preprocessing Pipeline)
             │
             ↓
      CanonicalExperimentResult + Manifest
```

1. **`RandomSearchSampler`**: A pure, stateless PRNG configuration sampler instantiated with a dedicated `random.Random(seed)`. At each call, it samples uniformly from the allowed model families and discrete parameter spaces.
2. **`RandomSearchEngine`**: Baseline engine that implements the `receive_profile(profile) -> ActionDecision` and `receive_result(result) -> ActionDecision` contract expected by `ExecutionOrchestrator`. It maintains no state reasoning, no LLM deliberation, no MetaMemory, and no adaptive ranking.
3. **`ExecutionOrchestrator` Integration**: Accepts `system="random_search"`, tracks duplicate configurations, records each step into `self.trajectory`, enforces the hard evaluation ceiling (10 or 20 evaluations), evaluates the isolated test set at completion, and returns a `CanonicalExperimentResult`.
4. **Unified Runner Dispatch**: [`stratml/runner.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/runner.py) provides a common programmatic entry point `run_experiment()`.

---

## 3. Search-Space Mapping

Random Search is strictly bounded to the same classical-ML search space available to StratML:

### Classification Space (8 Models)
| Model | Sampled Parameters & Options | Source |
| :--- | :--- | :--- |
| **`RandomForestClassifier`** | `n_estimators`: `[50, 100, 200]`, `max_depth`: `[None, 5, 10]`, `max_features`: `["sqrt", "log2"]` | Paper Grid |
| **`LogisticRegression`** | `C`: `[0.01, 0.1, 1.0, 10.0]`, `solver`: `["lbfgs", "saga"]` | Paper Grid |
| **`GradientBoostingClassifier`** | `learning_rate`: `[0.01, 0.1, 0.3]`, `n_estimators`: `[50, 100, 200]`, `max_depth`: `[3, 5, 7]` | Paper Grid |
| **`ExtraTreesClassifier`** | `n_estimators`: `[50, 100, 200]`, `max_depth`: `[None, 5, 10]` | Paper Grid |
| **`SVC`** | `C`: `[0.1, 1.0, 10.0]`, `kernel`: `["rbf", "linear"]`, `gamma`: `["scale", "auto"]` | Paper Grid |
| **`KNeighborsClassifier`** | `n_neighbors`: `[3, 5, 7, 11]` | Paper Grid |
| **`GaussianNB`** | `var_smoothing`: `[1e-10, 1e-9, 1e-8, 1e-7]` | Paper Grid |
| **`DecisionTreeClassifier`** | `max_depth`: `[None, 5, 10, 20]`, `min_samples_split`: `[2, 5, 10]` | Paper Grid |

### Regression Space (8 Models)
| Model | Sampled Parameters & Options | Source |
| :--- | :--- | :--- |
| **`RandomForestRegressor`** | `n_estimators`: `[50, 100, 200]`, `max_depth`: `[None, 5, 10]`, `max_features`: `["sqrt", "log2"]` | Paper Grid |
| **`GradientBoostingRegressor`** | `learning_rate`: `[0.01, 0.1, 0.3]`, `n_estimators`: `[50, 100, 200]`, `max_depth`: `[3, 5, 7]` | Paper Grid |
| **`ExtraTreesRegressor`** | `n_estimators`: `[50, 100, 200]`, `max_depth`: `[None, 5, 10]` | Paper Grid |
| **`DecisionTreeRegressor`** | `max_depth`: `[None, 5, 10, 20]`, `min_samples_split`: `[2, 5, 10]` | Paper Grid |
| **`Ridge`** | `alpha`: `[0.01, 0.1, 1.0, 10.0]` | Paper Grid |
| **`Lasso`** | `alpha`: `[0.01, 0.1, 1.0, 10.0]` | Paper Grid |
| **`ElasticNet`** | `alpha`: `[0.01, 0.1, 1.0, 10.0]`, `l1_ratio`: `[0.2, 0.5, 0.7]` | Paper Grid |
| **`KNeighborsRegressor`** | `n_neighbors`: `[3, 5, 7, 11]` | Paper Grid |

No models or parameter ranges were artificially pruned or restricted.

---

## 4. Budget Semantics

Random Search adopts the canonical Phase 2 budget semantics:
$$\text{Budget} = \text{Maximum Number of Completed Candidate Model Evaluations}$$

- Supported ceilings: **10** and **20** model evaluations.
- Under paper standard execution (`tune=False`):
  $$1 \text{ sampled configuration} \to 1 \text{ candidate evaluation} \to 1 \text{ budget unit consumed}$$
- Ceiling enforcement: central orchestrator enforces `self.actual_evaluations >= self.evaluation_budget` prior to every evaluation.
- Decoupled counters: `actual_evaluations`, `actual_fits`, and `decision_iterations` are tracked and logged independently.
- Termination reason is set to `"budget_exhausted"` when the evaluation ceiling is reached.

---

## 5. Seed / Reproducibility Design

- **Isolated PRNG Instance**: `RandomSearchSampler` creates an independent `random.Random(seed)` instance rather than relying on global `random` or `numpy.random` state.
- **Key Sorting**: Dictionary parameter keys are sorted prior to selection (`for param in sorted(param_grid.keys()):`), guaranteeing that hash randomization across Python runtimes does not alter sampling sequences.
- **Deterministic Candidate Sequence**: Two runs configured with identical `(seed, budget, problem_type)` generate the exact same sequence of candidate models and hyperparameters.
- **Provenance in Manifest**: The seed is recorded at the top level of `manifest.json`.

---

## 6. Metric Handling

Random Search utilizes the Phase 1 canonical metric resolver without any hardcoded optimization direction:

- **Binary Classification**:
  - Primary: **ROC-AUC** (`maximize`)
  - Secondary: **F1** (`maximize`)
- **Multiclass Classification**:
  - Primary: **Log Loss** (`minimize`)
  - Secondary: **F1** (`maximize`)
- **Regression**:
  - Primary: **RMSE** (`minimize`)
  - Secondary: **MAE** (`minimize`), **R²** (`maximize`)

Candidate validation scores are evaluated via `is_better_score(score, best_score, goal)`. For Log Loss and RMSE, lower values strictly outperform higher values.

---

## 7. Failure & Duplicate Handling

- **Failure Accounting**: If an evaluation fails (e.g. numerical solver exception), the orchestrator records `failed=True`, `status="failed"`, increments `actual_evaluations += 1`, and does not trigger an infinite retry loop. If remaining budget exists, Random Search samples the next candidate independently.
- **Duplicate Configuration Accounting**: If the PRNG samples a configuration identical to an earlier candidate:
  1. The candidate is executed normally.
  2. It consumes 1 model evaluation from the budget.
  3. The repetition is detected via configuration signature hashing and recorded in `orchestrator.repeated_configs` and `manifest["budget"]["actual_consumption"]["repeated_configs"]`.
  4. The trajectory step logs `repeated_config=True`.

---

## 8. Canonical Experiment Result Integration

[`CanonicalExperimentResult`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/core/schemas.py) is implemented to provide a shared, standardized contract for both StratML and Random Search:

```python
class CanonicalExperimentResult(BaseModel):
    system: str
    dataset: str
    task_type: str
    seed: int
    evaluation_budget: int
    actual_evaluations: int
    actual_fits: int
    decision_iterations: int
    termination_reason: str
    best_validation_score: Optional[float]
    best_test_score: Optional[float]
    runtime: float
    trajectory: list[TrajectoryStep]
    manifest: dict
```

For fields specific to StratML (e.g. `coordinator_weights`, `signals`), Random Search does not fabricate fake data; trajectory steps specify `decision_source="random_search"`.

---

## 9. Manifest Changes

The generated `outputs/<run_id>/manifest.json` now includes the standardized top-level keys:

```json
{
  "manifest_version": "1.0",
  "system": "random_search",
  "run_id": "rs_australian_...",
  "dataset": {
    "name": "australian",
    "path": "data/research/classification/australian.arff",
    "rows": 690,
    "columns": 15,
    "target_column": "A15"
  },
  "task": "classification",
  "task_type": "binary_classification",
  "seed": 42,
  "evaluation_budget": 10,
  "configured_budget": 10,
  "actual_evaluations": 10,
  "actual_fits": 10,
  "decision_iterations": 10,
  "termination_reason": "budget_exhausted",
  "best_validation_score": 0.8924,
  "best_test_score": 0.8871,
  "runtime": 1.45,
  "budget": { ... }
}
```

---

## 10. Tests and Results

A comprehensive unit test suite was implemented in [`tests/unit/test_random_search.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/tests/unit/test_random_search.py):

- **Test A — Deterministic Seed**: Verified identical candidate sequences with identical seeds; verified divergent sequences with differing seeds.
- **Test B — Budget 10**: Verified search terminates cleanly at exactly 10 evaluations with `termination_reason="budget_exhausted"`.
- **Test C — Budget 20**: Verified search terminates cleanly at exactly 20 evaluations with `termination_reason="budget_exhausted"`.
- **Test D — Metric Parity**: Verified canonical metric assignments across binary classification (`roc_auc`), multiclass (`log_loss`), and regression (`rmse`).
- **Test E — Directionality**: Verified `is_better_score` under both maximization and minimization, and verified score tracking in `RandomSearchEngine`.
- **Test F — Failure Accounting**: Verified simulated pipeline failure consumes 1 evaluation without retry loop.
- **Test G — Duplicate Configuration**: Verified repeated candidate configurations are executed, consume budget, and increment duplicate counters.
- **Test H — Manifest Structure**: Verified emission of all mandatory manifest fields.
- **Test I — Test-Set Isolation**: Verified test split is never evaluated during search, with exactly one test evaluation occurring at finalization.
- **Test J — End-to-End Reproducibility**: Verified two full pipeline runs with identical seed produce identical candidate sequences, scores, and test metrics.

---

## 11. Smoke-Test Results

The 3 representative research datasets were tested under `system="random_search"`, `budget=2`:

| Dataset | Problem Type | Target | Metric Resolved | Direction | Budget | Actual Evals | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Australian** | Binary Classification | `A15` | `roc_auc` | Maximize | 2 | 2 | Verified |
| **Vehicle** | Multiclass Classification | `Class` | `log_loss` | Minimize | 2 | 2 | Verified |
| **Boston** | Regression | `MEDV` | `rmse` | Minimize | 2 | 2 | Verified |

All smoke tests executed successfully with complete test isolation and valid canonical results.

---

## 12. Files Modified

1. [`stratml/core/schemas.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/core/schemas.py):
   - Added `"random_search"` to `DecisionReason.source` pattern.
   - Added `"random"` to `DecisionReason.selection_mode` pattern.
   - Added `TrajectoryStep` and `CanonicalExperimentResult` models.
2. [`stratml/execution/schemas.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/execution/schemas.py):
   - Re-exported `TrajectoryStep` and `CanonicalExperimentResult`.
3. [`stratml/baselines/__init__.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/baselines/__init__.py):
   - Created baselines package interface.
4. [`stratml/baselines/random_search.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/baselines/random_search.py):
   - Implemented `RandomSearchSampler`, `RandomSearchEngine`, and parameter space mappings.
5. [`stratml/orchestration/orchestrator.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/orchestration/orchestrator.py):
   - Added `system` argument.
   - Tracked duplicate configurations and `repeated_configs`.
   - Recorded `TrajectoryStep` across iterations and saved `trajectory.json`.
   - Recorded `best_test_score` and `best_val_score`.
   - Populated `system`, `task_type`, and score fields in `manifest.json`.
   - Built and returned `CanonicalExperimentResult`.
6. [`stratml/runner.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/runner.py):
   - Implemented unified `run_experiment()` dispatch function.
7. [`stratml/cli/config.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/cli/config.py):
   - Added `system` and `seed` configuration overrides.
8. [`stratml/cli/main.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/cli/main.py):
   - Added `--system`, `--budget`, and `--seed` flags to CLI `run` subparser.
9. [`stratml/cli/commands/run.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/stratml/cli/commands/run.py):
   - Dispatched `RandomSearchEngine` when `system == "random_search"`.
10. [`tests/unit/test_random_search.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20(CLI)/multi-agent-auto-ml/tests/unit/test_random_search.py):
    - Created unit tests A through J and ARFF smoke tests.

---

## 13. Protocol Impact

- Zero alterations to the frozen experimental protocol.
- Dataset definitions, budgets (10, 20), seeds, metrics, and baselines remain strictly frozen.
- StratML internal decision mechanics, state modeling, and value modeling were preserved unchanged.

---

## 14. Remaining Issues

None. The Random Search baseline is completely implemented, verified, and adheres to the identical execution pipeline and budget enforcement as StratML.

---

## 15. Final Verdict

RANDOM SEARCH PHASE COMPLETE
