# Budget Implementation Report

## 1. Budget Semantics

In accordance with the frozen experimental protocol for the StratML paper (*"Beyond Configuration Search: A State-Aware Framework for AutoML Experimentation"*), the experimental budget is defined canonically as:

$$\text{Budget} = \text{Maximum number of completed candidate model evaluations}$$

Specifically, the frozen protocol mandates two experimental budget ceilings:
- **10 Model Evaluations**
- **20 Model Evaluations**

Crucially, **budget is decoupled from decision iterations and underlying model fits**:
- A **decision iteration** represents one step in the agent's interaction loop (requesting an action and receiving a result).
- A **model evaluation** represents an actual candidate model architecture/hyperparameter configuration evaluated against the validation criteria.
- A **model fit** represents an underlying execution of an estimator's `.fit()` call (e.g. including $K$-fold cross-validation splits and final refits).

The legacy configuration default of `max_iterations = 5` has been removed. The orchestrator, decision engine, CLI, and schemas now natively support and enforce `evaluation_budget = 10` and `evaluation_budget = 20` (while maintaining backward-compatible property accessors for legacy callers).

---

## 2. Canonical Evaluation Definition

A **Model Evaluation** is defined as the training and validation scoring of a single distinct candidate model configuration:
1. In standard paper execution mode (`tune=False`):
   - Exactly 1 candidate model configuration is trained and evaluated per iteration.
   - `eval_count = 1`.
   - `actual_evaluations` increments by 1.
2. In hyperparameter tuning mode (`tune=True`):
   - `RandomizedSearchCV` samples and evaluates $K$ hyperparameter candidates ($K \le 10$).
   - `eval_count = K` ($K$ distinct configurations evaluated).
   - `actual_evaluations` increments by $K$.
   - The number of candidates evaluated in any tuning step is strictly bounded by the remaining evaluation budget:
     $$\text{n\_iter} = \min(10, \text{remaining\_evaluation\_budget})$$
   - Thus, tuning can never cause a budget overrun.
3. In Deep Learning mode (`model_type == "dl"`):
   - 1 neural architecture configuration is trained and evaluated per iteration.
   - `eval_count = 1`.

---

## 3. Orchestrator Enforcement

Budget enforcement is centralized in [`stratml/orchestration/orchestrator.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/orchestration/orchestrator.py):

1. **Pre-Evaluation Ceiling Check**:
   Before initiating any candidate evaluation (Phase 4/5), the orchestrator checks:
   ```python
   if self.actual_evaluations >= self.evaluation_budget:
       self.termination_reason = "budget_exhausted"
       break
   ```
   The orchestrator strictly guarantees that no candidate evaluation is ever initiated when `actual_evaluations >= evaluation_budget`.

2. **Early Termination**:
   The decision engine is permitted to terminate early (e.g., convergence detected, well-fitted signal confirmed, or diminishing returns). When the agent returns `action_type == "terminate"` before the budget ceiling is reached:
   - The loop breaks cleanly.
   - `termination_reason = "agent_terminated"`.
   - `actual_evaluations < evaluation_budget`.

3. **Post-Evaluation Boundary Check**:
   Immediately following candidate evaluation and result assembly (Phase 8), remaining budget and soft time budget are verified. If `actual_evaluations >= evaluation_budget`, the final result is reported to the decision engine (for logging/auditing), and the loop terminates cleanly with `termination_reason = "budget_exhausted"`.

---

## 4. Evaluation vs Fit Accounting

The system strictly tracks both quantities independently and preserves them across state, artifacts, and manifests:

| Metric | Definition | Standard (`tune=False`) | Tuning (`tune=True`, 3-fold CV) |
| :--- | :--- | :--- | :--- |
| **`actual_evaluations`** | Logical candidate configurations evaluated | $1$ per iteration | $K$ candidates ($K \le 10$) |
| **`actual_fits`** | Total estimator `.fit()` executions | $1$ per iteration | $(K \times 3) + 1$ refit |
| **`decision_iterations`** | Agent decision-result cycles | $1$ per iteration | $1$ per iteration |

The experimental protocol budget applies strictly to **`actual_evaluations`**. Underlying CV fits are recorded for computational transparency and resource tracking, but do not artificially compress the candidate evaluation budget.

---

## 5. Tuning Path

When `tune=True`:
1. `RandomizedSearchCV` evaluates $K$ hyperparameter candidates across $C=3$ CV folds.
2. In [`stratml/execution/pipelines/ml_pipeline.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/execution/pipelines/ml_pipeline.py), `n_iter` is capped by `config.max_evaluations`:
   ```python
   if getattr(config, "max_evaluations", None) is not None:
       n_iter = max(1, min(10, int(config.max_evaluations)))
   ```
3. `MLPipelineResult` records:
   - `eval_count = n_candidates`
   - `fit_count = (n_candidates * n_splits) + refit_count`
4. The orchestrator updates:
   - `self.actual_evaluations += evals_this_iter`
   - `self.actual_fits += fits_this_iter`
5. If the remaining budget was 4, `n_iter` evaluates at most 4 candidates, ensuring `actual_evaluations` hits the budget ceiling exactly without overrun.

---

## 6. Duplicate Evaluation Handling

- **Rule**: If the decision engine or coordinator proposes a candidate configuration identical to one previously evaluated, that evaluation executes and counts as **1 model evaluation** towards `actual_evaluations` and the budget ceiling.
- **Tracking**: The decision engine records repeated configurations in `self._repeated_configs`, which is exposed in `StateSearch.repeated_configs`.
- **Rationale**: Silently skipping duplicate configurations without consuming budget would artificially expand the effective search budget beyond the frozen protocol ceiling.

---

## 7. Failed Evaluation Handling

- **Rule**: An attempted candidate evaluation that fails during pipeline execution (e.g. numerical crash, solver non-convergence, or resource exhaustion) **consumes 1 model evaluation attempt from the budget**:
  ```python
  if eval_failed:
      fits_this_iter = 1
      evals_this_iter = 1
  ```
- **Guarantees**:
  1. Failed evaluations increment `actual_evaluations += 1` and `actual_fits += 1`.
  2. The failure is converted into an `ExperimentResult` with `failed=True`, `status="failed"`, and worst-case metrics, then dispatched to the decision engine.
  3. The decision engine receives the failure signal, updates state, and logs the outcome.
  4. **No infinite retry loops**: Even if every model evaluation fails, the orchestrator terminates deterministically when `actual_evaluations == evaluation_budget` with `termination_reason = "budget_exhausted"`.

---

## 8. Termination Handling

Every run deterministically resolves to one of the following explicit termination reasons:

1. **`budget_exhausted`**:
   - `actual_evaluations >= evaluation_budget` OR `total_runtime >= time_budget`.
   - The search reached the frozen protocol ceiling.
2. **`agent_terminated`**:
   - `actual_evaluations < evaluation_budget`.
   - The Decision Engine explicitly emitted `action_type == "terminate"` (e.g. convergence achieved, signals satisfied).
3. **`execution_failure`**:
   - Fatal, unrecoverable exception during dataset loading, profiling, or pipeline infrastructure setup before completed evaluations.

---

## 9. Manifest Changes

The canonical experiment manifest ([`outputs/<run_id>/manifest.json`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/outputs)) and computational budget accounting ([`budget_accounting.json`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/outputs)) now record:

```json
{
  "evaluation_budget": 20,
  "configured_budget": 20,
  "actual_evaluations": 20,
  "actual_fits": 20,
  "decision_iterations": 20,
  "termination_reason": "budget_exhausted",
  "budget": {
    "configured_budget": {
      "evaluation_budget": 20,
      "configured_budget": 20,
      "max_iterations": 20,
      "timeout_per_run_seconds": 300.0,
      "tune": false,
      "budget_type": "paper_standard"
    },
    "actual_consumption": {
      "decision_iterations": 20,
      "model_evaluations": 20,
      "actual_evaluations": 20,
      "model_fits": 20,
      "actual_fits": 20,
      "runtime_seconds": 1.4523,
      "timeout_triggered": false
    },
    "termination_reason": "budget_exhausted"
  }
}
```

---

## 10. Tests Added

A dedicated unit test suite has been created in [`tests/unit/test_budget_enforcement.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/tests/unit/test_budget_enforcement.py):

| Test | Expected | Actual | Status |
| :--- | :--- | :--- | :--- |
| **Test A: Budget = 10** | Never exceeds 10 model evaluations; terminates at 10 | `actual_evaluations == 10`, `termination_reason == "budget_exhausted"` | PASS |
| **Test B: Budget = 20** | Never exceeds 20 model evaluations; terminates at 20 | `actual_evaluations == 20`, `termination_reason == "budget_exhausted"` | PASS |
| **Test C: Early Termination** | Terminates before ceiling when agent emits terminate | `actual_evaluations == 3 < 10`, `termination_reason == "agent_terminated"` | PASS |
| **Test D: Budget Exhaustion** | Automatically terminates at exactly configured ceiling | `actual_evaluations == 4`, `termination_reason == "budget_exhausted"` | PASS |
| **Test E: Duplicate Config** | Repeated configuration consumes evaluation budget | `actual_evaluations == 3`, `repeated_configs >= 1` | PASS |
| **Test F: Failed Evaluation** | Failure consumes budget attempt; no retry loops | `actual_evaluations == 3`, terminates with `budget_exhausted` | PASS |
| **Test G: tune=False** | Evaluations == fits == iterations | `actual_evaluations == actual_fits == decision_iterations == 3` | PASS |
| **Test H: tune=True** | Model fits exceed model evaluations due to CV | `actual_evaluations == 5`, `actual_fits == 16` ($16 > 5$) | PASS |
| **Test I: Iterations != Evaluations** | Decision iterations differ from model evaluations | `decision_iterations == 1`, `actual_evaluations == 8` | PASS |
| **Test J: Manifest Records** | Manifest & budget file record all budget & termination fields | All fields verified in `manifest.json` & `budget_accounting.json` | PASS |

---

## 11. Dataset Smoke Tests

Three benchmark ARFF datasets from the frozen 20-dataset corpus were evaluated with budget enforcement enabled:

| Dataset | Benchmark Task | Target | Configured Budget | Actual Evaluations | Actual Fits | Termination Reason | Manifest Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **australian.arff** | Binary Classification | `A15` | 2 | 2 | 2 | `budget_exhausted` | Verified |
| **vehicle.arff** | Multiclass Classification | `Class` | 2 | 2 | 2 | `budget_exhausted` | Verified |
| **boston.arff** | Regression | `MEDV` | 2 | 2 | 2 | `budget_exhausted` | Verified |

---

## 12. Remaining Issues

None. All budget semantics, ceiling enforcement points, failure consumption rules, tuning paths, and manifest outputs are verified and consistent with the frozen protocol.

---

## 13. Files Modified

1. [`stratml/execution/schemas.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/execution/schemas.py) — Added `eval_count` and `fit_count` to `ExperimentResult`; added `max_evaluations` to `ExperimentConfig`.
2. [`stratml/execution/result_builder.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/execution/result_builder.py) — Propagated `status`, `failed`, `eval_count`, `fit_count` into `ExperimentResult`.
3. [`stratml/execution/config/experiment_config_builder.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/execution/config/experiment_config_builder.py) — Passed `max_evaluations` into `ExperimentConfig`.
4. [`stratml/execution/pipelines/ml_pipeline.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/execution/pipelines/ml_pipeline.py) — Capped `RandomizedSearchCV` `n_iter` by remaining evaluation budget.
5. [`stratml/execution/pipelines/dl_pipeline.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/execution/pipelines/dl_pipeline.py) — Added `fit_count` and `eval_count` to `DLPipelineResult`.
6. [`stratml/core/schemas.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/core/schemas.py) — Added `actual_evaluations` and `actual_fits` to `StateResources`; added `evaluation_budget` to `StateConstraints`.
7. [`stratml/decision/state/state_builder.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/decision/state/state_builder.py) — Accepted and bound `evaluation_budget`, `actual_evaluations`, and `actual_fits`.
8. [`stratml/decision/engine.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/decision/engine.py) — Added `evaluation_budget` support, properties for `evaluation_budget`, `max_iterations`, and `configured_budget`, tracked actual evaluations and fits.
9. [`stratml/orchestration/orchestrator.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/orchestration/orchestrator.py) — Implemented pre-evaluation hard ceiling enforcement, failure handling with evaluation accounting, remaining budget propagation to tuning, `termination_reason` taxonomy, and canonical manifest/budget fields.
10. [`stratml/cli/config.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/cli/config.py) — Added `evaluation_budget` to `DEFAULT_CONFIG` and added `--budget` support.
11. [`stratml/cli/main.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/cli/main.py) — Bound `evaluation_budget` in CLI execution pipeline.
12. [`stratml/cli/commands/run.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/stratml/cli/commands/run.py) — Passed `evaluation_budget` to `DecisionEngine` and `ExecutionOrchestrator`.
13. [`tests/unit/test_budget_enforcement.py`](file:///mnt/c/Users/Atharva%20Kulkarni/Desktop/Programming/ML%20PROJECTS/MLI%20%28CLI%29/multi-agent-auto-ml/tests/unit/test_budget_enforcement.py) — Comprehensive test suite covering Tests A through J and ARFF smoke tests.

---

## 14. Protocol Impact

Did the frozen experimental protocol change?

**NO.**

The frozen protocol datasets, task types, primary and secondary metrics, budget ceilings (10 and 20 evaluations), ablations, and baseline models remain unaltered.

---

## 15. Verdict

**BUDGET PHASE COMPLETE**
