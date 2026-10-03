# Experiment Compatibility Audit

**Evaluation Protocol:** *"Beyond Configuration Search: A State-Aware Framework for AutoML Experimentation"*  
**Audit Date:** 2026-10-03  
**Auditor:** Antigravity Autonomous Agent (Google DeepMind)  
**Status:** Audit Complete — Code Modifications Deferred Pending Review  

---

## 1. Executive Summary

This audit evaluates the current StratML codebase against the frozen experimental protocol for the StratML paper. The audit covers all 20 evaluation datasets, metric computation and propagation, evaluation budget accounting, baseline readiness (Random Search and Auto-sklearn), ablation mechanics, experience modes, warm-start isolation, seed propagation, test-set holdout boundary, RQ3 observability, trajectory logging, provenance manifests, failure handling, and runtime isolation.

### Key Audit Findings:
1. **Critical Metric Path Discrepancy:** The frozen protocol mandates ROC-AUC (binary classification), Log Loss (multiclass classification), and RMSE/MAE/R² (regression). The current implementation only computes Accuracy and F1 for classification, and MSE/RMSE/R² for regression (omitting MAE). Crucially, `ExecutionOrchestrator` hardcodes `primary = accuracy if ... else r2`, completely ignoring ROC-AUC, Log Loss, and RMSE.
2. **Inverted Minimization Optimization:** Optimization direction assumes maximization throughout the orchestrator, history trajectory, value model backfill, performance agent, and signal tools. When evaluating Log Loss or RMSE, higher loss/error is rewarded as "improvement," and lower error is penalized as "degradation."
3. **Budget Semantic Mismatch:** The protocol defines budgets in terms of **model evaluations** (10 and 20). The orchestrator loop operates on **decision iterations** and defaults to a legacy budget of `max_iterations = 5` without enforcing an evaluation cap in `orchestrator.run()`.
4. **Baselines Not Implemented / Incompatible:** Random Search as a controlled AutoML baseline is completely absent. Auto-sklearn is incompatible with the project's Python 3.12 and `scikit-learn 1.8.0` environment and lacks an evaluation harness.
5. **Ablation Gap:** `StratML − Evaluator Feedback` has no configuration flag or code path to disable; evaluator auditing is unconditionally executed.
6. **Transfer Corpus Contamination Risk:** Live runs in non-independent modes append decisions and reflections directly to `runs/decision_logs/decision_dataset.csv` and `runs/decision_logs/meta_memory.jsonl`, mutating the historical corpus during evaluation runs.
7. **Holdout Test Isolation Verified (PASS):** Test splits are strictly isolated until post-loop evaluation.

**Verdict:** **NOT READY FOR PILOT** (Detailed blocking issues and required changes are documented in Sections 17 and 18).

---

## 2. Dataset Compatibility

The protocol specifies 20 benchmark datasets from OpenML (10 Classification, 10 Regression). None of the 20 datasets currently reside in `data/raw/`, and the `openml` Python client is not installed. 

### Compatibility Matrix

| Dataset | OpenML Task (DID) | Problem Subtype | Load | Preprocess | Models | Metric Compatible | Status | Key Issues & Bottlenecks |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Australian** | 146818 (40981) | Binary Class (690 x 14) | PASS | PASS | PASS | FAIL | **PARTIAL** | Missing ROC-AUC calculation; target `A15` needs binary proba output. |
| **blood-transfusion** | 359955 (1464) | Binary Class (748 x 4) | PASS | PASS | PASS | FAIL | **PARTIAL** | Missing ROC-AUC calculation; proba output missing in pipeline. |
| **credit-g** | 168757 (31) | Binary Class (1000 x 20) | PASS | PASS | PASS | FAIL | **PARTIAL** | String target (`good`/`bad`) throws ValueError in ROC-AUC without binarizer; 14 categorical features. |
| **vehicle** | 190146 (54) | Multiclass (846 x 18, 4 cls) | PASS | PASS | PASS | FAIL | **FAIL** | Primary metric Log Loss missing; multi-class probability output not captured; minimization unhandled. |
| **kc1** | 359962 (1067) | Binary Class (2109 x 21) | PASS | PASS | PASS | FAIL | **PARTIAL** | Missing ROC-AUC calculation; string/bool target binarization required. |
| **phoneme** | 168350 (1489) | Binary Class (5404 x 5) | PASS | PASS | PASS | FAIL | **PARTIAL** | Missing ROC-AUC calculation. |
| **jasmine** | 168911 (41143) | Binary Class (2984 x 144) | PASS | PASS | PARTIAL | FAIL | **PARTIAL** | 137 categorical features expand dimension > 300; SVC/KNN slow; missing ROC-AUC. |
| **bank-marketing** | 359982 (1461) | Binary Class (45211 x 16) | PASS | PASS | PARTIAL | FAIL | **PARTIAL** | 45k rows; SVC and KNN risk timeout; missing ROC-AUC. |
| **adult** | 359983 (1590) | Binary Class (48842 x 14) | PASS | PASS | PARTIAL | FAIL | **PARTIAL** | 6,465 missing values in categoricals; 48k rows; string targets; missing ROC-AUC. |
| **jannis** | 211979 (41168) | Multiclass (83733 x 54, 4 cls) | PASS | PASS | PARTIAL | FAIL | **FAIL** | 83k rows; Log Loss missing; multiclass proba missing; SVC/KNN prohibitively slow. |
| **abalone** | 359944 (42726) | Regression (4177 x 8) | PASS | PASS | PASS | FAIL | **PARTIAL** | Target is integer ring count; MAE missing; RMSE direction inverted (orchestrator uses R² max). |
| **boston** | 359950 (531) | Regression (506 x 13) | PASS | PASS | PASS | FAIL | **PARTIAL** | MAE missing; RMSE minimization inverted; CHAS binary feature. |
| **Brazilian houses** | 359938 (42688) | Regression (10692 x 12) | PASS | PASS | PASS | FAIL | **PARTIAL** | 4 categoricals; large target values (`total_(BRL)` > 50k) break signal thresholds (`primary >= 0.75`). |
| **elevators** | 359936 (216) | Regression (16599 x 18) | PASS | PASS | PASS | FAIL | **PARTIAL** | Small target scale (~0.002) falsely triggers `underfitting: strong` (signal threshold < 0.60). |
| **house 16H** | 359952 (574) | Regression (22784 x 16) | PASS | PASS | PASS | FAIL | **PARTIAL** | MAE missing; RMSE minimization inverted. |
| **Moneyball** | 167210 (41021) | Regression (1232 x 14) | PASS | PASS | PASS | FAIL | **PARTIAL** | 3,600 missing values across 6 categoricals + 8 numerics; MAE missing; RMSE minimization inverted. |
| **OnlineNewsPopularity** | 359941 (42724) | Regression (39644 x 59) | PASS | PASS | PARTIAL | FAIL | **PARTIAL** | 39k rows; extreme target skew; SVR slow; MAE missing; RMSE minimization inverted. |
| **pol** | 359946 (201) | Regression (15000 x 48) | PASS | PASS | PASS | FAIL | **PARTIAL** | MAE missing; RMSE minimization inverted. |
| **wine quality** | 359935 (287) | Regression (6497 x 11) | PASS | PASS | PASS | FAIL | **PARTIAL** | Overlaps with old demo dataset name; MAE missing; RMSE minimization inverted. |
| **yprop** | 359940 (416) | Regression (8885 x 251) | PASS | PASS | PARTIAL | FAIL | **PARTIAL** | High dimensional (251 features); SVR/Ensembles slow; MAE missing; RMSE minimization inverted. |

---

## 3. Metric Audit

### Protocol Requirements
- **Binary classification:** PRIMARY = ROC-AUC (maximize), SECONDARY = F1 (maximize).
- **Multiclass classification:** PRIMARY = Log Loss (minimize), SECONDARY = F1 (maximize).
- **Regression:** PRIMARY = RMSE (minimize), SECONDARY = MAE (minimize), SECONDARY = R² (maximize).

### Trace of Current Implementation
1. **Schema Definition (`stratml/core/schemas.py`, `stratml/execution/schemas.py`):**
   - `ExperimentMetrics` fields: `accuracy`, `f1_score`, `precision`, `recall`, `train_loss`, `validation_loss`, `mse`, `rmse`, `r2`.
   - **Missing Fields:** `roc_auc`, `log_loss`, `mae`.
   - `SecondaryMetrics` in `StateMetrics` also lacks `roc_auc`, `log_loss`, and `mae`.
2. **Metrics Engine (`stratml/execution/metrics/metrics_engine.py`):**
   - For classification: Only calls `accuracy_score`, `f1_score`, `precision_score`, `recall_score`. It takes `y_pred` (discrete labels) only. It does not accept or compute probabilities for `roc_auc` or `log_loss`.
   - For regression: Only computes `mse`, `rmse`, `r2`. It completely omits `mae`.
3. **Pipeline Output (`stratml/execution/pipelines/ml_pipeline.py`):**
   - `MLPipelineResult` stores `y_val_pred = model.predict(data_split.X_val)` (discrete class labels). It does not store `y_val_proba`.
   - In lines 169–174, it attempts `log_loss(data_split.y_val, model.predict_proba(data_split.X_val))` solely for loss curves. For regression or `SVC(probability=False)`, this throws an exception and silently sets `val_loss = 0.0`.
4. **Execution Orchestrator Metric Selection (`stratml/orchestration/orchestrator.py`):**
   - Line 188–192:
     ```python
     primary = (
         metrics.accuracy
         if metrics.accuracy is not None
         else (metrics.r2 if metrics.r2 is not None else 0.0)
     )
     ```
   - Line 193: `is_best = primary > best_val_score` (hardcodes maximization).
   - Lines 284–288: Post-run holdout test evaluation hardcodes:
     ```python
     primary_test = (
         test_metrics.accuracy
         if test_metrics.accuracy is not None
         else (test_metrics.r2 if test_metrics.r2 is not None else 0.0)
     )
     ```
5. **Decision Engine & State History (`stratml/decision/engine.py`, `state_history.py`):**
   - `engine.py` lines 238–245 defaults `primary_metric = "r2"` for regression and `"accuracy"` for classification.
   - `state_history.py` line 61: `best_score = max(scores)` (assumes higher is better).
   - `state_history.py` line 64: `if current > self._best_score + 1e-6:` (rewards increasing values).
   - `state_builder.py` line 165:
     ```python
     metrics=StateMetrics(
         primary=traj.best_score if optimization_goal == "maximize" else traj.mean_score,
     ```
     (For minimization, it passes `mean_score` instead of `min_score`).
6. **Signal Extraction Tool Assumptions (`stratml/decision/state/signals.py`):**
   - Line 75–82: `uf_strong = primary < 0.60`, `wf_strong = primary >= 0.75`.
   - These hardcoded thresholds assume bounded 0..1 accuracy metrics. On Brazilian houses (RMSE ~ 15,000), `primary >= 0.75` triggers a false `well_fitted` signal. On Elevators (RMSE ~ 0.002), `primary < 0.60` triggers a false `underfitting` signal.
7. **Value Model & Performance Agent Inversion:**
   - In `dataset_builder.py` line 178: `headroom = 1.0 - best` (invalid when `best` is unbounded RMSE).
   - In `performance_agent.py` line 76: `table.get(...) + e.predicted_gain * 0.3`. When optimizing RMSE or Log Loss, an improvement yields a negative delta, which penalizes the action's performance score.

---

## 4. Budget Audit

### Protocol Requirements
- Two evaluation budgets: **10 model evaluations** and **20 model evaluations**.
- Budget defined strictly in **model evaluations**, not decision cycles.

### Audit Findings
1. **Hardcoded Legacy Budget Default:**
   - `stratml/orchestration/orchestrator.py` line 68:
     ```python
     self.max_iterations = max_iterations if max_iterations is not None else 5
     ```
   - `stratml/orchestration/orchestrator.py` lines 366–369:
     ```python
     "target_paper_config": {
         "max_iterations": 5,
         "tune": False,
     }
     ```
   - `stratml/reporting/pdf_builder.py` line 587: hardcodes `"The target paper budget configuration is max_iterations: 5, tune: false."`
2. **Missing Loop Evaluation Enforcement:**
   - In `orchestrator.py` line 134:
     ```python
     while action.action_type != "terminate":
         iteration += 1
     ```
   - The loop in `orchestrator.run()` **never checks `iteration >= self.max_iterations` or `self.actual_evaluations >= budget`**. It relies entirely on Team B to emit `action_type == "terminate"`.
   - In `DecisionEngine`: `remaining = max(0.0, self.max_iterations - result.iteration)`.
     If `engine` is instantiated with default `max_iterations=20` but orchestrator was set to 10, the engine continues until iteration 20 because orchestrator does not stop it.
3. **Iteration vs Evaluation Counting:**
   - `orchestrator.py` tracks both `self.decision_iterations` and `self.actual_evaluations`.
   - When `--tune` is disabled (`tune=False`), each iteration performs 1 model evaluation. Iteration count equals evaluation count.
   - When `tune=True`, `RandomizedSearchCV` executes 10 iterations x 3 folds = 30 fits / 10 evaluations. The engine only increments its iteration counter by 1, allowing evaluation budget overruns.
4. **Failed and Duplicate Evaluations:**
   - Duplicate configurations are logged in `_repeated_configs` and consume a budget iteration.
   - If an evaluation throws an unhandled exception in `ml_pipeline.py`, the run aborts immediately; it is not counted or recovered cleanly.
5. **Manifest Reporting:**
   - `manifest["budget"]["configured_budget"]["max_iterations"]` records configured budget.
   - `manifest["budget"]["actual_consumption"]["model_evaluations"]` records actual evaluations.

---

## 5. Random Search Baseline

### Protocol Requirements
Controlled Random Search baseline sharing datasets, splits, preprocessing, model families, hyperparameter mutation space, metrics, 10/20 budgets, seeds, and test isolation.

### Audit Findings
- **Status:** **NOT IMPLEMENTED**.
- There is no Random Search baseline module, script, or CLI command in the repository.
- `RandomizedSearchCV` in `ml_pipeline.py` is only an internal sub-step for hyperparameter tuning.
- **Required Implementation:** A standalone `stratml/baselines/random_search.py` harness that:
  - Takes the identical `Dataset`, `SplitConfig`, `base_split`, and `profile`.
  - Samples uniformly from `PAPER_CLASSIFICATION_MODELS` / `PAPER_REGRESSION_MODELS` and `get_paper_mutation_space()`.
  - Executes for exactly 10 and 20 evaluations.
  - Computes the identical primary and secondary metrics.
  - Evaluates the best validation model on the holdout test set using identical test isolation.
  - Outputs an equivalent `manifest.json` and `results.json`.

---

## 6. Auto-sklearn Compatibility

### Protocol Requirements
Benchmark against Auto-sklearn under identical data splits, seeds, evaluation budgets, and metrics.

### Audit Findings
- **Status:** **NOT IMPLEMENTED / ENVIRONMENT INCOMPATIBLE**.
- **Python & Dependency Incompatibility:**
  - Auto-sklearn (latest `0.15.0`) requires `scikit-learn < 0.25` / `< 1.0` and Python `<= 3.10`.
  - The repository environment runs Python `3.12` with `scikit-learn==1.8.0`. Auto-sklearn cannot be installed or executed inside this virtual environment.
  - Auto-sklearn is strictly Linux-only (relies on `resource` and daemon forks).
- **Harness Status:** No Auto-sklearn execution or translation wrapper exists in the codebase.
- **Protocol Adaptation Required:**
  - Auto-sklearn must be run in a decoupled external environment (e.g. Docker container or separate Python 3.10 virtual environment).
  - An adapter script must feed the pre-split train/val/test CSVs to ensure identical holdout evaluation.
  - Auto-sklearn budget parameters (`time_left_for_this_task` and `per_run_time_limit`) must be calibrated to approximate 10 and 20 evaluation allocations, noting that Auto-sklearn natively optimizes wall-clock time rather than a fixed discrete evaluation count.

---

## 7. Ablation Audit

The frozen protocol requires four independent configurations:
1. **Full StratML**
2. **StratML − MetaMemory**
3. **StratML − Value Model**
4. **StratML − Evaluator Feedback**

### Trace by Ablation

| Ablation | Implementation Mechanism | Disabling Code Path | Clean Disabling? | Influence Leaks? | Manifest Recorded? | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MetaMemory OFF** | `enable_meta_memory=False` | `engine.py` line 247: skips retrieval; line 536: skips record | **YES** | None. Returns empty bootstrap advice. | Recorded in `manifest["warm_start"]["meta_memory_enabled"]` and condition string. | **PASS** |
| **Value Model OFF** | `enable_value_model=False` | `engine.py` line 427: yields neutral predictions (`gain=0.05, cost=0.5`) | **YES** | Coordinator receives constant stub; weight learning does not activate. | Recorded in `manifest["warm_start"]["value_model_enabled"]` and condition string. | **PASS** |
| **Evaluator Feedback OFF** | `enable_evaluator_feedback` | **DOES NOT EXIST** | **NO** | Unconditionally executed (`evaluator_agent.audit` always called). | Omitted from manifest schema and CLI flags. | **NOT IMPLEMENTED** |

**Ablation Independence Note:** MetaMemory and Value Model are properly decoupled in `engine.py`. However, disabling Evaluator Feedback is impossible without code modification.

---

## 8. Experience Mode Audit

The protocol specifies three experience modes:
1. **Cold:** Clean slate, no historical experience.
2. **Transfer:** Historical corpus from non-evaluation datasets.
3. **Continual:** Sequential runs on the same evaluation dataset where completed runs carry forward.

### Trace of History Modes
- **Cold (Configured as `history_mode="independent"`):**
  - Isolates corpus path to `outputs/<run_id>/decision_logs/decision_dataset.csv`.
  - `_meta_memory.retrieve_similar_actions` explicitly returns `[]`.
  - **Leakage Issue:** `dataset_builder.record()` line 131 unconditionally appends cold decisions to the unified corpus `runs/decision_logs/decision_dataset.csv`. This contaminates the default transfer corpus with cold evaluation data.
- **Transfer (`history_mode="transfer"`):**
  - Reads `runs/decision_logs/decision_dataset.csv` and `meta_memory.jsonl`.
  - Filters by `dataset_id != current_dataset_id` and `dataset_fingerprint != current_dataset_fingerprint`.
  - **Mutation Issue:** Evaluation runs in transfer mode append their new decisions and reflections to `runs/decision_logs/`. Subsequent runs in a transfer matrix will see data from prior evaluation datasets.
- **Continual (`history_mode="continual"`):**
  - Filters by `dataset_id == current_dataset_id` and carries experience across sequential runs. Verified working.

---

## 9. Warm-Start Isolation

### Protocol Requirements
- Historical corpus constructed exclusively from non-evaluation datasets.
- Corpus must be frozen before final evaluation begins.
- Manifest must record the start snapshot hash of the corpus before any evaluation mutation.
- No evaluation or test data may enter the warm-start corpus.

### Audit Findings
- **Start Snapshot Mechanism (P1-28 / P1-30):** **PASS**.
  - `orchestrator.run()` calls `engine.capture_start_snapshot(lock=True)` before dataset loading, profiling, or model fitting.
  - The start snapshot hash is saved to `warm_start_start_snapshot.json` and persisted in `manifest.json`. Post-run file mutations do not alter the manifest record.
- **Corpus Location & Separation:** **FAIL**.
  - The system points warm-start directly at `runs/decision_logs/decision_dataset.csv` and `runs/decision_logs/meta_memory.jsonl`.
  - Because active runs write to this exact location, the transfer corpus is mutated during evaluation.
  - The 200–300 observation transfer corpus does not exist yet, and `scripts/seed_value_model.py` includes `wine_quality_red.csv` (which overlaps with evaluation dataset `wine quality`).

---

## 10. Seed / Reproducibility Audit

### Protocol Requirements
3 independent seeds per condition. Complete propagation of seed across all stochastic components.

### Audit Trace
1. **Dataset Split:** `train_test_split(..., random_state=config.random_seed)` in `splitter.py` -> **PASS**.
2. **Preprocessing:** `SMOTE(random_state=seed)` and `RandomUnderSampler(random_state=seed)` in `preprocessor.py` -> **PASS**.
3. **Classical Models:** `hp["random_state"] = seed` in `ml_pipeline.py` lines 126–128 -> **PASS**.
4. **Value Model:** `RandomForestRegressor(..., random_state=eff_seed)` in `value_model.py` line 197 -> **PASS**.
5. **Action Selector (Epsilon-Greedy):** `select(..., rng=self._rng)` where `self._rng = random.Random(seed)` -> **PASS**.
6. **LLM Deliberation:** `ChatGroq(model="...", temperature=0.2)` in `action_generator.py` and agents has **no seed parameter passed to Groq API**. Groq API requests introduce stochastic variance unpinned to the run seed -> **PARTIAL / UNCONTROLLED STOCHASTICITY**.
7. **Manifest Logging:** Run seed recorded in `manifest["seed"]` and `manifest["evaluation_configuration"]["random_seed"]` -> **PASS**.

---

## 11. Test-Set Isolation Audit

### Protocol Requirements
Test set must NEVER be used during candidate generation, candidate ranking, hyperparameter selection, model selection, Value Model training, MetaMemory decision making, termination, or evaluator feedback.

### Audit Data Flow Trace
1. In `orchestrator.run()` line 113, `split_dataset` separates `base_split.X_test` and `y_test`.
2. Inside the iteration loop (lines 151–153), `apply_preprocessing(..., transform_test=False)` explicitly preserves `X_test` and `y_test` untouched in memory.
3. `run_ml_pipeline` only receives and fits on `clean_split.X_train` and evaluates on `clean_split.X_val`.
4. `compute_metrics` is called only on `clean_split.y_val`.
5. `build_experiment_result` receives only validation metrics.
6. `DecisionEngine`, `ValueModel`, `MetaMemory`, and `EvaluatorAgent` consume only `ExperimentResult` containing validation scores.
7. `X_test` and `y_test` are accessed exclusively after the `while` loop terminates (lines 266–283), where `best_model.predict(test_split.X_test)` is evaluated once for final reporting.
- **Verdict:** **PASS** (Zero leakage detected).

---

## 12. RQ3 Logging Audit

### Protocol Measures
1. **Valid Action Rate:** `valid executable decisions / total decisions`
2. **Constraint Violation Rate:** `constraint violations / total decisions`
3. **Trajectory Completeness:** `state → candidate/action → decision → execution → outcome → next-state`
4. **Decision Source Attribution:** `bootstrap | rule | llm | value_model | hybrid | fallback`
5. **State-Transition Integrity:** next state corresponds to executed outcome.

### Audit Findings

| Measure | Logged Raw Data | Location | Machine Readable? | Retrospectively Computable? | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Valid Action Rate** | Execution status (`completed` vs `failed`) and evaluator verdict (`correct` vs `flawed`) | `trajectory.jsonl`, `*_*.json` | YES | YES (`status == 'completed'`) | **PASS** |
| **Constraint Violation Rate** | Omitted. No explicit constraint violation event or counter is logged | None | NO | NO (Must be inferred post-hoc against `allowed_models`) | **PARTIAL** |
| **Trajectory Completeness** | Iteration, state_id, candidates, decision, parameters, execution outcome, evaluator verdict, next_state_id | `trajectory.jsonl` | YES | YES | **PASS** |
| **Decision Source Attribution** | `reason.source` (`bootstrap`, `rule`, `llm`, `value_model`, `hybrid`, `fallback`) and `selection_mode` | `trajectory.jsonl`, `*_*.json` | YES | YES | **PASS** |
| **State-Transition Integrity** | `next_state_id` linked via `decision_logger.update_outcome` | `trajectory.jsonl`, `*_*.json` | YES | YES | **PASS** |

---

## 13. Trajectory Audit

### Trace of `trajectory.jsonl` and Artifacts
- Each cycle generates `outputs/<run_id>/decision_logs/{run_id}_{iteration:04d}.json`.
- When execution completes, `decision_logger.update_outcome()` appends `execution_result`, `evaluator_result`, and `next_state_id`.
- `_sync_trajectory_log()` compiles a sequential JSONL artifact:
  - `iteration`: int
  - `state_id`: `f"{run_id}_{iteration}"`
  - `candidate_set`: list of all generated / ranked candidate actions
  - `selected_action`: chosen `ActionDecision`
  - `executed_configuration`: hyperparameters and model name
  - `execution_outcome`: metrics, status, gain, success flag
  - `evaluator_result`: validity, risk, verdict, audit rationale
  - `next_state_id`: pointer to subsequent state
- Runner-up candidate rankings are preserved in `ranked_candidates` within each record.
- **Verdict:** **PASS**.

---

## 14. Manifest / Provenance Audit

### Manifest Completeness Checklist

| Required Field | Manifest Location | Present? | Details / Issues |
| :--- | :--- | :--- | :--- |
| **Dataset ID** | `manifest["dataset"]["name"]` | PARTIAL | Logs file stem (e.g. `australian`); does not log OpenML task/dataset ID. |
| **Dataset Fingerprint** | `manifest["dataset_fingerprint"]` | PASS | 16-hex content fingerprint. |
| **Task Type** | `manifest["task"]` | PARTIAL | Logs `classification` or `regression`; does not distinguish binary vs multiclass. |
| **Seed** | `manifest["seed"]` | PASS | Integer random seed. |
| **Run ID** | `manifest["run_id"]` | PASS | Unique run string. |
| **Model Space** | `manifest["model_space"]` | PASS | List of permitted model names. |
| **Action Space** | `manifest["action_space"]` | PASS | Permitted high-level action types. |
| **Mutation Space** | `manifest["hyperparameter_mutation_space"]` | PASS | Dictionary of mutation grids. |
| **Evaluation Budget** | `manifest["budget"]["configured_budget"]["max_iterations"]` | PASS | Configured budget limit. |
| **Actual Evaluation Count**| `manifest["budget"]["actual_consumption"]["model_evaluations"]`| PASS | Evaluated model count. |
| **Model Fit Count** | `manifest["budget"]["actual_consumption"]["model_fits"]` | PASS | Fit count including CV folds. |
| **Tuning Configuration** | `manifest["budget"]["configured_budget"]["tune"]` | PASS | Boolean flag. |
| **LLM Configuration** | `manifest["llm_configuration"]` | PASS | Provider, model, temperature, prompts. |
| **Ablation Flags** | `manifest["resolved_experiment_config"]["ablations"]` | PARTIAL | Has `enable_meta_memory`, `enable_value_model`; lacks `enable_evaluator_feedback`. |
| **Experience Mode** | `manifest["warm_start"]["history_mode"]` | PASS | `independent`, `continual`, `transfer`. |
| **Warm-Start Corpus Path** | `manifest["warm_start"]["decision_corpus"]["path"]` | PASS | Resolved corpus path. |
| **Warm-Start Snapshot Hash**| `manifest["start_snapshot_hash"]` | PASS | Pre-experiment SHA-256 hash. |
| **StratML Version / Commit**| `manifest["stratml_version"]` | PASS | Version and git commit hash. |
| **Configuration Hash** | `manifest["config_hash"]` | PASS | SHA-256 hash of resolved configuration. |
| **Evaluation Configuration**| `manifest["evaluation_configuration"]` | PASS | Split ratios, seed, best validation score. |
| **Termination Reason** | `manifest["termination_reason"]` | **FAIL** | **MISSING**. Omitted from manifest root. |

---

## 15. Failure Handling

### Audit Findings
1. **Uncaught Model Fitting Exceptions:**
   - In `orchestrator.py` lines 158–164:
     ```python
     if config.model_type == "ml":
         pipeline_result = run_ml_pipeline(config, clean_split)
     ```
   - Execution is not wrapped in a `try...except`. If a model raises `ValueError` (e.g. invalid hyperparameter, convergence failure, NaN handling), the orchestrator crashes ungracefully.
2. **Missing Failure State Emission:**
   - The system does not construct an `ExperimentResult(status="failed", failed=True)` upon exception.
   - Partial runs leave dangling `*_*.json` logs with `status="pending"`, and `manifest.json` is never written.
3. **Failure Classification:**
   - Evaluator agent classifies decisions as `correct` or `flawed`, but does not differentiate infrastructure/environment failures from algorithmic failures.

---

## 16. Experiment Isolation

### Audit Findings
1. **File System Outputs:** `outputs/<run_id>/` isolates artifacts, logs, and tensorboard runs cleanly per execution -> **PASS**.
2. **Module-Level Global State Contamination:** **FAIL**.
   - `stratml/decision/agents/coordinator_agent.py` maintains module globals: `_CURRENT_WEIGHTS`, `_WEIGHT_UPDATE_HISTORY`, and `_LEARNING_STATE`. If `reset_weight_update_history()` is not called between runs, coordinator weight adaptations bleed into subsequent experiments.
   - `decision_logger._LOG_DIR`, `counterfactual._CF_LOG`, `value_model._DATASET_PATH`, and `meta_memory._MEMORY_FILE` are mutated at runtime via module globals. In concurrent or looped execution, these references collide.
3. **Cross-Run Accumulation:** `dataset_builder.record()` writes to `runs/decision_logs/decision_dataset.csv` even during independent/cold runs.

---

## 17. Blocking Issues

The following items directly block faithful execution of the frozen experimental protocol:

1. **[BLOCKER 1] Incompatible Metric System (Missing ROC-AUC, Log Loss, MAE; Hardcoded Accuracy/R²):** Models are evaluated and optimized against Accuracy or R² rather than the protocol's required primary metrics.
2. **[BLOCKER 2] Inverted Metric Optimization Direction for Minimization:** Minimization metrics (RMSE, Log Loss, MAE) are treated as maximization targets by `orchestrator.py`, `state_history.py`, and `performance_agent.py`.
3. **[BLOCKER 3] Missing Probability Outputs in Pipeline:** Pipeline does not return predicted probability arrays required for ROC-AUC and Log Loss.
4. **[BLOCKER 4] Budget Enforcement Semantic Mismatch:** Orchestrator does not enforce an evaluation cap in `run()` and defaults to `max_iterations = 5` instead of 10 / 20 model evaluations.
5. **[BLOCKER 5] Missing Random Search Baseline:** Complete absence of controlled Random Search harness.
6. **[BLOCKER 6] Auto-sklearn Environment Incompatibility:** Python 3.12 / scikit-learn 1.8.0 incompatibility prevents local Auto-sklearn execution without an external environment wrapper.
7. **[BLOCKER 7] Missing Evaluator Feedback Ablation:** No mechanism or flag exists to disable Evaluator Feedback.
8. **[BLOCKER 8] Transfer Corpus Contamination:** Transfer runs write evaluation observations back to the shared historical corpus files.
9. **[BLOCKER 9] Missing Evaluation Datasets & OpenML Ingestion:** None of the 20 benchmark datasets are downloaded or configured with automated ingestion.
10. **[BLOCKER 10] Missing Manifest Fields:** `termination_reason` and OpenML Task/Dataset IDs are absent from `manifest.json`.
11. **[BLOCKER 11] Uncaught Model Exceptions:** Model training errors crash the orchestrator rather than logging failed iterations.
12. **[BLOCKER 12] State Bleed Across Runs:** Coordinator weight learning state persists across sequential runs.

---

## 18. Required Changes Before Pilot

| Target File / Path | Component / Class / Function | Current Behavior | Protocol Requirement | Status | Exact Change Required | Scientific Validity Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `stratml/execution/schemas.py`, `stratml/core/schemas.py` | `ExperimentMetrics`, `SecondaryMetrics` | Missing `roc_auc`, `log_loss`, `mae` fields | Primary & secondary metrics for all 3 tasks | FAIL | Add `roc_auc: Optional[float] = None`, `log_loss: Optional[float] = None`, `mae: Optional[float] = None` to metric schemas. | **CRITICAL** |
| `stratml/execution/pipelines/ml_pipeline.py` | `MLPipelineResult`, `run_ml_pipeline` | Returns only `y_val_pred` (labels); no probabilities | Probabilities required for ROC-AUC & Log Loss | FAIL | Return `y_val_proba: Optional[np.ndarray]` in `MLPipelineResult`; enable `probability=True` on `SVC`. | **CRITICAL** |
| `stratml/execution/metrics/metrics_engine.py` | `compute_metrics` | Computes only accuracy, f1, precision, recall, mse, rmse, r2 | Compute ROC-AUC (binary), Log Loss (multiclass), MAE (regression) | FAIL | Add `y_proba` parameter; compute `roc_auc_score` with target binarization; compute `log_loss`; compute `mean_absolute_error`. | **CRITICAL** |
| `stratml/orchestration/orchestrator.py` | `ExecutionOrchestrator.run`, `_save_manifest` | Hardcodes `primary = accuracy if ... else r2`; `is_best = primary > best_val_score` | Primary metric determined by task; handle minimize and maximize | FAIL | Dynamically resolve primary metric from task (binary: `roc_auc`, multiclass: `log_loss`, regression: `rmse`); compare using `optimization_goal`. | **CRITICAL** |
| `stratml/decision/state/state_history.py` | `ExperimentHistory.compute_trajectory` | `best_score = max(scores)`; `current > self._best_score` | Support minimization for RMSE and Log Loss | FAIL | Accept `optimization_goal`; compute `min(scores)` when minimizing; mark improvement when `current < best - eps`. | **CRITICAL** |
| `stratml/decision/state/signals.py` | `assess_fitting`, `assess_convergence`, `_rule_based` | Thresholds assume `primary >= 0.75` for well_fitted and `< 0.60` for underfitting | Scale-invariant signal extraction | FAIL | Normalize primary metric delta or branch thresholds based on problem type and optimization goal. | **HIGH** |
| `stratml/decision/agents/performance_agent.py` | `_rule_score` | `scores = table + e.predicted_gain * 0.3` | Negative gain in minimization indicates improvement | FAIL | Invert sign of `predicted_gain` when `optimization_goal == "minimize"`. | **HIGH** |
| `stratml/orchestration/orchestrator.py` | `ExecutionOrchestrator.__init__`, `run` | Defaults `max_iterations = 5`; while loop has no evaluation cap | Enforce discrete model evaluation budgets (10 and 20) | FAIL | Check `self.actual_evaluations >= self.budget_limit` in while loop; update default budgets to 10 and 20. | **CRITICAL** |
| `stratml/baselines/` | **NEW FILE:** `random_search.py` | Not implemented | Fair controlled Random Search baseline | NOT IMPLEMENTED | Implement `RandomSearchBaseline` sharing splits, mutation spaces, budgets, seeds, and metrics. | **CRITICAL** |
| `stratml/decision/engine.py`, `stratml/cli/config.py` | `DecisionEngine`, `build_experiment_config` | Only `enable_meta_memory` and `enable_value_model` exist | Support `StratML − Evaluator Feedback` | NOT IMPLEMENTED | Add `enable_evaluator_feedback: bool = True`; bypass `evaluator_agent.audit` and omit from coordinator ranking when False. | **HIGH** |
| `stratml/decision/engine.py`, `dataset_builder.py` | `DecisionEngine.__init__`, `_append` | Live transfer runs append to `runs/decision_logs/` | Frozen read-only transfer corpus | FAIL | Point transfer mode to a dedicated read-only corpus file; do not append new evaluations to the transfer corpus. | **HIGH** |
| `data/scripts/` | **NEW FILE:** `download_openml_benchmarks.py` | None of 20 benchmark datasets present | Automated ingestion of all 20 frozen tasks | NOT IMPLEMENTED | Add download script using `urllib`/`scikit-learn` to fetch and cache all 20 OpenML task datasets into `data/raw/`. | **HIGH** |
| `stratml/orchestration/orchestrator.py` | `_save_manifest` | Omits `termination_reason` | Provenance completeness | FAIL | Record `manifest["termination_reason"] = "budget_exhausted" | "converged" | "timeout"`. | **MEDIUM** |
| `stratml/orchestration/orchestrator.py` | `run` | `run_ml_pipeline` unhandled | Resilient failure handling | FAIL | Wrap training in `try...except`; construct `ExperimentResult(failed=True, status="failed")` on model crash. | **HIGH** |
| `stratml/decision/agents/coordinator_agent.py` | `rank`, `reset_weight_update_history` | Module globals retain learned weights across runs | Seed & run isolation | FAIL | Reset coordinator weights and learning state on each `DecisionEngine.__init__`. | **HIGH** |

---

## 19. Readiness Verdict

# **NOT READY FOR PILOT**

The codebase contains fundamental metric, optimization-direction, budget-accounting, and baseline incompatibilities with the frozen experimental protocol. Full-matrix experimentation or pilot benchmarking conducted in the current state would yield scientifically invalid results. 

Execution of the fixes detailed in Section 18 must be completed and verified before initiating pilot runs.
