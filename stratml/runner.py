"""
stratml/runner.py
-----------------
Unified experiment runner dispatching StratML and baseline systems (e.g. Random Search)
through a standardized experimental protocol harness.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from stratml.baselines.random_search import RandomSearchEngine
from stratml.core.schemas import CanonicalExperimentResult
from stratml.decision.engine import DecisionEngine
from stratml.execution.schemas import SplitConfig
from stratml.orchestration.orchestrator import ExecutionOrchestrator


def run_experiment(
    dataset_path: str,
    target_column: str,
    system: str = "stratml",
    budget: int = 10,
    seed: int = 42,
    time_budget: Optional[float] = None,
    tune: bool = False,
    run_id: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    resolved_config: Optional[dict] = None,
    allowed_models: Optional[list[str]] = None,
    split_method: str = "stratified",
    test_size: float = 0.2,
    val_size: float = 0.1,
    llm_mode: Optional[str] = None,
    enable_meta_memory: bool = True,
    enable_value_model: bool = True,
) -> CanonicalExperimentResult:
    """
    Execute an AutoML experiment for the specified system under the frozen experimental protocol.

    Supported systems:
      - "stratml"
      - "random_search"
    """
    sys_clean = str(system).strip().lower()
    if sys_clean not in ("stratml", "random_search"):
        raise ValueError(
            f"Unsupported system '{system}'. Must be 'stratml' or 'random_search'."
        )

    dataset_name = Path(dataset_path).stem
    if run_id is None:
        prefix = "rs" if sys_clean == "random_search" else "stratml"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}_{dataset_name}_{timestamp}_{uuid.uuid4().hex[:6]}"

    split_cfg = SplitConfig(
        method=split_method,
        test_size=test_size,
        val_size=val_size,
        random_seed=seed,
    )

    if sys_clean == "random_search":
        engine = RandomSearchEngine(
            evaluation_budget=budget,
            seed=seed,
            run_id=run_id,
            time_budget=time_budget,
            allowed_models=allowed_models,
        )
        send_profile = engine.receive_profile
        send_result = engine.receive_result
    else:
        engine = DecisionEngine(
            evaluation_budget=budget,
            max_iterations=budget,
            time_budget=time_budget,
            allowed_models=allowed_models,
            run_id=run_id,
            seed=seed,
            llm_mode=llm_mode,
            enable_meta_memory=enable_meta_memory,
            enable_value_model=enable_value_model,
        )
        send_profile = engine.receive_profile
        send_result = engine.receive_result

    orchestrator = ExecutionOrchestrator(
        send_profile=send_profile,
        send_result=send_result,
        split_config=split_cfg,
        time_budget=time_budget,
        run_id=run_id,
        log=log,
        tune=tune,
        evaluation_budget=budget,
        max_iterations=budget,
        resolved_config=resolved_config,
        system=sys_clean,
    )

    result = orchestrator.run(dataset_path, target_column)
    return result
