"""
DAG Executor — 参考 DeepEye DAG scheduler + ExecutionEngine

支持：
- 基于依赖关系的拓扑排序
- 并行执行就绪步骤（ThreadPoolExecutor）
- 依赖结果传递
- 步骤状态跟踪
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("DAGExecutor")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGStep:
    """DAG 执行步骤"""
    step_id: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str = ""
    # executor context
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)

    def is_ready(self, completed: set[str]) -> bool:
        return all(dep in completed for dep in self.depends_on)


@dataclass
class DAGPlan:
    """DAG 执行计划"""
    steps: list[DAGStep] = field(default_factory=list)
    step_map: dict[str, DAGStep] = field(default_factory=dict)

    def __post_init__(self):
        if not self.step_map:
            self.step_map = {s.step_id: s for s in self.steps}

    def get_step(self, step_id: str) -> DAGStep | None:
        return self.step_map.get(step_id)

    def get_ready_steps(self, completed: set[str]) -> list[DAGStep]:
        ready: list[DAGStep] = []
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.is_ready(completed):
                ready.append(step)
        return ready

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)

    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def get_done_results(self) -> dict[str, Any]:
        return {s.step_id: s.result for s in self.steps if s.status == StepStatus.DONE}


@dataclass
class DAGExecutionConfig:
    max_concurrency: int = 4
    dependency_timeout: float = 600.0
    fail_fast: bool = False  # 任一步骤失败则立即停止


@dataclass
class DAGExecutionResult:
    success: bool
    step_results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    steps_completed: int = 0
    steps_failed: int = 0


class DAGExecutor:
    """DAG 执行引擎

    用法::

        plan = DAGPlan(steps=[
            DAGStep(step_id="1", description="Read schema"),
            DAGStep(step_id="2", description="Query", depends_on=["1"]),
        ])
        executor = DAGExecutor(config=DAGExecutionConfig(max_concurrency=4))

        def step_handler(step: DAGStep, dep_results: dict) -> Any:
            # Execute the step using tools
            return tool_result

        result = executor.execute(plan, step_handler)
    """

    def __init__(self, config: DAGExecutionConfig | None = None):
        self.config = config or DAGExecutionConfig()

    def execute(
        self,
        plan: DAGPlan,
        step_handler: Callable[[DAGStep, dict[str, Any]], Any],
        on_step_start: Callable[[DAGStep], None] | None = None,
        on_step_end: Callable[[DAGStep, Any], None] | None = None,
    ) -> DAGExecutionResult:
        """执行 DAG 计划。

        Args:
            plan: DAG 执行计划
            step_handler: 步骤执行函数 (step, dep_results) -> result
            on_step_start: 步骤开始回调
            on_step_end: 步骤结束回调 (step, result)
        """
        completed: set[str] = set()
        failed: set[str] = set()
        step_results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        steps_completed = 0
        steps_failed = 0

        # 验证依赖关系
        step_ids = {s.step_id for s in plan.steps}
        for step in plan.steps:
            unresolved = set(step.depends_on) - step_ids
            if unresolved:
                raise ValueError(f"Step {step.step_id} depends on unknown steps: {unresolved}")

        while not plan.is_complete():
            ready = plan.get_ready_steps(completed)

            if not ready:
                if plan.has_failures() and self.config.fail_fast:
                    break
                # 检查是否有步骤在运行（waiting for async, but we're sync here）
                running = [s for s in plan.steps if s.status == StepStatus.RUNNING]
                if not running:
                    # Deadlock: no ready steps, no running steps, plan not complete
                    pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                    if pending:
                        logger.error(
                            "Deadlock detected: %d pending steps with unsatisfied dependencies",
                            len(pending),
                        )
                    break
                continue

            # 并行执行就绪步骤
            if len(ready) == 1:
                # 单步骤直接执行
                results = {ready[0].step_id: self._execute_single(
                    ready[0], step_handler, completed, step_results, on_step_start, on_step_end,
                )}
            else:
                results = self._execute_parallel(
                    ready, step_handler, completed, step_results, on_step_start, on_step_end,
                )

            # 更新状态
            for step_id, (ok, result_or_error) in results.items():
                step = plan.get_step(step_id)
                if step is None:
                    continue
                if ok:
                    step.status = StepStatus.DONE
                    step.result = result_or_error
                    step_results[step_id] = result_or_error
                    completed.add(step_id)
                    steps_completed += 1
                else:
                    step.status = StepStatus.FAILED
                    step.error = result_or_error
                    errors[step_id] = result_or_error
                    failed.add(step_id)
                    steps_failed += 1
                    if self.config.fail_fast:
                        break

                logger.info(
                    "DAG step %s: %s", step_id,
                    "DONE" if step.status == StepStatus.DONE else f"FAILED: {step.error[:80]}",
                )

        return DAGExecutionResult(
            success=not plan.has_failures(),
            step_results=step_results,
            errors=errors,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
        )

    def _execute_single(
        self,
        step: DAGStep,
        handler: Callable,
        completed: set[str],
        step_results: dict[str, Any],
        on_start: Callable | None,
        on_end: Callable | None,
    ) -> tuple[bool, Any]:
        step.status = StepStatus.RUNNING
        if on_start:
            on_start(step)
        try:
            dep_results = {dep: step_results[dep] for dep in step.depends_on if dep in step_results}
            result = handler(step, dep_results)
            if on_end:
                on_end(step, result)
            return (True, result)
        except Exception as exc:
            if on_end:
                on_end(step, None)
            return (False, str(exc))

    def _execute_parallel(
        self,
        steps: list[DAGStep],
        handler: Callable,
        completed: set[str],
        step_results: dict[str, Any],
        on_start: Callable | None,
        on_end: Callable | None,
    ) -> dict[str, tuple[bool, Any]]:
        """负责并行执行一批就绪的步骤, 即所有依赖都已满足的步骤"""
        results: dict[str, tuple[bool, Any]] = {}
        max_workers = min(self.config.max_concurrency, len(steps))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for step in steps:
                step.status = StepStatus.RUNNING # 标记步骤为运行中
                if on_start:
                    on_start(step) # 执行开始回调 --> 如日志记录
                dep_results = {dep: step_results[dep] for dep in step.depends_on if dep in step_results} # 收集该步骤依赖的前置步骤结果
                futures[executor.submit(handler, step, dep_results)] = step.step_id # 提交任务到线程池 TODO: 这里有bug, 一个简单的步骤比如：list_content 提交到react的时候 react会完整的走完直至得到answer，这也是当前系统为什么慢的原因

            for future in as_completed(futures):
                step_id = futures[future]
                try:
                    result = future.result()
                    results[step_id] = (True, result)
                except Exception as exc:
                    results[step_id] = (False, str(exc))

        return results


# ============================================================
# Utility: build DAGPlan from plan JSON
# ============================================================

def build_dag_plan_from_json(steps: list[dict]) -> DAGPlan:
    """从 JSON 步骤列表构建 DAGPlan。

    每个 step dict 格式::

        {
            "step": 1,           # 或 step_id
            "description": "...",
            "depends_on": [1],   # 依赖的 step 编号
            "action": "tool_name",
            "details": "..."
        }
    """
    dag_steps: list[DAGStep] = []
    for item in steps:
        step_id = str(item.get("step_id", item.get("step", "?")))
        deps_raw = item.get("depends_on", [])
        depends_on = [str(d) for d in deps_raw] if deps_raw else []
        dag_steps.append(DAGStep(
            step_id=step_id,
            description=item.get("description", ""),
            depends_on=depends_on,
            action=item.get("action", ""),
            action_input=item.get("details", {}),
        ))
    return DAGPlan(steps=dag_steps)
