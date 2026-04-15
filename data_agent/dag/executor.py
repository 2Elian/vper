import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from data_agent.core.types import PlanStep, StepResult, StepStatus
from data_agent.dag.graph import DAGGraph


@dataclass
class DAGExecutionConfig:
    """DAG 执行配置"""
    max_concurrency: int = 5               # 最大并行度
    dependency_timeout: float = 600.0      # 依赖等待超时 / s
    check_interval: float = 5.0            # 依赖检查间隔 / s
    pass_dependency_results: bool = True   # 是否传递依赖结果
    retry_on_failure: bool = True          # 失败时是否重试
    max_retries: int = 2                   # 最大重试次数


@dataclass
class DAGExecutionResult:
    """DAG 执行结果"""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    results: Dict[str, StepResult] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "skipped_steps": self.skipped_steps,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "execution_time": round(self.execution_time, 3),
            "success": self.success,
            "error": self.error,
        }


class DAGExecutor:
    """DAG并行执行引擎
        1. Parallel: 所有任务并行（无依赖时）
        2. Sequential: 顺序执行（简单任务）
        3. Hybrid: 基于DAG依赖的混合执行

    核心机制：
        - ThreadPoolExecutor 控制并发
        - 依赖就绪检测（增量检查而非死等）
        - 结果传递（依赖结果注入到后续任务）
        - 失败处理（重试 + 跳过依赖失败的节点）
    """

    def __init__(self, config: DAGExecutionConfig = None):
        self.config = config or DAGExecutionConfig()

    def execute(
        self,
        graph: DAGGraph,
        step_executor: Callable[[PlanStep, Dict[str, StepResult]], StepResult],
    ) -> DAGExecutionResult:
        """执行一个有向无环图
        Args:
            graph: DAG 图
            step_executor: 步骤执行函数 (step, dependency_results) -> StepResult
        Returns:
            DAGExecutionResult
        """
        start_time = time.time()

        if not graph.nodes:
            return DAGExecutionResult(success=True)

        # 先验证 DAG
        is_valid, cycle = graph.validate()
        if not is_valid:
            return DAGExecutionResult(
                error=f"DAG has cycle: {cycle}",
                total_steps=len(graph.nodes),
            )

        # 选择执行模式
        has_dependencies = any(
            len(node.dependencies) > 0 for node in graph.nodes.values()
        )

        if has_dependencies:
            result = self._execute_hybrid(graph, step_executor)
        else:
            result = self._execute_parallel(graph, step_executor)

        result.execution_time = time.time() - start_time
        return result

    def execute_sequential(
        self,
        graph: DAGGraph,
        step_executor: Callable[[PlanStep, Dict[str, StepResult]], StepResult],
    ) -> DAGExecutionResult:
        """
        顺序执行

        参考 Shannon 的 Sequential 模式：
        按拓扑排序顺序执行，可配置是否传递前序结果
        """
        start_time = time.time()
        sorted_order = graph.topological_sort()

        if sorted_order is None:
            return DAGExecutionResult(error="DAG has cycle")

        results: Dict[str, StepResult] = {}
        completed: Set[str] = set()
        prev_result: Optional[StepResult] = None

        for node_id in sorted_order:
            node = graph.nodes[node_id]
            step = node.data

            if step is None:
                continue

            # 收集依赖结果
            dep_results = {}
            if self.config.pass_dependency_results:
                for dep_id in node.dependencies:
                    if dep_id in results:
                        dep_results[dep_id] = results[dep_id]

            # 如果配置传递前序结果
            if prev_result and self.config.pass_dependency_results and not dep_results:
                dep_results["previous"] = prev_result

            # 执行步骤
            step_result = step_executor(step, dep_results)
            results[node_id] = step_result

            if step_result.success:
                completed.add(node_id)
                prev_result = step_result
            else:
                # 失败时检查是否需要重试
                if self.config.retry_on_failure and step.retry_count < step.max_retries:
                    step.retry_count += 1
                    step_result = step_executor(step, dep_results)
                    results[node_id] = step_result
                    if step_result.success:
                        completed.add(node_id)

        total = len(graph.nodes)
        completed_count = len(completed)
        failed_count = total - completed_count

        return DAGExecutionResult(
            total_steps=total,
            completed_steps=completed_count,
            failed_steps=failed_count,
            results=results,
            execution_time=time.time() - start_time,
            success=failed_count == 0,
        )

    def _execute_parallel(
        self,
        graph: DAGGraph,
        step_executor: Callable[[PlanStep, Dict[str, StepResult]], StepResult],
    ) -> DAGExecutionResult:
        """
        并行执行

        参考 Shannon 的 Parallel 模式：
        使用 ThreadPoolExecutor + 信号量控制并发
        """
        results: Dict[str, StepResult] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures: Dict[Future, str] = {}

            for node_id, node in graph.nodes.items():
                step = node.data
                if step is None:
                    continue
                future = executor.submit(step_executor, step, {})
                futures[future] = node_id

            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    step_result = future.result()
                    results[node_id] = step_result
                    if step_result.success:
                        completed.add(node_id)
                    else:
                        failed.add(node_id)
                except Exception as exc:
                    results[node_id] = StepResult(
                        step_id=node_id,
                        success=False,
                        error=str(exc),
                    )
                    failed.add(node_id)

        total = len(graph.nodes)
        return DAGExecutionResult(
            total_steps=total,
            completed_steps=len(completed),
            failed_steps=len(failed),
            results=results,
            success=len(failed) == 0,
        )

    def _execute_hybrid(
        self,
        graph: DAGGraph,
        step_executor: Callable[[PlanStep, Dict[str, StepResult]], StepResult],
    ) -> DAGExecutionResult:
        """使用增量检查而非死等
            1. 为每个任务启动一个线程
            2. 线程等待依赖完成
            3. 依赖满足后获取信号量执行
            4. 完成后释放信号量
        """
        results: Dict[str, StepResult] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()
        running: Set[str] = set()
        lock_results = {}  # 用于线程安全的结果字典

        def execute_node(node_id: str) -> None:
            """执行单个节点"""
            node = graph.nodes[node_id]
            step = node.data

            if step is None:
                return

            # 等待依赖完成
            dep_completed = self._wait_for_dependencies(
                node.dependencies, completed, failed
            )

            if not dep_completed:
                # 依赖失败，跳过此节点
                results[node_id] = StepResult(
                    step_id=node_id,
                    success=False,
                    error="Dependency failed",
                )
                failed.add(node_id)
                return

            # 收集依赖结果
            dep_results = {}
            if self.config.pass_dependency_results:
                for dep_id in node.dependencies:
                    if dep_id in results:
                        dep_results[dep_id] = results[dep_id]

            # 执行步骤
            step_result = step_executor(step, dep_results)
            results[node_id] = step_result

            if step_result.success:
                completed.add(node_id)
            else:
                # 重试逻辑
                if self.config.retry_on_failure and step.retry_count < step.max_retries:
                    step.retry_count += 1
                    retry_result = step_executor(step, dep_results)
                    results[node_id] = retry_result
                    if retry_result.success:
                        completed.add(node_id)
                    else:
                        failed.add(node_id)
                else:
                    failed.add(node_id)

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            # 提交所有任务
            futures_map: Dict[Future, str] = {}
            for node_id in graph.nodes:
                future = executor.submit(execute_node, node_id)
                futures_map[future] = node_id

            # 收集结果
            for future in as_completed(futures_map):
                try:
                    future.result()
                except Exception as exc:
                    node_id = futures_map[future]
                    results[node_id] = StepResult(
                        step_id=node_id,
                        success=False,
                        error=str(exc),
                    )
                    failed.add(node_id)

        total = len(graph.nodes)
        return DAGExecutionResult(
            total_steps=total,
            completed_steps=len(completed),
            failed_steps=len(failed),
            results=results,
            success=len(failed) == 0,
        )

    def _wait_for_dependencies(
        self,
        dependencies: Set[str],
        completed: Set[str],
        failed: Set[str],
    ) -> bool:
        """
        等待依赖完成

        参考 Shannon 的 waitForDependencies：
        增量检查而非死等，有超时机制

        Returns:
            True 如果所有依赖成功完成，False 如果有依赖失败或超时
        """
        if not dependencies:
            return True

        start_time = time.time()
        timeout = self.config.dependency_timeout
        check_interval = self.config.check_interval

        while True:
            # 检查是否所有依赖都完成
            all_done = True
            any_failed = False
            for dep_id in dependencies:
                if dep_id in completed:
                    continue
                elif dep_id in failed:
                    any_failed = True
                    break
                else:
                    all_done = False

            if any_failed:
                return False
            if all_done:
                return True

            # 检查超时
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return False

            # 等待检查间隔
            time.sleep(min(check_interval, timeout - elapsed))