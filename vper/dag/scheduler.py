from typing import Any, Callable, Dict, List, Optional, Set

from vper.core.types import Plan, PlanStep, StepResult, StepStatus
from vper.dag.graph import DAGGraph
from vper.dag.executor import DAGExecutor, DAGExecutionConfig, DAGExecutionResult


class DAGScheduler:
    def __init__(self, config: DAGExecutionConfig = None):
        self.executor = DAGExecutor(config)
        self.config = config or DAGExecutionConfig()

    def select_execution_strategy(self, plan: Plan) -> str:
        """
        1. 有依赖 -> Hybrid
        2. LLM 建议 Sequential -> Sequential
        3. 默认 -> Parallel (或根据 plan_type)
        Returns:
            "parallel" | "sequential" | "hybrid"
        """
        # 检查是否有显式依赖
        has_dependencies = False
        for step in plan.steps:
            if len(step.depends_on) > 0:
                has_dependencies = True
                break
        # 有依赖 -> Hybrid
        if has_dependencies:
            return "hybrid"

        # 尊重 Plan 的 plan_type
        if plan.plan_type == "sequential":
            return "sequential"
        elif plan.plan_type == "dag":
            return "hybrid"


        # 兜底策略 -> Parallel
        return "parallel"

    def build_graph_from_plan(self, plan: Plan) -> DAGGraph:
        r"""build a DAG from Plan None"""
        return DAGGraph.from_steps(plan.steps)

    def execute_plan(
        self,
        plan: Plan,
        step_executor: Callable[[PlanStep, Dict[str, StepResult]], StepResult],
        strategy: Optional[str] = None,
    ) -> DAGExecutionResult:
        if strategy is None:
            strategy = self.select_execution_strategy(plan)

        graph = self.build_graph_from_plan(plan)

        if strategy == "sequential":
            result = self.executor.execute_sequential(graph, step_executor)
        elif strategy == "hybrid":
            result = self.executor.execute(graph, step_executor)
        else:
            result = self.executor._execute_parallel(graph, step_executor)

        # 更新 Plan 状态
        self._update_plan_status(plan, result)

        return result

    def _update_plan_status(self, plan: Plan, result: DAGExecutionResult) -> None:
        """根据执行结果更新 Plan 状态"""
        for step_id, step_result in result.results.items():
            step = plan.get_step(step_id)
            if step is None:
                continue

            if step_result.success:
                step.status = StepStatus.DONE
                step.result = step_result.data
            else:
                step.status = StepStatus.FAILED
                step.result = {"error": step_result.error}

    def get_next_ready_steps(
        self,
        plan: Plan,
        completed: Set[str],
        running: Optional[Set[str]] = None,
    ) -> List[PlanStep]:
        """
        Args:
            plan: 执行计划
            completed: 已完成步骤ID集合
            running: 正在执行步骤ID集合
        Returns:
            就绪步骤列表
        """
        graph = self.build_graph_from_plan(plan)
        ready_ids = graph.get_ready_nodes(completed, running or set())

        ready_steps = []
        for step_id in ready_ids:
            step = plan.get_step(step_id)
            if step is not None:
                ready_steps.append(step)

        return ready_steps

    def get_execution_layers(self, plan: Plan) -> List[List[PlanStep]]:
        """
        获取执行层级（用于可视化）

        Args:
            plan: 执行计划

        Returns:
            分层后的步骤列表
        """
        graph = self.build_graph_from_plan(plan)
        layers_ids = graph.get_execution_layers()

        layers_steps = []
        for layer_ids in layers_ids:
            layer_steps = []
            for step_id in layer_ids:
                step = plan.get_step(step_id)
                if step is not None:
                    layer_steps.append(step)
            layers_steps.append(layer_steps)

        return layers_steps