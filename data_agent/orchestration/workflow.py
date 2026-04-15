import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from data_agent.agents.base import Agent
from data_agent.agents.planner import PlannerAgent
from data_agent.agents.executor import ExecutorAgent
from data_agent.agents.replanner import ReplannerAgent
from data_agent.core.types import (
    AgentAction,
    AgentEvent,
    AgentInput,
    History,
    Plan,
    PlanStep,
    Session,
    StepResult,
    StepStatus,
)
from data_agent.core.workspace import Workspace
from data_agent.dag.scheduler import DAGScheduler
from data_agent.dag.executor import DAGExecutionConfig


@dataclass
class WorkflowConfig:
    """工作流配置"""
    max_iterations: int = 20              # 最大迭代次数 --> 参考 Eino 默认20
    max_steps_per_step: int = 8           # 每步骤最大执行步数
    dag_max_concurrency: int = 5          # DAG最大并发
    dag_dependency_timeout: float = 600   # 依赖等待超时
    enable_dag: bool = True               # 是否启用DAG并行
    enable_replanning: bool = True        # 是否启用Replanner


@dataclass
class WorkflowResult:
    """工作流执行结果"""
    task_id: str
    answer: Optional[Dict[str, Any]] = None
    success: bool = False
    failure_reason: str = ""
    plan: Optional[Dict[str, Any]] = None
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    iterations_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "plan": self.plan,
            "step_results": self.step_results,
            "events": self.events,
            "execution_time": round(self.execution_time, 3),
            "iterations_used": self.iterations_used,
        }


class PlanExecuteReplanWorkflow:
    """参考 Eino 的架构：
    SequentialAgent([
        PlanExecuteAgent(Planner, Executor, Replanner),
        ReportAgent,
    ])
        整合 DAG 调度：
            - Planner 创建带依赖的计划
            - DAG Scheduler 解析依赖，并行调度
            - Executor 执行步骤
            - Replanner 审查结果，调整计划
    """

    """
    - SequentialAgent: Planner -> PlanExecuteAgent -> ReportAgent
    - LoopAgent: Executor <-> Replanner 循环
    完整流程：
        1. Phase 1: Workspace 冷启动
        2. Phase 2: Planner 生成计划 (写入 Session)
        3. Phase 3: 循环执行
           a. DAG Scheduler 获取就绪步骤
           b. Executor 执行步骤
           c. Replanner 审查结果
           d. 重复直到计划完成或达到最大迭代
        4. Phase 4: 合成最终答案
    """

    def __init__(
        self,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        replanner: ReplannerAgent,
        config: WorkflowConfig = None,
    ):
        self.planner = planner
        self.executor = executor
        self.replanner = replanner
        self.config = config or WorkflowConfig()
        self.dag_scheduler = DAGScheduler(
            DAGExecutionConfig(
                max_concurrency=self.config.dag_max_concurrency,
                dependency_timeout=self.config.dag_dependency_timeout,
            )
        )

    def run(
        self,
        task_id: str,
        question: str,
        context_dir: Path,
        difficulty: str = "",
    ) -> WorkflowResult:
        """
        Args:
            task_id: 任务ID
            question: 用户问题
            context_dir: 上下文目录
            difficulty: 难度
        Returns:
            WorkflowResult
        """
        start_time = time.time()

        # 初始化 Session 和 History
        session = Session()
        history = History()

        # Phase 1: Workspace 冷启动
        workspace = Workspace(context_dir)
        workspace.cold_start()

        # 注入 workspace 信息到 Session
        session.set("schema_summary", workspace.get_schema_summary())
        session.set("knowledge", workspace.knowledge or "")
        session.set("context_dir", context_dir)

        # 记录事件
        events: List[Dict[str, Any]] = []

        # Phase 2: Planner 生成计划
        planner_input = AgentInput(
            task_id=task_id,
            question=question,
            context={"difficulty": difficulty, "context_dir": context_dir},
        )

        planner_event = self.planner.run(planner_input, session, history)
        events.append(planner_event.to_dict())

        plan = session.get("plan")
        if plan is None:
            return WorkflowResult(
                task_id=task_id,
                success=False,
                failure_reason="Planner failed to create plan",
                events=events,
                execution_time=time.time() - start_time,
            )
        history.add_event(planner_event)
        # Phase 3: loop-agent (Executor <-> Replanner)
        step_results: List[Dict[str, Any]] = []
        completed_steps: Set[str] = set()
        for iteration in range(1, self.config.max_iterations + 1):
            # 检查计划是否完成 --> 所有的steps 要么完成 要么跳过
            if plan.is_complete():
                break
            # 所有的步骤都失败了 直接return
            if plan.all_steps_failed():
                break

            # 3a. 获取就绪步骤
            if self.config.enable_dag:
                ready_steps = self.dag_scheduler.get_next_ready_steps(
                    plan, completed_steps
                )
            else:
                # 顺序模式：取第一个未完成的步骤
                ready_steps = []
                for step in plan.steps:
                    if step.status == StepStatus.PENDING:
                        ready_steps = [step]
                        break

            if not ready_steps:
                # 没有就绪步骤，可能是死锁
                if plan.has_failures():
                    break
                continue

            # 3b. 执行就绪步骤
            for step in ready_steps:
                plan.mark_step_running(step.step_id)

                # 收集依赖结果
                dep_results = {}
                for dep_id in step.depends_on:
                    dep_step = plan.get_step(dep_id)
                    if dep_step and dep_step.result:
                        dep_results[dep_id] = StepResult(
                            step_id=dep_id,
                            success=True,
                            data=dep_step.result,
                        )

                # 构建执行输入
                executor_input = AgentInput(
                    task_id=task_id,
                    question=question,
                    step=step,
                    context={
                        "difficulty": difficulty,
                        "context_dir": context_dir,
                        "schema_summary": workspace.get_schema_summary(),
                    },
                    dependency_results=dep_results,
                    max_steps=step.max_steps,
                )

                # 执行步骤
                executor_event = self.executor.run(executor_input, session, history)
                events.append(executor_event.to_dict())
                history.add_event(executor_event)

                if executor_event.step_result:
                    step_results.append(executor_event.step_result.to_dict())

                    if executor_event.step_result.success:
                        plan.mark_step_done(step.step_id, executor_event.step_result.data)
                        completed_steps.add(step.step_id)
                    else:
                        plan.mark_step_failed(step.step_id, executor_event.step_result.error)

            # 3c. Replanner 审查结果
            if self.config.enable_replanning:
                replanner_input = AgentInput(
                    task_id=task_id,
                    question=question,
                )

                replanner_event = self.replanner.run(replanner_input, session, history)
                events.append(replanner_event.to_dict())
                history.add_event(replanner_event)

                # 如果 Replanner 决定退出
                if replanner_event.is_terminal():
                    break

                # 更新 Plan（Replanner 可能修改了）
                updated_plan = session.get("plan")
                if updated_plan:
                    plan = updated_plan

        # Phase 4: 合成最终答案
        final_result = self._synthesize_answer(task_id, plan, session, step_results)

        final_result.events = events
        final_result.execution_time = time.time() - start_time
        final_result.plan = plan.to_dict()

        return final_result

    def run_with_dag(self, task_id: str, question: str, context_dir: Path, difficulty: str = "") -> WorkflowResult:
        """
        与 run() 的区别：
            - 使用 DAGExecutor 批量调度就绪步骤
            - 支持并行执行独立步骤
            - 依赖等待和结果传递
        """
        start_time = time.time()

        session = Session()
        history = History()

        # 冷启动
        workspace = Workspace(context_dir)
        workspace.cold_start()
        session.set("schema_summary", workspace.get_schema_summary())
        session.set("knowledge", workspace.knowledge or "")
        session.set("context_dir", context_dir)

        events: List[Dict[str, Any]] = []

        # Planner
        planner_input = AgentInput(
            task_id=task_id,
            question=question,
            context={"difficulty": difficulty, "context_dir": context_dir},
        )

        planner_event = self.planner.run(planner_input, session, history)
        events.append(planner_event.to_dict())
        history.add_event(planner_event)

        plan = session.get("plan")
        if plan is None:
            return WorkflowResult(
                task_id=task_id,
                success=False,
                failure_reason="Planner failed to create plan",
                events=events,
                execution_time=time.time() - start_time,
            )

        # DAG 批量执行
        def step_executor_fn(step: PlanStep, dep_results: Dict[str, StepResult]) -> StepResult:
            """DAG 步骤执行函数"""
            executor_input = AgentInput(
                task_id=task_id,
                question=question,
                step=step,
                context={
                    "difficulty": difficulty,
                    "context_dir": context_dir,
                    "schema_summary": workspace.get_schema_summary(),
                },
                dependency_results=dep_results,
                max_steps=step.max_steps,
            )

            event = self.executor.run(executor_input, session, history)
            events.append(event.to_dict())
            history.add_event(event)

            if event.step_result:
                return event.step_result

            return StepResult(
                step_id=step.step_id,
                success=False,
                error="No result from executor",
            )

        # 选择策略并执行
        strategy = self.dag_scheduler.select_execution_strategy(plan)
        dag_result = self.dag_scheduler.execute_plan(plan, step_executor_fn, strategy)

        # Replanner 审查
        if self.config.enable_replanning:
            replanner_input = AgentInput(task_id=task_id, question=question)
            replanner_event = self.replanner.run(replanner_input, session, history)
            events.append(replanner_event.to_dict())

        # 合成答案
        final_result = self._synthesize_answer(
            task_id, plan, session,
            dag_result.results if hasattr(dag_result, 'results') else {},
        )
        final_result.events = events
        final_result.execution_time = time.time() - start_time
        final_result.plan = plan.to_dict()

        return final_result

    def _synthesize_answer(
        self,
        task_id: str,
        plan: Plan,
        session: Session,
        step_results: Any,
    ) -> WorkflowResult:
        """合成最终答案"""
        # 检查 Session 中的最终结果
        final_result = session.get("final_result")
        if final_result:
            # 确保格式正确
            if isinstance(final_result, dict):
                columns = final_result.get("columns", [])
                rows = final_result.get("rows", [])
                if columns and rows:
                    return WorkflowResult(
                        task_id=task_id,
                        answer={"columns": columns, "rows": rows},
                        success=True,
                    )

        # 从计划步骤结果中提取
        for step in reversed(plan.steps):
            if step.status == StepStatus.DONE and step.result:
                result = step.result
                if isinstance(result, dict):
                    columns = result.get("columns", [])
                    rows = result.get("rows", [])
                    if columns and rows:
                        return WorkflowResult(
                            task_id=task_id,
                            answer={"columns": columns, "rows": rows},
                            success=True,
                        )
                    # 可能是其他格式，尝试直接作为答案
                    if "columns" in result or "rows" in result:
                        return WorkflowResult(
                            task_id=task_id,
                            answer=result,
                            success=True,
                        )

        # 检查是否有任何成功结果
        successful = [s for s in plan.steps if s.status == StepStatus.DONE]
        if not successful:
            return WorkflowResult(
                task_id=task_id,
                success=False,
                failure_reason="All steps failed",
            )

        # 使用最后一个成功步骤的结果
        last = successful[-1]
        if last.result:
            return WorkflowResult(
                task_id=task_id,
                answer=last.result,
                success=True,
            )

        return WorkflowResult(
            task_id=task_id,
            success=False,
            failure_reason="Could not synthesize answer",
        )