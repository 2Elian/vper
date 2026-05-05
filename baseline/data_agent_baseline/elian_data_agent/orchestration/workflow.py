import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from elian_data_agent.agents.base import Agent
from elian_data_agent.agents.planner import PlannerAgent
from elian_data_agent.agents.executor import ExecutorAgent
from elian_data_agent.agents.replanner import ReplannerAgent
from elian_data_agent.agents.validator import ValidatorAgent
from elian_data_agent.core.types import (
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
from elian_data_agent.core.workspace import Workspace
from elian_data_agent.dag.scheduler import DAGScheduler
from elian_data_agent.dag.executor import DAGExecutionConfig


@dataclass
class WorkflowConfig:
    """工作流配置"""
    max_iterations: int = 20              # 最大迭代次数 --> 参考 Eino 默认20
    max_steps_per_step: int = 8           # 每步骤最大执行步数
    dag_max_concurrency: int = 5          # DAG最大并发
    dag_dependency_timeout: float = 600   # 依赖等待超时
    enable_dag: bool = True               # 是否启用DAG并行
    enable_replanning: bool = True        # 是否启用Replanner
    enable_validation: bool = True        # 是否启用Validator验证
    validation_threshold: float = 0.7     # Validator通过阈值
    num_validation_trajectories: int = 5  # TOT评分分支数
    max_validation_replans: int = 3       # 验证失败最大重新规划次数


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
    """
    参考 Eino 的 Plan-Execute-Replan 架构，整合 DAG 调度。

    完整流程：
        1. Phase 1: Workspace 冷启动
        2. Phase 2: Planner 生成计划 (写入 Session)
        3. Phase 3: 循环执行 (Executor ↔ Replanner)
        4. Phase 3.5: Validator 验证最终答案
           - 通过 → Phase 4
           - 失败 → Replanner 修正 → 重新执行 Phase 3 → 再次验证
        5. Phase 4: 合成最终答案
    """

    def __init__(
        self,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        replanner: ReplannerAgent,
        validator: Optional[ValidatorAgent] = None,
        config: WorkflowConfig = None,
    ):
        self.planner = planner
        self.executor = executor
        self.replanner = replanner
        self.validator = validator
        self.config = config or WorkflowConfig()
        self.dag_scheduler = DAGScheduler(
            DAGExecutionConfig(
                max_concurrency=self.config.dag_max_concurrency,
                dependency_timeout=self.config.dag_dependency_timeout,
            )
        )

    # ------------------------------------------------------------------
    # 主入口 run()
    # ------------------------------------------------------------------

    def run(
        self,
        task_id: str,
        question: str,
        context_dir: Path,
        difficulty: str = "",
    ) -> WorkflowResult:
        start_time = time.time()

        session = Session()
        history = History()

        # Phase 1: Workspace 冷启动
        workspace = Workspace(context_dir)
        workspace.cold_start()
        session.set("schema_summary", workspace.get_schema_summary())
        session.set("knowledge", workspace.knowledge or "")
        session.set("context_dir", context_dir)

        events = []

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

        # Phase 3: Executor ↔ Replanner 循环
        step_results = []
        completed_steps = self._execute_plan_loop(
            plan=plan,
            task_id=task_id,
            question=question,
            context_dir=context_dir,
            difficulty=difficulty,
            workspace=workspace,
            session=session,
            history=history,
            events=events,
            step_results=step_results,
            completed_steps=set(),
        )

        # Phase 3.5: Validator 验证（在 Phase 3 循环结束后）
        if self.config.enable_validation and self.validator and plan.has_successful_steps():
            plan = self._validate_and_fix(
                plan=plan,
                task_id=task_id,
                question=question,
                context_dir=context_dir,
                difficulty=difficulty,
                workspace=workspace,
                session=session,
                history=history,
                events=events,
                step_results=step_results,
                completed_steps=completed_steps,
            )

        # Phase 4: 合成最终答案
        final_result = self._synthesize_answer(task_id, plan, session, step_results)
        final_result.events = events
        final_result.execution_time = time.time() - start_time
        final_result.plan = plan.to_dict()

        return final_result

    # ------------------------------------------------------------------
    # DAG 入口 run_with_dag()
    # ------------------------------------------------------------------

    def run_with_dag(self, task_id: str, question: str, context_dir: Path, difficulty: str = "") -> WorkflowResult:
        start_time = time.time()

        session = Session()
        history = History()

        # Phase 1
        workspace = Workspace(context_dir)
        workspace.cold_start()
        session.set("schema_summary", workspace.get_schema_summary())
        session.set("knowledge", workspace.knowledge or "")
        session.set("context_dir", context_dir)

        events = []

        # Phase 2
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

        # Phase 3: DAG 批量执行
        def step_executor_fn(step, dep_results):
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
            return StepResult(step_id=step.step_id, success=False, error="No result from executor")

        strategy = self.dag_scheduler.select_execution_strategy(plan)
        dag_result = self.dag_scheduler.execute_plan(plan, step_executor_fn, strategy)

        # Replanner 审查
        if self.config.enable_replanning:
            replanner_input = AgentInput(task_id=task_id, question=question)
            replanner_event = self.replanner.run(replanner_input, session, history)
            events.append(replanner_event.to_dict())

        # Phase 3.5: Validator
        if self.config.enable_validation and self.validator and plan.has_successful_steps():
            candidate_result = self._extract_candidate_result(plan)
            if candidate_result:
                validator_input = AgentInput(
                    task_id=task_id,
                    question=question,
                    context={"candidate_result": candidate_result},
                )
                validator_event = self.validator.run(validator_input, session, history)
                events.append(validator_event.to_dict())
                history.add_event(validator_event)

                v_output = validator_event.output or {}
                if v_output.get("decision") == "reject" and self.config.enable_replanning:
                    session.set("validation_feedback", v_output)
                    replanner_input = AgentInput(
                        task_id=task_id,
                        question=question,
                        context={"validation_reject": True, "feedback": v_output},
                    )
                    replanner_event = self.replanner.run(replanner_input, session, history)
                    events.append(replanner_event.to_dict())

        # Phase 4
        final_result = self._synthesize_answer(
            task_id, plan, session,
            dag_result.results if hasattr(dag_result, "results") else {},
        )
        final_result.events = events
        final_result.execution_time = time.time() - start_time
        final_result.plan = plan.to_dict()

        return final_result

    # ------------------------------------------------------------------
    # Phase 3: Executor ↔ Replanner 循环
    # ------------------------------------------------------------------

    def _execute_plan_loop(
        self,
        plan,
        task_id,
        question,
        context_dir,
        difficulty,
        workspace,
        session,
        history,
        events,
        step_results,
        completed_steps,
    ):
        """Phase 3: 循环执行计划步骤直到完成或失败"""
        for iteration in range(1, self.config.max_iterations + 1):
            if plan.is_complete():
                break
            if plan.all_steps_failed():
                break

            # 3a. 获取就绪步骤
            if self.config.enable_dag:
                ready_steps = self.dag_scheduler.get_next_ready_steps(plan, completed_steps)
            else:
                ready_steps = []
                for step in plan.steps:
                    if step.status == StepStatus.PENDING:
                        ready_steps = [step]
                        break

            if not ready_steps:
                if plan.has_failures():
                    break
                continue

            # 3b. 执行就绪步骤
            for step in ready_steps:
                plan.mark_step_running(step.step_id)

                dep_results = {}
                for dep_id in step.depends_on:
                    dep_step = plan.get_step(dep_id)
                    if dep_step and dep_step.result:
                        dep_results[dep_id] = StepResult(
                            step_id=dep_id,
                            success=True,
                            data=dep_step.result,
                        )

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

            # 3c. Replanner 审查
            if self.config.enable_replanning:
                replanner_input = AgentInput(task_id=task_id, question=question)
                replanner_event = self.replanner.run(replanner_input, session, history)
                events.append(replanner_event.to_dict())
                history.add_event(replanner_event)

                if replanner_event.is_terminal():
                    break

                updated_plan = session.get("plan")
                if updated_plan:
                    plan = updated_plan

        return plan

    # ------------------------------------------------------------------
    # Phase 3.5: Validator 验证 + 失败后修正循环
    # ------------------------------------------------------------------

    def _validate_and_fix(
        self,
        plan,
        task_id,
        question,
        context_dir,
        difficulty,
        workspace,
        session,
        history,
        events,
        step_results,
        completed_steps,
    ):
        """
        Phase 3.5: 验证最终答案

        循环逻辑：
            1. 提取候选答案
            2. Validator 验证
            3. 通过 → 返回 plan
            4. 失败 → Replanner 修正 → 重新执行待定步骤 → 回到步骤 1
            5. 最多循环 max_validation_replans 次
        """
        for val_round in range(self.config.max_validation_replans):
            # 提取候选答案
            candidate_result = self._extract_candidate_result(plan)
            if candidate_result is None:
                break

            # 调用 Validator
            validator_input = AgentInput(
                task_id=task_id,
                question=question,
                context={"candidate_result": candidate_result},
            )
            validator_event = self.validator.run(validator_input, session, history)
            events.append(validator_event.to_dict())
            history.add_event(validator_event)

            v_output = validator_event.output or {}
            v_decision = v_output.get("decision", "accept")

            if v_decision == "accept":
                session.set("validation_passed", True)
                break

            # 验证失败 → 触发 Replanner
            session.set("validation_feedback", v_output)

            if not self.config.enable_replanning:
                break

            replanner_input = AgentInput(
                task_id=task_id,
                question=question,
                context={"validation_reject": True, "feedback": v_output},
            )
            replanner_event = self.replanner.run(replanner_input, session, history)
            events.append(replanner_event.to_dict())
            history.add_event(replanner_event)

            if replanner_event.is_terminal():
                break

            # Replanner 修改了计划 → 重新执行待定步骤
            updated_plan = session.get("plan")
            if updated_plan:
                plan = updated_plan

            # 重新执行 Phase 3 循环（只执行 PENDING 步骤）
            plan = self._execute_plan_loop(
                plan=plan,
                task_id=task_id,
                question=question,
                context_dir=context_dir,
                difficulty=difficulty,
                workspace=workspace,
                session=session,
                history=history,
                events=events,
                step_results=step_results,
                completed_steps=completed_steps,
            )

        return plan

    # ------------------------------------------------------------------
    # Phase 4: 合成最终答案
    # ------------------------------------------------------------------

    def _synthesize_answer(
        self,
        task_id: str,
        plan: Plan,
        session: Session,
        step_results: Any,
    ) -> WorkflowResult:
        """合成最终答案"""
        final_result = session.get("final_result")
        if final_result:
            if isinstance(final_result, dict):
                columns = final_result.get("columns", [])
                rows = final_result.get("rows", [])
                if columns and rows:
                    return WorkflowResult(
                        task_id=task_id,
                        answer={"columns": columns, "rows": rows},
                        success=True,
                    )

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
                    if "columns" in result or "rows" in result:
                        return WorkflowResult(
                            task_id=task_id,
                            answer=result,
                            success=True,
                        )

        successful = [s for s in plan.steps if s.status == StepStatus.DONE]
        if not successful:
            return WorkflowResult(
                task_id=task_id,
                success=False,
                failure_reason="All steps failed",
            )

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

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_candidate_result(plan):
        """从 Plan 中提取候选结果（最后一个 DONE 步骤）"""
        if not plan or not hasattr(plan, "steps"):
            return None

        for step in reversed(plan.steps):
            if step.status == StepStatus.DONE and step.result:
                return step.result

        return None