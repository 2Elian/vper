import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from vper.agents import (
    PlannerAgent,
    ExecutorAgent,
    ReplannerAgent,
    ValidatorAgent,
)
from vper.core.types import (
    AgentInput,
    History,
    Plan,
    PlanStep,
    Session,
    StepResult,
    StepStatus,
)
from vper.core.workspace import Workspace
from vper.dag.scheduler import DAGScheduler
from vper.dag.executor import DAGExecutionConfig
from vper.utils import get_logger

logger = get_logger("vper-Workflow")


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
    - SequentialAgent: Planner -> PlanExecuteAgent -> ReportAgent
    - LoopAgent: Executor <-> Replanner 循环
    完整流程：
        1. Phase 1: Workspace 冷启动
        2. Phase 2: Planner 生成计划 (写入 Session)
        3. Phase 3: 循环执行   --> 复杂任务自动激活DAG 简单任务按照顺序执行 中等任务并行执行
           a. Executor 执行步骤
           b. Replanner 审查结果
           c. 重复直到计划完成或达到最大迭代
        4. Phase 4: 验证Agent 验证是否答案通过
           a. 通过 --> loop to Phase 5
           b. 不通过 --> loop to Replaner* times=2
        5. Phase 4: 合成最终答案
    """

    def __init__(self, planner: PlannerAgent, executor: ExecutorAgent, replanner: ReplannerAgent, validator: Optional[ValidatorAgent] = None, config: WorkflowConfig = None,):
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

    async def run(self, task_id: str, question: str, context_dir: Path, difficulty: str = "") -> WorkflowResult:
        start_time = time.time()

        # 初始化 Session 和 History
        session = Session()
        history = History()

        # Phase 1: Workspace 冷启动
        workspace = Workspace(context_dir)
        workspace.cold_start()
        logger.info("冷启动完成")
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

        planner_event = await self.planner.run(planner_input, session, history)
        events.append(planner_event.to_dict())

        plan = session.get("plan")
        if plan is None:
            logger.error(f"The Init Plan Error, Task_id: {task_id}, question: {question}")
            return WorkflowResult(
                task_id=task_id,
                success=False,
                failure_reason="Planner failed to create plan",
                events=events,
                execution_time=time.time() - start_time,
            )
        history.add_event(planner_event)
        # Phase 3: loop-agent  --> Executor(react) <-> Replanner
        step_results: List[Dict[str, Any]] = []
        completed_steps = set()
        logger.info("🎹"*150)
        logger.info("开始执行所有计划：")
        plan = await self._execute_plan_loop(
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
        logger.info("⭐" * 150)
        logger.info("所有计划执行完成，开始验证结果和trajectory：")
        # Phase 3.5: Validator 验证 --> 在 Phase 3 循环结束后
        if self.config.enable_validation and self.validator and plan.has_successful_steps():
            plan = await self._validate_and_fix(
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

    def run_with_dag(self, task_id: str, question: str, context_dir: Path, difficulty: str = "") -> WorkflowResult:
        """
        还不能用呢！
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

    async def _execute_plan_loop(self, plan, task_id, question, context_dir, difficulty, workspace, session, history, events, step_results, completed_steps,):
        for iteration in range(1, self.config.max_iterations + 1):
            if plan.is_complete():
                logger.info(f"所有计划均已顺利完成")
                break
            if plan.all_steps_failed():
                logger.warning(f"所有计划均已失败：task_id = {task_id}, question={question}")
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

                executor_event = await self.executor.run(executor_input, session, history)
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
                replanner_event = await self.replanner.run(replanner_input, session, history)
                events.append(replanner_event.to_dict())
                history.add_event(replanner_event)

                if replanner_event.is_terminal():
                    break

                updated_plan = session.get("plan")
                if updated_plan:
                    logger.info(f"the replaner update plan: \n {plan} \n")
                    plan = updated_plan

        return plan

    async def _validate_and_fix(self, plan, task_id, question, context_dir, difficulty, workspace, session, history, events, step_results, completed_steps,):
        """
        1. Extract candidate answers
        2. Validator verification
        3. Via → Back to plan
        4. Failure → Replanner correction → Re-execute pending steps → Return to phrase 1
        5. Repeat up to max_validation_replans times
        """
        for val_round in range(self.config.max_validation_replans):
            # 提取候选答案
            candidate_result = self._extract_candidate_result(plan)
            logger.info(f"当前的候选答案：{candidate_result}")
            if candidate_result is None:
                break

            # 调用 Validator
            validator_input = AgentInput(
                task_id=task_id,
                question=question,
                context={"candidate_result": candidate_result},
            )
            validator_event = await self.validator.run(validator_input, session, history)
            events.append(validator_event.to_dict())
            history.add_event(validator_event)

            v_output = validator_event.output or {}
            v_decision = v_output.get("decision", "accept")
            logger.info(f"验证器的决策：{v_decision}")
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
            logger.info(f"开始重新规划有问题的plan")
            replanner_event = await self.replanner.run(replanner_input, session, history)
            # 在这里注意一下session的变化
            events.append(replanner_event.to_dict())
            history.add_event(replanner_event)

            if replanner_event.is_terminal():
                break

            # Replanner 修改了计划 → 重新执行待定步骤
            updated_plan = session.get("plan")
            if updated_plan:
                logger.info(f"验证器和重规划器重新规划了计划状态：{updated_plan}")
                plan = updated_plan

            # 重新执行 Phase 3 循环 --> 只执行 PENDING 步骤
            plan = await self._execute_plan_loop(
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
            else:
                logger.warning(f"最终的结果格式不对，目前结果是：\n {final_result}\n 请调整格式自己")
        # 兜底1：从计划步骤结果中提取
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
                else:
                    logger.warning(f"最终的结果格式不对，目前结果是：\n {result}\n 请调整格式自己")

        # 检查是否有任何成功结果
        successful = [s for s in plan.steps if s.status == StepStatus.DONE]
        if not successful:
            return WorkflowResult(
                task_id=task_id,
                success=False,
                failure_reason="All steps failed",
            )

        # 最后兜底：使用最后一个成功步骤的结果
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

    @staticmethod
    def _extract_candidate_result(plan):
        """从Plan中提取最后一个DONE步骤作为候选答案"""
        if not plan or not hasattr(plan, "steps"):
            return None

        for step in reversed(plan.steps):
            if step.status == StepStatus.DONE and step.result:
                return step.result

        return None