import json
from typing import Any, Dict, List, Optional

from openjudge.evaluation_strategy import AverageEvaluationStrategy
from openjudge.graders.agent.trajectory.trajectory_comprehensive import (
    TrajectoryComprehensiveGrader,
)
from openjudge.graders.base_grader import GraderError, GraderScore
from openjudge.models import BaseChatModel

from vper.llms import BaseLLMClient

from .base import ChatModelAgent
from vper.core.types import (
    AgentAction,
    AgentEvent,
    AgentInput,
    History,
    Session,
    StepStatus,
    ValidationScore, ValidationDecision
)
from vper.utils.async_to_sync import run_async
from vper.judge import DataAnalysisGrader


class ValidatorAgent(ChatModelAgent):
    def __init__(self, model: BaseLLMClient, slow_model: BaseChatModel, num_trajectories=3, validation_threshold=0.6):
        super().__init__(model=model)
        self.validation_threshold = validation_threshold
        self.num_trajectories = num_trajectories

        # DataAnalysisGrader + AverageEvaluationStrategy (TOT)
        self.answer_grader = DataAnalysisGrader(
            model=slow_model,
            strategy=AverageEvaluationStrategy(num_evaluations=num_trajectories), # 把同一个评估跑 N 次，取 平均分，用来减少 LLM 评分的随机性/波动。num_evaluations会在DataAnalysisGrader里面把_aevaluate函数执行num_evaluations次，然后分数取平均，最后给出原因。
            #
        )
        # TrajectoryComprehensiveGrader (单次评估，内部已含多维度)
        """
        这是一个 Agent 轨迹评估器，用来评估一个 AI Agent 完成任务的全过程——不只看最终答案对不对，还要逐步评估每一步做得好不好。
        整体架构pipeline：
            用户原始问题 + Agent 完整对话记录（messages）
                │
                ▼
            ┌─────────────────────────────────┐
            │_extract_trajectory_from_messages│  ← 剥离 system prompt，提取轨迹
            └───────────────┬─────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────┐
            │      LLM 逐步评分（1-5分）       │
            │                                 │
            │  Step 1: 贡献性/相关性/准确性/效率│
            │  Step 2: 贡献性/相关性/准确性/效率│
            │  Step 3: 贡献性/相关性/准确性/效率│
            │  ...                            │
            └───────────────┬─────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────┐
            │      _create_trajectory_callback │  ← 后处理：算平均、归一化
            │                                 │
            │  每步 4 个维度取平均 → 该步得分   │
            │  所有步骤取平均 → 总分           │
            │  1-5 归一化 → 0.0-1.0           │
            └─────────────────────────────────┘
            四个评估维度
            维度	英文	含义
            贡献性	contribution	这一步对最终解决问题有多大帮助？
            相关性	relevance	这一步和用户问题是否相关？有没有跑偏？
            准确性	accuracy	这一步的工具调用/数据引用是否正确？
            效率	efficiency	这一步是否简洁高效？有没有做无用功？
        """
        self.trajectory_grader = TrajectoryComprehensiveGrader(
            model=slow_model,
        )

    @property
    def name(self):
        return "Validator"

    @property
    def description(self):
        return "Validates answers using OpenJudge (DataAnalysis + Trajectory Comprehensive)"

    def _build_system_prompt(self):
        return "Validator Agent (OpenJudge-based)"

    async def run(self, agent_input, session, history) -> AgentEvent:
        question = agent_input.question
        knowledge = session.get("knowledge", "")
        plan = session.get("plan")
        candidate_result = agent_input.context.get("candidate_result")

        if candidate_result is None:
            candidate_result = self._extract_candidate_result(plan)

        if candidate_result is None:
            return AgentEvent(
                agent_name=self.name,
                output={"decision": "reject", "reason": "No candidate result to validate"},
                action=AgentAction.CONTINUE,
            )

        # 构建执行追踪
        execution_trace = self._build_execution_trace(history, plan)
        context_str = (("==== Knowledge documents that describe data ===\n"+knowledge) or ("==== Knowledge documents that describe data ===\n")) + "\n\n" + execution_trace

        # 构建轨迹 messages (供 TrajectoryComprehensiveGrader 使用)
        trajectory_messages = self._build_trajectory_messages(history, plan, question)

        # 1. 答案评估 (DataAnalysisGrader + AverageEvaluationStrategy TOT)
        try:
            answer_result = await self.answer_grader._aevaluate(
                    query=question,
                    response=json.dumps(candidate_result, ensure_ascii=False),
                    context=context_str,
                )

        except Exception as grader_e:
            self.logger.error(f"the DataAnalysisGrader.aevaluate method an error occurred: {grader_e}")
            raise GraderError

        # 2. 轨迹评估 (TrajectoryComprehensiveGrader)
        trajectory_result = None
        if trajectory_messages:
            try:
                trajectory_result = await self.trajectory_grader.aevaluate(
                        messages=trajectory_messages,
                        query=question,
                    )
            except Exception as trajectory_error:
                self.logger.error(f"the TrajectoryComprehensiveGrader.aevaluate method an error occured: {trajectory_error}")
                trajectory_result = None

        # 3. 聚合分数
        answer_score = self._normalize_answer_score(answer_result)
        trajectory_score = self._extract_trajectory_score(trajectory_result)

        # 最终分数 = 答案分 * 0.6 + 轨迹分 * 0.4
        if trajectory_score is not None:
            final_score = answer_score * 0.6 + trajectory_score * 0.4
        else:
            final_score = answer_score
        self.logger.info(f"the final score is {final_score}")
        # 构建结果
        answer_reason = ""
        suggestion = ""
        if isinstance(answer_result, GraderScore):
            answer_reason = answer_result.reason
            suggestion = answer_result.metadata.get("suggestion", "") if answer_result.metadata else ""

        trajectory_reason = ""
        if isinstance(trajectory_result, GraderScore):
            trajectory_reason = trajectory_result.reason

        decision = "accept" if final_score >= self.validation_threshold else "reject"

        validation_score = ValidationScore(
            score=round(final_score, 3),
            reason=answer_reason,
            metadata={
                "answer_score": round(answer_score, 3),
                "trajectory_score": round(trajectory_score, 3) if trajectory_score is not None else None,
                "answer_raw_score": answer_result.score if isinstance(answer_result, GraderScore) else None,
                "trajectory_metadata": trajectory_result.metadata if isinstance(trajectory_result, GraderScore) else None,
                "num_trajectories": self.num_trajectories,
            },
        )

        output = validation_score.to_dict()
        output["decision"] = decision
        output["suggestion"] = suggestion
        output["trajectory_reason"] = trajectory_reason
        output["threshold"] = self.validation_threshold

        if decision == "accept":
            return AgentEvent(
                agent_name=self.name,
                output=output,
                action=AgentAction.EXIT,
            )
        else:
            return AgentEvent(
                agent_name=self.name,
                output=output,
                action=AgentAction.CONTINUE,
            )

    async def validate(self, question, candidate_result, knowledge, execution_trace):
        """
        核心验证方法（可独立调用）
        """
        context_str = (knowledge or "") + "\n\n" + execution_trace
        answer_result = await self.answer_grader.aevaluate(
                query=question,
                response=json.dumps(candidate_result, ensure_ascii=False),
                context=context_str,
            )
        answer_score = self._normalize_answer_score(answer_result)

        answer_reason = ""
        suggestion = ""
        if isinstance(answer_result, GraderScore):
            answer_reason = answer_result.reason
            suggestion = answer_result.metadata.get("suggestion", "") if answer_result.metadata else ""

        decision = "accept" if answer_score >= self.validation_threshold else "reject"

        return ValidationDecision(
            decision=decision,
            score=ValidationScore(
                score=round(answer_score, 3),
                reason=answer_reason,
                metadata={"answer_raw_score": answer_result.score if isinstance(answer_result, GraderScore) else None},
            ),
            suggestion=suggestion,
        )

    @staticmethod
    def _normalize_answer_score(result):
        """将 1-5 分归一化到 0-1"""
        if isinstance(result, GraderScore):
            s = result.score
            s = max(1, min(5, float(s)))
            return (s - 1) / 4.0
        return 0.0

    @staticmethod
    def _extract_trajectory_score(result):
        """从 TrajectoryComprehensiveGrader 结果提取 0-1 分数"""
        if isinstance(result, GraderScore):
            return result.score
        return None

    def _build_execution_trace(self, history, plan):
        """构建执行追踪文本（用于 DataAnalysisGrader 的 context）"""
        lines = []

        if plan and hasattr(plan, "steps"):
            lines.append("=== Execution Results ===")
            for step in plan.steps:
                status = step.status.value if hasattr(step.status, "value") else str(step.status)
                lines.append("\n[Step: {}] Status: {}".format(step.step_id, status))
                lines.append("  Description: {}".format(step.description))

                if hasattr(step, "result") and step.result:
                    result_str = json.dumps(step.result, ensure_ascii=False, indent=2)
                    lines.append("  Result: {}".format(result_str))

        lines.append("\n=== ReAct History ===")
        messages = history.get_messages()
        # TODO 这里要想办法做压缩
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append("[{}]: {}".format(role, content))

        return "\n".join(lines)

    @staticmethod
    def _build_trajectory_messages(history, plan, question):
        """
        构建 OpenJudge TrajectoryComprehensiveGrader 需要的 messages 格式

        标准 OpenAI messages 格式:
        [{"role": "user", "content": "..."}, {"role": "assistant", "tool_calls": [...]}, ...]
        """
        messages = []

        # 初始用户问题
        messages.append({"role": "user", "content": question})

        # 从 history 构建
        for msg in history.get_messages()[-50:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                # 在 ReAct 中 user 消息是 Observation
                messages.append({"role": "tool", "content": content, "name": "tool"})
            elif role == "assistant":
                # 尝试解析为 tool_calls
                parsed = None
                try:
                    if content.startswith("{"):
                        import re
                        fence = re.search(r"```json\s*(.*?)\s*```", content, re.IGNORECASE | re.DOTALL)
                        text = fence.group(1) if fence else content
                        parsed = json.loads(text)
                except Exception:
                    pass

                if parsed and "action" in parsed and "action_input" in parsed:
                    action = parsed["action"]
                    action_input = parsed.get("action_input", {})
                    args_str = json.dumps(action_input, ensure_ascii=False)
                    messages.append({
                        "role": "assistant",
                        "content": parsed.get("thought", ""),
                        "tool_calls": [{
                            "id": str(len(messages)),
                            "function": {"name": action, "arguments": args_str},
                            "type": "function",
                        }],
                    })
                else:
                    messages.append({"role": "assistant", "content": content})

        return messages

    @staticmethod
    def _extract_candidate_result(plan):
        """从 Plan 中提取候选结果（最后一个 DONE 步骤）"""
        if not plan or not hasattr(plan, "steps"):
            return None
        for step in reversed(plan.steps):
            if step.status == StepStatus.DONE and step.result:
                return step.result
        return None