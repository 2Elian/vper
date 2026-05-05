"""
Validator Agent - 基于 OpenJudge 的答案验证器

核心设计：
    使用 OpenJudge 的两个评估器组合验证：
    1. DataAnalysisGrader (自定义 LLMGrader): 评估最终答案的正确性、完整性
       - 输入: query + response + context(knowledge+execution_trace)
       - 输出: GraderScore(score=1-5, reason, suggestion)
    2. TrajectoryComprehensiveGrader (内置): 评估执行过程质量
       - 输入: messages (完整 ReAct 轨迹)
       - 输出: GraderScore(score=0-1 归一化, step_evaluations)
    3. AverageEvaluationStrategy (内置): TOT 多轨迹评分
       - 并行执行 N 次评估
       - 取平均分

    最终分数 = answer_score * 0.6 + trajectory_score * 0.4

异步适配：
    OpenJudge 全部是异步接口 (aevaluate)
    通过 run_async() 桥接到同步调用
"""

import json
from typing import Any, Dict, List, Optional

from openjudge.evaluation_strategy import AverageEvaluationStrategy
from openjudge.graders.agent.trajectory.trajectory_comprehensive import (
    TrajectoryComprehensiveGrader,
)
from openjudge.graders.base_grader import GraderError, GraderScore

from elian_data_agent.agents.base import ChatModelAgent
from elian_data_agent.core.types import (
    AgentAction,
    AgentEvent,
    AgentInput,
    History,
    Session,
    StepStatus,
)
from elian_data_agent.validation.data_analysis_grader import DataAnalysisGrader
from elian_data_agent.validation.oj_adapter import create_oj_model, run_async
from elian_data_agent.validation.types import ValidationScore, ValidationDecision


class ValidatorAgent(ChatModelAgent):
    """
    Validator Agent - 基于 OpenJudge 的答案验证器

    组合使用：
    - DataAnalysisGrader (自定义 LLMGrader) + AverageEvaluationStrategy = TOT 答案评估
    - TrajectoryComprehensiveGrader (内置) = 执行过程评估

    继承 ChatModelAgent 保持与项目 Agent 接口一致，
    内部通过 OJ 适配层调用 OpenJudge 异步评估器。
    """

    def __init__(
        self,
        model,
        model_config=None,
        num_trajectories=3,
        validation_threshold=0.6,
    ):
        """
        Args:
            model: BaseLLMClient (项目的 LLM 客户端，用于 ChatModelAgent 接口)
            model_config: dict (用于创建 OpenJudge 的 OpenAIChatModel)
                         需要 model, api_key, api_base
            num_trajectories: TOT 分支数 (通过 AverageEvaluationStrategy)
            validation_threshold: 通过阈值 (0-1 归一化)
        """
        super().__init__(model=model)
        self.validation_threshold = validation_threshold
        self.num_trajectories = num_trajectories

        # 创建 OpenJudge 评估器
        if model_config:
            oj_model = create_oj_model(model_config)
        else:
            raise ValueError("model_config is required for OpenJudge integration")

        # 1. DataAnalysisGrader + AverageEvaluationStrategy (TOT)
        self.answer_grader = DataAnalysisGrader(
            model=oj_model,
            strategy=AverageEvaluationStrategy(num_evaluations=num_trajectories),
        )

        # 2. TrajectoryComprehensiveGrader (单次评估，内部已含多维度)
        self.trajectory_grader = TrajectoryComprehensiveGrader(
            model=oj_model,
        )

    @property
    def name(self):
        return "Validator"

    @property
    def description(self):
        return "Validates answers using OpenJudge (DataAnalysis + Trajectory Comprehensive)"

    def _build_system_prompt(self):
        return "Validator Agent (OpenJudge-based)"

    def run(self, agent_input, session, history):
        """
        Agent 接口方法（供 Workflow 调用）
        """
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
        context_str = (knowledge or "") + "\n\n" + execution_trace

        # 构建轨迹 messages (供 TrajectoryComprehensiveGrader 使用)
        trajectory_messages = self._build_trajectory_messages(history, plan, question)

        # 1. 答案评估 (DataAnalysisGrader + AverageEvaluationStrategy TOT)
        answer_result = run_async(
            self.answer_grader.aevaluate(
                query=question,
                response=json.dumps(candidate_result, ensure_ascii=False),
                context=context_str,
            )
        )

        # 2. 轨迹评估 (TrajectoryComprehensiveGrader)
        trajectory_result = None
        if trajectory_messages:
            try:
                trajectory_result = run_async(
                    self.trajectory_grader.aevaluate(
                        messages=trajectory_messages,
                        query=question,
                    )
                )
            except Exception:
                trajectory_result = None

        # 3. 聚合分数
        answer_score = self._normalize_answer_score(answer_result)
        trajectory_score = self._extract_trajectory_score(trajectory_result)

        # 最终分数 = 答案分 * 0.6 + 轨迹分 * 0.4
        if trajectory_score is not None:
            final_score = answer_score * 0.6 + trajectory_score * 0.4
        else:
            final_score = answer_score

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

    def validate(self, question, candidate_result, knowledge, execution_trace):
        """
        核心验证方法（可独立调用）
        """
        context_str = (knowledge or "") + "\n\n" + execution_trace
        answer_result = run_async(
            self.answer_grader.aevaluate(
                query=question,
                response=json.dumps(candidate_result, ensure_ascii=False),
                context=context_str,
            )
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
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "\n  ... (truncated)"
                    lines.append("  Result: {}".format(result_str))

        lines.append("\n=== ReAct History ===")
        messages = history.get_messages()
        for msg in messages[-30:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 500:
                content = content[:500] + "\n... (truncated)"
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