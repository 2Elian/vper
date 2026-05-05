"""
Lead Agent - 主控 Agent

职责：
1. 分析问题类型和难度
2. 制定执行计划（DAG 或顺序）
3. 分配子任务给 Worker
4. 验证 Worker 结果
5. 合成最终答案提交

核心流程：
Phase 1: INITIAL_PLANNING -> 分析任务、分解子任务
Phase 2: EVENT_LOOP -> 监控 Worker 状态、验证结果、重新分配
Phase 3: SHUTDOWN_SYNTHESIS -> 合成答案、提交
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_agent.core.workspace import Workspace
from data_agent.core.mailbox import Mailbox, Message
from data_agent.workers.worker import WorkerAgent, WorkerResult


@dataclass
class SubTask:
    """子任务定义"""
    subtask_id: str
    description: str
    hint: str = ""
    max_steps: int = 8
    expected_output: str = ""
    depends_on: List[str] = field(default_factory=list)  # 依赖的子任务
    assigned_to: str = ""  # Worker ID
    status: str = "pending"  # pending | running | done | failed
    result: Any = None


@dataclass
class ExecutionPlan:
    """执行计划"""
    task_id: str
    question: str
    difficulty: str
    plan_type: str  # sequential | dag
    subtasks: List[SubTask] = field(default_factory=list)
    current_step: int = 0


@dataclass
class LeadResult:
    """Lead Agent 最终结果"""
    task_id: str
    answer: Optional[Dict[str, Any]] = None
    success: bool = False
    failure_reason: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    subtask_results: List[WorkerResult] = field(default_factory=list)


class LeadAgent(object):
    """
    Lead Agent: 主控 Agent，事件驱动

    解决的核心问题：
    1. 单 Agent 无规划能力 -> Lead 制定计划、分解子任务
    2. 无结果验证 -> Lead 验证 Worker 结果质量
    3. 无错误恢复 -> Lead 可重新分配失败子任务
    """

    def __init__(self, model, workspace, tools=None):
        # type: (Any, Workspace, Any) -> None
        self.model = model
        self.workspace = workspace
        self.tools = tools
        self.mailbox = Mailbox()
        self.workers = {}  # type: Dict[str, WorkerAgent]
        self.plan = None  # type: Optional[ExecutionPlan]
        self.max_workers = 3

    def run(self, task):
        # type: (Any) -> LeadResult
        """
        运行 Lead Agent 处理任务

        task 结构 (来自 benchmark):
        - task_id
        - question
        - difficulty
        - context_dir
        """
        # Phase 1: 冷启动 Workspace
        self.workspace.cold_start()

        # Phase 2: 分析问题、制定计划
        self.plan = self._analyze_and_plan(task)

        # Phase 3: 执行计划（事件循环）
        results = self._execute_plan()

        # Phase 4: 合成答案
        final_result = self._synthesize_answer(results)

        return final_result

    def _analyze_and_plan(self, task):
        # type: (Any) -> ExecutionPlan
        """
        分析问题，制定执行计划

        根据问题类型选择策略：
        1. 简单查询 -> 单 Worker + Python/SQL
        2. 跨源查询 -> 多 Worker + 数据传递
        3. 非结构化提取 -> 专门的 Reader Worker
        4. 复杂聚合 -> 多步骤顺序执行
        """
        question = task.question
        difficulty = task.difficulty
        task_id = task.task_id

        # 获取 Schema 信息
        schema_summary = self.workspace.get_schema_summary()

        # 获取 knowledge 相关信息
        knowledge_context = ""
        relevant_use_cases = []
        if self.workspace.knowledge_guide:
            guide = self.workspace.knowledge_guide
            knowledge_context = self._extract_relevant_knowledge(question, guide)
            relevant_use_cases = self._find_relevant_use_cases(question, guide)

        # 构建 planning prompt
        planning_prompt = (
            "You are a planning agent. Analyze the question and create an execution plan.\n\n"
            "=== Available Data Sources ===\n"
            "{schema}\n\n"
            "=== Relevant Knowledge ===\n"
            "{knowledge}\n\n"
            "=== Question ===\n"
            "{question}\n\n"
            "=== Difficulty ===\n"
            "{difficulty}\n\n"
            "=== Task ===\n"
            "Break down the question into subtasks. Each subtask should:\n"
            "1. Have a clear description of what to do\n"
            "2. Have a hint about how to approach it\n"
            "3. Specify expected output type\n"
            "4. If it depends on previous subtask results, specify depends_on\n\n"
            "Output format (JSON in code block):\n"
            "```json\n"
            "{{\n"
            "  \"plan_type\": \"sequential\",  // or \"dag\"\n"
            "  \"subtasks\": [\n"
            "    {{\n"
            "      \"subtask_id\": \"sub_1\",\n"
            "      \"description\": \"...\",\n"
            "      \"hint\": \"...\",\n"
            "      \"expected_output\": \"...\",\n"
            "      \"depends_on\": [],\n"
            "      \"max_steps\": 8\n"
            "    }}\n"
            "  ]\n"
            "}}\n"
            "```\n\n"
            "Keep subtasks focused and achievable within 8 steps each.\n"
        ).format(
            schema=schema_summary,
            knowledge=knowledge_context,
            question=question,
            difficulty=difficulty,
        )

        messages = [
            {"role": "system", "content": "You are an expert planner for data analysis tasks."},
            {"role": "user", "content": planning_prompt},
        ]

        try:
            raw_response = self.model.complete(messages)
            parsed = self._parse_json_response(raw_response)

            if parsed:
                plan_type = parsed.get("plan_type", "sequential")
                subtasks_raw = parsed.get("subtasks", [])

                subtasks = []
                for i, st in enumerate(subtasks_raw):
                    subtasks.append(SubTask(
                        subtask_id=st.get("subtask_id", "sub_{}".format(i + 1)),
                        description=st.get("description", ""),
                        hint=st.get("hint", ""),
                        max_steps=int(st.get("max_steps", 8)),
                        expected_output=st.get("expected_output", ""),
                        depends_on=st.get("depends_on", []),
                        status="pending",
                    ))

                return ExecutionPlan(
                    task_id=task_id,
                    question=question,
                    difficulty=difficulty,
                    plan_type=plan_type,
                    subtasks=subtasks,
                )
        except Exception:
            pass

        # 如果 parsing 失败，使用默认计划
        return self._create_default_plan(task)

    def _create_default_plan(self, task):
        # type: (Any) -> ExecutionPlan
        """创建默认的简单计划"""
        question = task.question

        # 根据问题特征创建默认子任务
        subtasks = []

        # 默认：先读取数据，再执行分析
        subtasks.append(SubTask(
            subtask_id="sub_1",
            description="读取相关数据文件，理解数据结构",
            hint="查看 knowledge.md 和数据文件列名",
            expected_output="数据 Schema 和关键列信息",
            max_steps=4,
        ))

        subtasks.append(SubTask(
            subtask_id="sub_2",
            description="执行查询或分析，回答问题",
            hint="使用 Python/pandas 或 SQL 执行查询",
            expected_output="查询结果数据",
            depends_on=["sub_1"],
            max_steps=8,
        ))

        return ExecutionPlan(
            task_id=task.task_id,
            question=question,
            difficulty=task.difficulty,
            plan_type="sequential",
            subtasks=subtasks,
        )

    def _extract_relevant_knowledge(self, question, guide):
        # type: (str, Any) -> str
        """从 knowledge 中提取与问题相关的内容"""
        context_parts = []

        # 数据库名
        context_parts.append("Database: {}".format(guide.database_name))

        # 根据问题关键词匹配实体
        question_lower = question.lower()
        for entity_name, fields in guide.entities.items():
            for f in fields:
                if f.name.lower() in question_lower or any(
                    kw in question_lower for kw in f.description.lower().split()[:5]
                ):
                    context_parts.append("Entity {} - {}: {}".format(
                        entity_name, f.name, f.description
                    ))
                    if f.values:
                        context_parts.append("  Values: {}".format(", ".join(f.values)))

        # 添加约束
        for constraint in guide.constraints:
            context_parts.append("Constraint {}: {}".format(
                constraint.category, constraint.rules[0] if constraint.rules else ""
            ))

        return "\n".join(context_parts)

    def _find_relevant_use_cases(self, question, guide):
        # type: (str, Any) -> List[Any]
        """查找与问题相关的示例用例"""
        parser = guide.raw_text  # 简化：直接用文本匹配
        relevant = []

        for use_case in guide.use_cases:
            if any(kw in question.lower() for kw in use_case.name.lower().split()[:3]):
                relevant.append(use_case)

        return relevant

    def _execute_plan(self):
        # type: () -> List[WorkerResult]
        """执行计划，事件循环"""
        results = []  # type: List[WorkerResult]

        if self.plan is None:
            return results

        # 创建 Worker
        worker_id = "worker_1"
        self.workers[worker_id] = WorkerAgent(
            worker_id=worker_id,
            model=self.model,
            workspace=self.workspace,
        )

        # 顺序执行子任务
        for subtask in self.plan.subtasks:
            # 等待依赖完成
            if subtask.depends_on:
                dep_results = {}
                for dep_id in subtask.depends_on:
                    for r in results:
                        if r.task_id == subtask.subtask_id.replace("sub_", "dep_"):
                            dep_results[dep_id] = r.data

                # 添加依赖结果作为 hint
                if dep_results:
                    subtask.hint += "\n\nPrevious results: {}".format(
                        json.dumps(dep_results, ensure_ascii=False, indent=2)[:500]
                    )

            # 分配给 Worker 执行
            subtask.status = "running"
            subtask.assigned_to = worker_id

            worker_result = self.workers[worker_id].execute({
                "task_id": self.plan.task_id,
                "subtask_id": subtask.subtask_id,
                "description": subtask.description,
                "hint": subtask.hint,
                "max_steps": subtask.max_steps,
                "expected_output": subtask.expected_output,
            })

            # 记录结果
            results.append(worker_result)
            subtask.result = worker_result.data
            subtask.status = "done" if worker_result.success else "failed"

            # 如果失败，可能需要重新规划（简化版本直接跳过）
            if not worker_result.success:
                # 可以在这里添加错误恢复逻辑
                pass

        return results

    def _synthesize_answer(self, subtask_results):
        # type: (List[WorkerResult]) -> LeadResult
        """合成最终答案"""
        if self.plan is None:
            return LeadResult(
                task_id="",
                success=False,
                failure_reason="No plan created",
            )

        # 检查是否有成功的结果
        successful_results = [r for r in subtask_results if r.success]

        if not successful_results:
            return LeadResult(
                task_id=self.plan.task_id,
                success=False,
                failure_reason="All subtasks failed",
                subtask_results=subtask_results,
            )

        # 最后一个成功的子任务应该返回答案
        final_result = successful_results[-1]

        # 格式化答案
        answer_data = final_result.data
        if answer_data and isinstance(answer_data, dict):
            columns = answer_data.get("columns", [])
            rows = answer_data.get("rows", [])

            return LeadResult(
                task_id=self.plan.task_id,
                answer={"columns": columns, "rows": rows},
                success=True,
                subtask_results=subtask_results,
                steps=[s for r in subtask_results for s in r.steps],
            )

        # 如果数据是列表或其他格式，尝试转换
        if answer_data:
            if isinstance(answer_data, list):
                # 单列答案
                return LeadResult(
                    task_id=self.plan.task_id,
                    answer={"columns": ["value"], "rows": [[v] for v in answer_data]},
                    success=True,
                    subtask_results=subtask_results,
                )

        return LeadResult(
            task_id=self.plan.task_id,
            success=False,
            failure_reason="Could not synthesize answer from results",
            subtask_results=subtask_results,
        )

    def _parse_json_response(self, raw_response):
        # type: (str) -> Optional[Dict[str, Any]]
        """解析 JSON 响应"""
        text = raw_response.strip()

        fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            payload, end = json.JSONDecoder().raw_decode(text)
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, ValueError):
            pass

        return None