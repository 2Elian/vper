from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from enum import Enum

class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentAction(Enum):
    """Agent输出动作"""
    CONTINUE = "continue"
    EXIT = "exit"
    TRANSFER = "transfer"
    ERROR = "error"


@dataclass
class PlanStep:
    """
    执行计划中的单个步骤
    参考 Eino 的计划分解和 DAG 的 Subtask 设计
    """
    step_id: str
    description: str
    hint: str = ""
    depends_on: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)
    suggested_tools: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    max_steps: int = 8
    max_retries: int = 2
    retry_count: int = 0
    priority: int = 0

    def is_ready(self, completed_steps: Set[str]) -> bool:
        """检查步骤是否可以执行 --> 依赖已满足"""
        return all(dep in completed_steps for dep in self.depends_on)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "hint": self.hint,
            "depends_on": self.depends_on,
            "produces": self.produces,
            "consumes": self.consumes,
            "suggested_tools": self.suggested_tools,
            "status": self.status.value,
            "result": self.result,
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "priority": self.priority,
        }


@dataclass
class Plan:
    """
    执行计划
    包含任务分解后的所有步骤，支持DAG依赖关系
    """
    task_id: str
    question: str
    difficulty: str = ""
    plan_type: str = "sequential"
    steps: List[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    context_dir: Optional[Path] = None

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_pending_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    def get_running_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.RUNNING]

    def get_ready_steps(self, completed: Set[str]) -> List[PlanStep]:
        ready = []
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.is_ready(completed):
                ready.append(step)
        ready.sort(key=lambda s: s.priority, reverse=True)
        return ready

    def get_completed_steps(self) -> Set[str]:
        return {s.step_id for s in self.steps if s.status == StepStatus.DONE}

    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.DONE, StepStatus.SKIPPED)
            for s in self.steps
        )

    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def all_steps_failed(self) -> bool:
        return all(s.status == StepStatus.FAILED for s in self.steps)

    def mark_step_running(self, step_id: str) -> None:
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.RUNNING

    def mark_step_done(self, step_id: str, result: Dict[str, Any]) -> None:
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.DONE
            step.result = result

    def mark_step_failed(self, step_id: str, error: str) -> None:
        step = self.get_step(step_id)
        if step:
            step.retry_count += 1
            if step.retry_count >= step.max_retries:
                step.status = StepStatus.FAILED
                step.result = {"error": error}
            else:
                step.status = StepStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "difficulty": self.difficulty,
            "plan_type": self.plan_type,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "context_dir": str(self.context_dir) if self.context_dir else None,
        }


@dataclass
class StepResult:
    """步骤执行结果"""
    step_id: str
    success: bool
    data: Any = None
    error: str = ""
    steps_taken: int = 0
    raw_response: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "steps_taken": self.steps_taken,
            "raw_response": self.raw_response,
            "tool_calls": self.tool_calls,
        }


@dataclass
class AgentInput:
    """Agent输入数据"""
    task_id: str
    question: str
    step: Optional[PlanStep] = None
    context: Dict[str, Any] = field(default_factory=dict)
    dependency_results: Dict[str, StepResult] = field(default_factory=dict)
    max_steps: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "step": self.step.to_dict() if self.step else None,
            "context": self.context,
            "dependency_results": {k: v.to_dict() for k, v in self.dependency_results.items()},
            "max_steps": self.max_steps,
        }


@dataclass
class AgentEvent:
    """Agent事件输出"""
    agent_name: str
    output: Optional[Dict[str, Any]] = None
    action: AgentAction = AgentAction.CONTINUE
    target_agent: Optional[str] = None
    error: Optional[str] = None
    step_result: Optional[StepResult] = None

    def is_terminal(self) -> bool:
        return self.action in (AgentAction.EXIT, AgentAction.ERROR)

    def needs_transfer(self) -> bool:
        return self.action == AgentAction.TRANSFER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "output": self.output,
            "action": self.action.value,
            "target_agent": self.target_agent,
            "error": self.error,
            "step_result": self.step_result.to_dict() if self.step_result else None,
        }


class Session:
    """会话状态管理"""
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._topics: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        if key in self._store:
            del self._store[key]

    def get_topic(self, topic: str) -> Any:
        return self._topics.get(topic)

    def set_topic(self, topic: str, value: Any) -> None:
        self._topics[topic] = value

    def get_all(self) -> Dict[str, Any]:
        return dict(self._store)

    def get_all_topics(self) -> Dict[str, Any]:
        return dict(self._topics)

    def clear(self) -> None:
        self._store.clear()
        self._topics.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {"store": self._store, "topics": self._topics}


class History:
    """历史记录管理"""

    def __init__(self):
        self._events: List[AgentEvent] = []
        self._messages: List[Dict[str, str]] = []

    def add_event(self, event: AgentEvent) -> None:
        self._events.append(event)

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_events(self) -> List[AgentEvent]:
        return list(self._events)

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def to_llm_messages(self) -> List[Dict[str, str]]:
        messages = list(self._messages)
        for event in self._events:
            if event.output:
                content = str(event.output)
                messages.append({"role": "assistant", "content": content})
            if event.error:
                messages.append({"role": "user", "content": "Error: " + event.error})
        return messages

    def get_last_event(self) -> Optional[AgentEvent]:
        return self._events[-1] if self._events else None

    def clear(self) -> None:
        self._events.clear()
        self._messages.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self._events],
            "messages": self._messages,
        }


@dataclass
class TaskContext:
    """任务上下文"""
    task_id: str
    question: str
    difficulty: str
    context_dir: Path
    files: Dict[str, Any] = field(default_factory=dict)
    schemas: Dict[str, Any] = field(default_factory=dict)
    knowledge: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "difficulty": self.difficulty,
            "context_dir": str(self.context_dir),
            "files": self.files,
            "schemas": self.schemas,
            "knowledge": self.knowledge,
        }