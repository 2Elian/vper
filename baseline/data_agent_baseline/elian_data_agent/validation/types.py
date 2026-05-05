"""
Validation 数据类型

参考 OpenJudge 的 GraderScore 设计：
- score: 数值评分 (0.0-1.0)
- reason: 评分理由（可解释性）
- metadata: 附加维度评分等信息
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationScore:
    """
    评分结果

    参考 OpenJudge GraderScore(name, score, reason, metadata)
    简化为同步版本，去除 Pydantic 依赖以兼容 Python 3.6.5
    """

    score: float = 0.0
    reason: str = ""
    dimension_scores: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # TOT 聚合信息
    all_scores: List[float] = field(default_factory=list)
    trimmed_scores: List[float] = field(default_factory=list)
    num_trajectories: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "reason": self.reason,
            "dimension_scores": self.dimension_scores,
            "metadata": self.metadata,
            "all_scores": [round(s, 3) for s in self.all_scores],
            "trimmed_scores": [round(s, 3) for s in self.trimmed_scores],
            "num_trajectories": self.num_trajectories,
        }


@dataclass
class ValidationDecision:
    """
    验证决策

    包含验证结果和修正建议（reject 时有值）
    """

    decision: str = "accept"  # "accept" 或 "reject"
    score: Optional[ValidationScore] = None
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "decision": self.decision,
            "suggestion": self.suggestion,
        }
        if self.score:
            result["score"] = self.score.to_dict()
        return result