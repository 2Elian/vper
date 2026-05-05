"""
Validation 模块 - 基于 OpenJudge 的答案验证组件

核心组件：
- DataAnalysisGrader: 自定义评估器（继承 LLMGrader），评估数据分析答案
- TrajectoryComprehensiveGrader: OpenJudge 内置评估器，评估执行过程质量
- AverageEvaluationStrategy: OpenJudge 内置策略，TOT 多轨迹 + 平均
- OJModel 适配器: 将项目 LLM 配置转为 OpenJudge 的 OpenAIChatModel
"""

from .oj_adapter import create_oj_model, run_async
from .data_analysis_grader import DataAnalysisGrader
from .types import ValidationScore, ValidationDecision

__all__ = [
    "create_oj_model",
    "run_async",
    "DataAnalysisGrader",
    "ValidationScore",
    "ValidationDecision",
]