"""
DataAnalysisGrader - 数据分析答案评估器（自定义 OpenJudge 评估器）

继承 OpenJudge 的 LLMGrader，专门评估数据分析场景下：
1. 答案是否正确回答了原始问题
2. 答案是否与数据源一致（无幻觉）
3. 推理过程是否合理

使用方式与 OpenJudge 内置评估器完全一致：
    grader = DataAnalysisGrader(model=oj_model)
    result = await grader.aevaluate(
        query="哪个城市销售额最高？",
        response='{"columns": ["city"], "rows": [["北京"]]}',
        context="领域知识...",
    )
"""

import textwrap
from typing import Optional

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate


DATA_ANALYSIS_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的数据分析评估专家。你的任务是评估一个数据分析结果是否正确、完整地回答了用户的原始问题。

<评分标准>
一个高质量的数据分析回答应该：
- **事实正确**：回答中的数据必须与数据源完全一致，不得捏造或歪曲数据
- **完整覆盖**：必须回答用户问题的所有方面，不得遗漏关键信息
- **推理合理**：分析过程逻辑严密，计算正确，推理链完整
- **格式规范**：结果以结构化格式呈现，字段含义清晰

以下情况必须严格扣分：
- 回答中的数字与实际数据不符（事实错误）
- 遗漏了问题要求的关键信息（不完整）
- 使用了错误的数据源或字段（推理错误）
- 捏造了数据源中不存在的数据（幻觉）
- 计算过程有明显错误（逻辑错误）
</评分标准>

<评估步骤>
1. 仔细阅读用户的原始问题，理解问题的核心需求
2. 对照领域知识（knowledge.md），理解数据的含义和约束
3. 检查回答中的数据是否与执行历史中的工具输出一致
4. 验证计算过程和推理逻辑是否正确
5. 判断回答是否完整覆盖了问题的所有方面
</评估步骤>

<评分量表>
- **5分**：完美回答。数据完全正确，完整覆盖问题所有方面，推理过程无懈可击
- **4分**：优秀回答。核心结论正确，但有微小瑕疵（如格式不够规范、非关键信息缺失）
- **3分**：一般回答。部分正确，但有关键遗漏或非核心数据存在偏差
- **2分**：较差回答。核心结论有误或关键数据缺失，推理过程有明显问题
- **1分**：完全错误。与数据源矛盾，或根本未回答用户问题
</评分量表>

<用户问题>
{query}
</用户问题>

<领域知识>
{context}
</领域知识>

<待评估回答>
{response}
</待评估回答>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<详细解释你的评分理由，指出具体的正确点和问题点>",
    "score": <1到5之间的整数>,
    "suggestion": "<如果分数低于4分，给出具体的改进建议；如果4分及以上，可填写'无'>"
}}
</输出格式>

JSON:
"""
).strip()

DATA_ANALYSIS_PROMPT_EN = textwrap.dedent(
    """
You are a professional data analysis evaluation expert. Your task is to evaluate whether a data analysis result correctly and completely answers the user's original question.

<Rubrics>
A high-quality data analysis response should:
- **Factually Correct**: Data in the response must match the data sources exactly, with no fabrication or distortion
- **Complete Coverage**: Must address all aspects of the user's question without omitting key information
- **Sound Reasoning**: Analysis process must be logically rigorous, with correct calculations and complete reasoning chains
- **Proper Format**: Results presented in structured format with clear field meanings

Strict deductions for:
- Numbers in the response don't match actual data (factual errors)
- Missing key information requested by the question (incomplete)
- Wrong data sources or fields used (reasoning errors)
- Fabricated data not present in data sources (hallucination)
- Obvious calculation errors (logic errors)
</Rubrics>

<Steps>
1. Read the user's original question carefully to understand core requirements
2. Reference domain knowledge (knowledge.md) to understand data meaning and constraints
3. Check if data in the response matches tool outputs from the execution history
4. Verify calculation process and reasoning logic are correct
5. Determine if the response fully covers all aspects of the question
</Steps>

<Scale>
- **5**: Perfect. Data completely correct, covers all aspects, reasoning is impeccable
- **4**: Excellent. Core conclusions correct with minor flaws (e.g., suboptimal format, non-critical info missing)
- **3**: Adequate. Partially correct, but has key omissions or non-core data deviations
- **2**: Poor. Core conclusions wrong or key data missing, obvious reasoning problems
- **1**: Completely wrong. Contradicts data sources, or fails to answer the question at all
</Scale>

<User Query>
{query}
</User Query>

<Domain Knowledge>
{context}
</Domain Knowledge>

<Response to Evaluate>
{response}
</Response to Evaluate>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<Detailed explanation of your score, noting specific correct points and issues>",
    "score": <integer between 1 and 5>,
    "suggestion": "<If score < 4, provide specific improvement suggestions; if >= 4, fill 'none'>"
}}
</Output Schema>

JSON:
"""
).strip()

DEFAULT_DATA_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="user", content=DATA_ANALYSIS_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=DATA_ANALYSIS_PROMPT_ZH),
        ],
    },
)


class DataAnalysisGrader(LLMGrader):
    """
    数据分析答案评估器

    继承 OpenJudge LLMGrader，专门评估数据分析场景：
    - 答案事实正确性（对照 knowledge 和 execution trace）
    - 问题覆盖完整性
    - 推理过程合理性

    评分范围: 1-5 (整数)
    可配合 AverageEvaluationStrategy 实现 TOT 多轨迹评分
    """

    DEFAULT_TEMPLATE = DEFAULT_DATA_ANALYSIS_TEMPLATE

    def __init__(
        self,
        model,
        template=None,
        language=LanguageEnum.ZH,
        strategy=None,
    ):
        super().__init__(
            name="data_analysis",
            mode=GraderMode.POINTWISE,
            description="Evaluates data analysis results for correctness, completeness and reasoning quality",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        query="",
        response="",
        context="",
        **kwargs,
    ):
        """
        评估数据分析结果

        Args:
            query: 用户原始问题
            response: 候选答案（JSON 字符串）
            context: 领域知识（knowledge.md）+ 执行历史摘要

        Returns:
            GraderScore(score=1-5, reason=..., metadata={suggestion=...})
        """
        try:
            result = await super()._aevaluate(
                query=query,
                response=response,
                context=context,
            )

            suggestion = result.metadata.get("suggestion", "") if result.metadata else ""

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata={"suggestion": suggestion},
            )

        except Exception as e:
            return GraderError(
                name=self.name,
                error="Evaluation error: {}".format(str(e)),
            )
