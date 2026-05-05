from openjudge.evaluation_strategy import AverageEvaluationStrategy
from openjudge.models import BaseChatModel, OpenAIChatModel

from vper.judge import DataAnalysisGrader

slow_model = OpenAIChatModel(
        model="deepseek-r1-0528",
        api_key="sk-a90eed08080a4195a6d1fa4e44a78c35",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
grader = DataAnalysisGrader(
            model=slow_model,
            strategy=AverageEvaluationStrategy(num_evaluations=3),)
import asyncio

async def main():
    res = await grader._aevaluate(query="你好", response="你好我是你哥哥", context="asdasdasdasdasdsad")
    print(res)
    return res

if __name__ == "__main__":
    asyncio.run(main())
