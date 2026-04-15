from data_agent.agents.base import ChatModelAgent
from data_agent.core.types import (
    AgentAction,
    AgentEvent,
    AgentInput,
    History,
    Plan,
    PlanStep,
    Session,
    StepStatus,
)
from data_agent.llms import OpenAIClient, Tokenizer

class Tester(ChatModelAgent):
    def __init__(self, model: OpenAIClient):
        super().__init__(model=model)

    @property
    def name(self) -> str:
        return "Tester"

    @property
    def description(self) -> str:
        return "Tester agent Class."

    def run(self, agent_input: AgentInput, session: Session, history: History) -> AgentEvent:
        pass

    def predict(self, text: str):
        message = [
            {
                "role": "system", "content": "you are a helpful assistant"
            },
            {
                "role": "user", "content": text
            }
        ]
        return self._call_model(messages=message)

if __name__ == '__main__':
    model_name = "glm-4.7"
    api_key = "sk-a90eed08080a4195a6d1fa4e44a78c35"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    tokenizer = Tokenizer(model_name)
    model = OpenAIClient(
        model_name = model_name,
        api_key = api_key,
        base_url = base_url,
        tokenizer=tokenizer
    )
    tester = Tester(
        model
    )
    print(tester.predict(text="你好"))