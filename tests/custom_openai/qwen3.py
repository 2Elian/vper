from pathlib import Path
from openjudge.models.openai_chat_model import OpenAIChatModel
from vper.llms import OpenAIClient, Tokenizer
import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
model_name = "qwen3-30b-a3b"
tokenizer_instance = Tokenizer(
    model_name=model_name
)
model_instance = OpenAIClient(
    model_name=model_name,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-a90eed08080a4195a6d1fa4e44a78c35",
    temperature=0.0,
    tokenizer=tokenizer_instance
)
messages = [
    {
        "role": "assistant",
        "content": "you are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Where is this?"
    }
]
response = model_instance.sync_generate_answer(messages)
extra_body = {
    "chat_template_kwargs": {"enable_thinking": False},
}
# slow_model = OpenAIChatModel(
#     model=model_name,
#     base_url="",
#     api_key="",
#     extra_body = {
#         "chat_template_kwargs": {"enable_thinking": True},
#     }
# )
