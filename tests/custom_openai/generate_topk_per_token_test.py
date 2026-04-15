from data_agent.llms import OpenAIClient, Tokenizer

if __name__ == "__main__":
    model_name = "glm-4.7"
    api_key = "sk-a90eed08080a4195a6d1fa4e44a78c35"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    tokenizer = Tokenizer(model_name)
    model = OpenAIClient(
        model_name = model_name,
        api_key = api_key,
        base_url = base_url
    )
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "你好啊，请问你是什么模型"}
    ]
    tokens = model.sync_generate_topk_per_token(current_messages=messages)
    print(tokens)