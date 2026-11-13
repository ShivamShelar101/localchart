from openai import OpenAI

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
client = OpenAI(base_url="http://localhost:8081/v1", api_key="local")

def chat(messages, max_tokens=600, temperature=0):
    r = client.chat.completions.create(model=MODEL, messages=messages,
                                       max_tokens=max_tokens, temperature=temperature)
    return r.choices[0].message.content, r.usage
