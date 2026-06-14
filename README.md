# llm-client

模块化的 LLM 客户端库，兼容 OpenAI API 格式，针对国内 vLLM 部署优化。

## 特性

- **模块化架构** — 客户端层、处理器层、解析器层清晰分离
- **流式支持** — 完整流式响应处理，增量 JSON 参数解析
- **工具调用** — 支持流式工具调用增量解析，`complete` 消息即用格式
- **思考模式** — 支持 DeepSeek（`thinking.type`）、GLM（`enable_thinking`）、Qwen3.5 等模型的 thinking 模式
- **模型适配器** — 通过适配器模式抽象不同模型的 API 差异，目前支持 DeepSeek、GLM、Qwen3.5
- **reasoning 保留** — 工具调用轮次的 `reasoning_content` 自动保留在 `complete` 消息中，下游只需 `messages.append(complete_msg)`
- **非流式 reasoning 访问** — 通过 `client.get_last_reasoning()` 获取上次非流式调用的思考内容
- **批量并发** — 内置线程池并发处理
- **Embedding** — 支持文本和多模态（图文混合）embedding，支持 Matryoshka 变长维度
- **依赖注入** — Logger 可注入，外部完全控制日志行为

## 安装

```bash
git clone https://github.com/aabbccddwasd/llm-client.git
cd llm-client
pip install -e .
```

## 快速开始

### 基础使用

```python
from llm_client import LLMHandler

handler = LLMHandler([
    {
        "call_name": "main",
        "name": "DeepSeek-V4-Flash",
        "api_key": "your-api-key",
        "api_base": "http://localhost:8000/v1"
    }
])

messages = [{"role": "user", "content": "你好"}]
response = handler.call_llm(messages, stream=False)
print(response)  # str
```

### 流式调用

```python
for chunk in handler.call_llm(messages, stream=True):
    if "content_stream" in chunk:
        print(chunk["content_stream"]["content"], end="", flush=True)
    elif "complete" in chunk:
        print(f"\n[完成]")
```

### 思考模式

```python
# 非流式 + 思考
response = handler.call_llm(messages, stream=False, enable_thinking=True)
reasoning = handler.clients["main"].get_last_reasoning()

# 流式 + 思考
for chunk in handler.call_llm(messages, stream=True, enable_thinking=True):
    if "content_stream" in chunk:
        print(chunk["content_stream"]["content"], end="")
    elif "complete" in chunk:
        print(chunk["complete"].get("reasoning_content"))  # 思考内容
```

### 工具调用（含思考模式）

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

for chunk in handler.call_llm(messages, stream=True, tools=tools, enable_thinking=True):
    if "tool_call" in chunk:
        tc = chunk["tool_call"]
        print(tc["function"]["name"], tc["function"]["arguments"])
    elif "complete" in chunk:
        print(chunk["complete"].get("reasoning_content"))
```

### 多轮工具调用（Agentic 循环）

```python
messages = [{"role": "user", "content": "帮我查北京的天气。"}]

while True:
    complete_msg = None
    for chunk in handler.call_llm(messages, stream=True, tools=tools, enable_thinking=True):
        if "tool_call" in chunk:
            tc = chunk["tool_call"]
        elif "complete" in chunk:
            complete_msg = chunk["complete"]

    if complete_msg.get("tool_calls"):
        # complete_msg 已含 reasoning_content，直接 append 即可保留思考内容
        messages.append(complete_msg)
        for tc in complete_msg["tool_calls"]:
            result = execute_tool(tc)  # 下游自己实现
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    else:
        print(complete_msg.get("content"))
        break
```

### 非流式 JSON Schema 结构化输出

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

response = handler.call_llm(messages, json_schema=schema, stream=False)
# response is a JSON string
```

### 批量处理

```python
results = handler.batch_llm(messages_list, max_workers=4, enable_thinking=True)
```

### Embedding

```python
vec = handler.embed_text("文本", dimensions=128)
vecs = handler.batch_embed_text(["文本1", "文本2"], dimensions=128)
```

## 适配器

| 模型 | 适配器 | 思考参数 | 说明 |
|------|--------|---------|------|
| DeepSeek 系列 | **DeepSeekAdapter** | `extra_body.thinking.type` + `reasoning_effort` | 官方 DeepSeek API 格式 |
| GLM-4.7 | **GLMAdapter** | `extra_body.chat_template_kwargs` | vLLM 部署格式 |
| Qwen3.5 | **Qwen35Adapter** | `extra_body.chat_template_kwargs` | vLLM 部署格式 |
| 其他 | **BaseModelAdapter** | 不自动开启 | 通用兼容 |

## 关于思考模式与工具调用

DeepSeek 文档要求：工具调用的轮次必须完整回传 `reasoning_content`。
vLLM 的 DeepSeek V4 tokenizer 在 `render_message()` 中通过 `drop_thinking=True` 和 `_drop_thinking_messages()` 自动过滤非最新 QA 轮的 reasoning，因此客户端无需手动管理。

库只需正确传递消息，下游直接 `messages.append(complete_msg)` 即可。

## 错误处理

```python
from llm_client import ModelNotFoundError

try:
    handler.call_llm(messages, model_name="invalid")
except ModelNotFoundError as e:
    print(e)
```

## 架构

```
llm_client/
├── handler.py             # LLMHandler 统一入口
├── types.py               # TypedDict 类型定义
├── config.py              # 模型适配器（Base/DeepSeek/GLM/Qwen35）
├── clients/               # API 客户端层
│   ├── base_client.py     # 抽象基类
│   └── openai_client.py   # OpenAI SDK 实现
├── handlers/              # 业务处理器层
│   ├── chat_handler.py    # 非流式聊天
│   ├── stream_handler.py  # 流式输出
│   ├── tool_handler.py    # 工具调用
│   ├── batch_handler.py   # 批量处理
│   └── embedding_handler.py  # Embedding
└── parsers/               # 响应解析器层
    ├── stream_parser.py   # 流式响应解析（reasoning/tool_calls）
    └── json_parser.py     # 流式 JSON 增量解析
```

## 许可证

MIT
