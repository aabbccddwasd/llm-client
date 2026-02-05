# llm-client

模组化的 LLM 客户端库，支持流式传输、工具调用、思考模式等功能。

## 开发者说明

这个包是我（aabbccddwasd）为了方便自己项目开发编写的，目标是将 OpenAI 包传回来的过于复杂的 `ChatCompletionChunk` 转换成一种方便开发者观察和调用的格式，同时也处理了流式工具调用的参数抽取、混合思考模式、保留思考模式等额外功能。

## 特性

- **模块化架构** - 清晰的分层设计：客户端层、处理器层、解析器层
- **流式支持** - 完整的流式响应处理，增量解析
- **工具调用** - 支持流式工具调用，增量解析 JSON 参数
- **思考模式** - 支持 GLM-4.7 等 OpenAI 兼容模型的 thinking/reasoning 模式
- **模型适配器** - 通过适配器模式抽象不同模型的 API 差异
- **批量并发** - 内置批量处理能力，使用线程池并发调用
- **依赖注入** - Logger 可注入，完全由外部控制日志行为

## 安装

```bash
pip install llm-client
```

或从源码安装：

```bash
git clone https://github.com/yourusername/llm-client.git
cd llm-client
pip install -e .
```

## 快速开始

### 基础使用

```python
from llm_client import LLMHandler

# 配置模型
models_config = [
    {
        "call_name": "main",
        "name": "GLM-4.7",
        "api_key": "your-api-key",
        "api_base": "http://localhost:8000/v1"
    }
]

# 创建处理器（可选：传入自定义 logger）
import logging
logging.basicConfig(level=logging.INFO)
handler = LLMHandler(models_config)

# 非流式调用
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = handler.call_llm(messages, enable_thinking=True)
print(response)
```

### 流式调用

```python
# 流式输出
for chunk in handler.call_llm(messages, stream=True, enable_thinking=True):
    if "content_stream" in chunk:
        content = chunk["content_stream"]["content"]
        print(content, end="", flush=True)
    elif "complete" in chunk:
        print(f"\n[Complete: {chunk['complete']}]")
```

### 工具调用

```python
from llm_client import ToolDefinition

# 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

# 流式工具调用
messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
for chunk in handler.call_llm(messages, stream=True, tools=tools):
    if "tool_call" in chunk:
        print(f"Tool call: {chunk['tool_call']}")
    elif "content_stream" in chunk:
        print(f"Content: {chunk['content_stream']['content']}", end="")
    elif "complete" in chunk:
        print(f"\n[Complete]")
```

### 批量处理

```python
# 批量并发调用
messages_list = [
    [{"role": "user", "content": f"Hello {i}"}]
    for i in range(10)
]
results = handler.batch_llm(messages_list, max_workers=4, enable_thinking=True)
for i, r in enumerate(results):
    print(f"[{i}] {r}")
```

## 配置日志

```python
import logging

# 方式 1: 使用标准 logging 配置
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler = LLMHandler(models_config)

# 方式 2: 传入自定义 logger
logger = logging.getLogger("my_llm")
logger.setLevel(logging.DEBUG)
handler = LLMHandler(models_config, logger=logger)
```

## API 文档

### LLMHandler

主入口类，统一管理所有 LLM 调用。

**方法:**

- `call_llm(messages, stream=False, tools=None, model_name=None, enable_thinking=False, clear_thinking=True, response_format=None, max_tokens=None)` - 统一调用接口
- `batch_llm(messages_list, model_name=None, max_workers=4, enable_thinking=False, ...)` - 批量调用

**属性:**

- `models` - 模型配置映射
- `model_names` - 可用模型名称列表

### 错误类型

- `LLMError` - 基础错误类
- `ModelNotFoundError` - 模型未找到
- `StreamParsingError` - 流式解析错误
- `ClientError` - 客户端调用错误

### 类型定义

- `Message` - 聊天消息
- `ToolDefinition` - 工具定义
- `ResponseFormat` - 响应格式
- `StreamChunk` - 流式块
- `ModelConfig` - 模型配置

## 架构

```
llm_client/
├── __init__.py       # 公共 API 导出
├── handler.py        # LLMHandler 统一入口
├── types.py          # 类型定义
├── config.py         # 模型适配器
├── clients/          # 客户端层
│   ├── base_client.py
│   └── openai_client.py
├── handlers/         # 处理器层
│   ├── chat_handler.py
│   ├── stream_handler.py
│   ├── tool_handler.py
│   └── batch_handler.py
└── parsers/          # 解析器层
    ├── json_parser.py
    └── stream_parser.py
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！