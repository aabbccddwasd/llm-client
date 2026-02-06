# -*- coding: utf-8 -*-
"""llm_client 使用示例"""

import logging
import json
from llm_client import LLMHandler, ToolDefinition

# 配置日志（可选）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 配置模型
models_config = [
    {
        "call_name": "main",
        "name": "GLM-4.7",
        "api_key": "your-api-key",
        "api_base": "http://localhost:8000/v1"
    },
    {
        "call_name": "backup",
        "name": "Qwen3-VL",
        "api_key": "your-api-key",
        "api_base": "http://localhost:8001/v1"
    }
]

# 创建处理器
handler = LLMHandler(models_config)
print(f"Available models: {handler.model_names}")


# ========== 示例 1: 非流式调用 ==========
print("\n=== 非流式调用 ===")
messages = [{"role": "user", "content": "介绍一下你自己"}]
response = handler.call_llm(messages, enable_thinking=True, model_name="main")
print(f"Response: {response}")


# ========== 示例 2: 流式调用 ==========
print("\n=== 流式调用 ===")
messages = [{"role": "user", "content": "写一首短诗"}]
for chunk in handler.call_llm(messages, stream=True, enable_thinking=True):
    if "content_stream" in chunk:
        content = chunk["content_stream"]["content"]
        print(content, end="", flush=True)
    elif "complete" in chunk:
        print(f"\n[完成: {chunk['complete']}]")
print()


# ========== 示例 3: 工具调用 ==========
print("\n=== 工具调用 ===")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

messages = [{"role": "user", "content": "北京现在的天气怎么样？"}]

for chunk in handler.call_llm(messages, stream=True, tools=tools):
    if "tool_call" in chunk:
        tc = chunk["tool_call"]
        print(f"[工具调用] id={tc['id']}")
        if tc["function"]["name"]:
            print(f"  工具: {tc['function']['name']}")
        if tc["function"]["arguments"]:
            print(f"  参数: {tc['function']['arguments']}")
    elif "content_stream" in chunk:
        print(chunk["content_stream"]["content"], end="", flush=True)
    elif "complete" in chunk:
        print(f"\n[完成]")
print()


# ========== 示例 4: 批量处理 ==========
print("\n=== 批量处理 ===")
messages_list = [
    [{"role": "user", "content": f"写一个关于{topic}的句子"}]
    for topic in ["春天", "夏天", "秋天", "冬天"]
]

results = handler.batch_llm(messages_list, max_workers=4)
for i, result in enumerate(results):
    print(f"[{i}] {result[:50]}...")


# ========== 示例 5: 结构化输出 (JSON Schema) ==========
print("\n=== 结构化输出 (vLLM JSON Schema) ===")

# 定义 JSON Schema
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "人物姓名"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "occupation": {"type": "string", "description": "职业"},
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "技能列表"
        }
    },
    "required": ["name", "age", "occupation"]
}

messages = [{"role": "user", "content": "创建一个人物介绍，包含姓名、年龄、职业和技能。"}]

response = handler.call_llm(
    messages,
    json_schema=person_schema
)
print(f"JSON 响应: {response}")

# 验证可以解析为 JSON
try:
    parsed = json.loads(response)
    print(f"解析后的 JSON: {json.dumps(parsed, indent=2, ensure_ascii=False)}")
except json.JSONDecodeError:
    print("警告: 响应不是有效的 JSON")


# ========== 示例 6: 复杂结构化输出 ==========
print("\n=== 复杂结构化输出 (嵌套 Schema) ===")

complex_schema = {
    "type": "object",
    "properties": {
        "task": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "assignee": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["id", "name"]
                },
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "completed": {"type": "boolean"}
                        },
                        "required": ["id", "title", "completed"]
                    }
                }
            },
            "required": ["title", "priority"]
        }
    },
    "required": ["task"]
}

messages = [{"role": "user", "content": "创建一个任务，包含优先级high、分配给Alice(id:001)，以及2个未完成的子任务。"}]

response = handler.call_llm(
    messages,
    json_schema=complex_schema
)
print(f"复杂 JSON 响应:\n{json.dumps(json.loads(response), indent=2, ensure_ascii=False)}")


# ========== 示例 7: 错误处理 ==========
print("\n=== 错误处理 ===")
try:
    from llm_client import ModelNotFoundError
    handler.call_llm(messages, model_name="invalid_model")
except ModelNotFoundError as e:
    print(f"捕获错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")