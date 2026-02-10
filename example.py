# -*- coding: utf-8 -*-
"""llm_client 使用示例"""

import logging
import json
import os
from dotenv import load_dotenv
from llm_client import LLMHandler, ToolDefinition

# 加载环境变量文件
load_dotenv()

# 配置日志（可选）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 从环境变量读取模型配置文件路径
models_config_path = os.getenv("MODELS_CONFIG_PATH", "models.json")
if os.path.exists(models_config_path):
    with open(models_config_path, "r", encoding="utf-8") as f:
        models_config = json.load(f)
else:
    # 默认配置（如果没有配置文件）
    models_config = [
        {
            "call_name": "main",
            "name": "GLM-4.7",
            "api_key": "your-api-key",
            "api_base": "http://192.168.3.123:8000/v1"
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
messages = [{"role": "user", "content": "'1'+'1'='11'是不是对的"}]
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


# ========== 示例 8: 文本 Embedding ==========
print("\n=== 文本 Embedding ===")

text = "这是一个句子，我们将其转换为向量表示。"
embedding = handler.embed_text(text, model_name="embedding")
print(f"文本: {text}")
print(f"Embedding 维度: {len(embedding)}")
print(f"Embedding 前 5 维: {embedding[:5]}")


# ========== 示例 9: 批量文本 Embedding ==========
print("\n=== 批量文本 Embedding ===")

texts = [
    "机器学习是人工智能的一个分支",
    "深度学习基于神经网络",
    "自然语言处理处理文本数据"
]

embeddings = handler.batch_embed_text(texts, model_name="embedding")
print(f"处理文本数: {len(texts)}")
for i, (text, emb) in enumerate(zip(texts, embeddings)):
    print(f"[{i}] {text[:20]}... 维度: {len(emb)}")


# ========== 示例 10: 图文混合 Embedding ==========
print("\n=== 图文混合 Embedding ===")

import base64

# 假设有一个本地图片，转换为 base64
# 注意: 此示例需要实际存在的图片文件
try:
    image_path = "example_image.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 构建多模态消息块
        msg_block = [
            {
                "type": "text",
                "text": "这张图片展示了什么内容？"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]

        embedding = handler.embed_multimodal(msg_block, model_name="embedding")
        print(f"图文混合 Embedding 维度: {len(embedding)}")
    else:
        print("未找到 example_image.jpg，跳过此示例")
except Exception as e:
    print(f"图文混合 Embedding 示例跳过: {e}")


# ========== 示例 11: 计算文本相似度 ==========
print("\n=== 计算文本相似度 ===")

queries = [
    "人工智能",
    "汽车维修",
    "机器学习算法"
]

# 计算 embedding
query_embeddings = [handler.embed_text(q, model_name="embedding") for q in queries]

# 计算相似度矩阵
import numpy as np
embeddings_array = np.array(query_embeddings)

# 归一化
norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
normalized_embeddings = embeddings_array / norms

# 计算余弦相似度
similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

print("相似度矩阵:")
for i, q1 in enumerate(queries):
    print(f"{q1}:")
    for j, q2 in enumerate(queries):
        if i != j:
            print(f"  与 '{q2}' 的相似度: {similarity_matrix[i][j]:.4f}")