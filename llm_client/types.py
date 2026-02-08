# -*- coding: utf-8 -*-
"""类型定义模块 - 提供 LLM 处理器所需的所有 TypedDict 类型"""

from typing import TypedDict, Dict, Any, List, Optional, Union


# ==================== 消息类型 ====================

class Message(TypedDict, total=False):
    """聊天消息类型"""
    role: str
    content: Union[str, List[Dict[str, Any]]]


class AssistantMessage(TypedDict, total=False):
    """最终助手消息类型"""
    role: str
    content: Optional[str]
    reasoning_content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]


class ImageContent(TypedDict):
    """图像内容块"""
    type: str  # "image_url"
    image_url: Dict[str, Any]


# ==================== 工具类型 ====================

class ToolFunction(TypedDict):
    """工具函数定义"""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolDefinition(TypedDict):
    """工具定义"""
    type: str  # "function"
    function: ToolFunction


class ToolCallFunction(TypedDict, total=False):
    """工具调用函数"""
    name: str
    arguments: Union[str, Dict[str, str]]  # Dict[str, str] 用于流式增量


class ToolCall(TypedDict):
    """工具调用"""
    id: str
    function: ToolCallFunction


# ==================== JSON Schema 类型 ====================

class JSONSchema(TypedDict, total=False):
    """
    JSON Schema 类型

    用于 vLLM 结构化输出，直接传递 JSON Schema 定义。

    vLLM 格式: {"structured_outputs": {"json": <JSONSchema>}}

    Example:
        >>> schema: JSONSchema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"}
        ...     },
        ...     "required": ["name", "age"]
        ... }
    """
    type: str
    properties: Optional[Dict[str, Any]]
    required: Optional[List[str]]
    additionalProperties: Optional[bool]
    items: Optional[Any]
    enum: Optional[List[Any]]
    minimum: Optional[Union[int, float]]
    maximum: Optional[Union[int, float]]
    format: Optional[str]
    description: Optional[str]


# ==================== LLM 调用选项 ====================

class LLMCallOptions(TypedDict, total=False):
    """LLM 调用选项"""
    stream: bool
    enable_thinking: bool
    clear_thinking: bool
    json_schema: Optional[JSONSchema]
    max_tokens: Optional[int]
    tools: Optional[List[ToolDefinition]]


# ==================== 流式响应类型 ====================

class ContentStreamDelta(TypedDict):
    """内容流增量"""
    id: str
    content: str


class ReasoningStreamDelta(TypedDict):
    """推理内容流增量"""
    id: str
    content: str


class ContentStream(TypedDict):
    """内容流消息"""
    content_stream: Union[ContentStreamDelta, ReasoningStreamDelta]


class ToolCallStream(TypedDict):
    """工具调用流消息"""
    tool_call: ToolCall


class CompleteMessage(TypedDict):
    """完成消息"""
    complete: AssistantMessage


class StreamChunk(TypedDict, total=False):
    """流式块（内容/工具/完成）"""
    content_stream: Union[ContentStreamDelta, ReasoningStreamDelta]
    tool_call: ToolCall
    complete: AssistantMessage


# ==================== 模型配置类型 ====================

class ModelConfig(TypedDict):
    """模型配置"""
    name: str
    call_name: str
    api_base: str
    api_key: str
    vision: bool
    audio: bool


class MappedModelConfig(TypedDict):
    """映射后的模型配置（内部使用）"""
    client: Any  # OpenAI client instance
    model_name: str
    vision: bool
    audio: bool
    label: str  # 用于日志标识


# ==================== Embedding 类型 ====================

class OpenAIMessageBlock(TypedDict, total=False):
    """OpenAI 消息块格式（用于多模态 embedding）"""
    role: str
    content: List[Dict[str, Any]]


class EmbeddingResponse(TypedDict, total=False):
    """Embedding 响应"""
    embedding: List[float]
    tokens: int
    index: int  # 批量场景下对应的原始索引


# ==================== 错误类型 ====================

class LLMError(Exception):
    """LLM 处理器基础错误类"""
    pass


class ModelNotFoundError(LLMError):
    """模型未找到错误"""
    pass


class StreamParsingError(LLMError):
    """流式响应解析错误"""
    pass


class ClientError(LLMError):
    """LLM 客户端调用错误"""
    pass