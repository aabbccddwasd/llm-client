# -*- coding: utf-8 -*-
"""llm_client - 模块化 LLM 客户端库"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# 主入口类
from .handler import LLMHandler

# 错误类型
from .types import (
    LLMError,
    ModelNotFoundError,
    StreamParsingError,
    ClientError,
)

# 公共 API 导出
__all__ = [
    # 主类
    "LLMHandler",
    # 错误类型
    "LLMError",
    "ModelNotFoundError",
    "StreamParsingError",
    "ClientError",
    # 版本信息
    "__version__",
]

# 内部类型（可选导出，用于类型提示和高级使用）
__all__ += [
    # TypedDict 类型
    "Message",
    "AssistantMessage",
    "ToolDefinition",
    "ResponseFormat",
    "ToolCall",
    "LLMCallOptions",
    "StreamChunk",
    "ModelConfig",
    # 解析器
    "StreamingJSONParser",
    "StreamResponseParser",
    # 客户端
    "BaseLLMClient",
    "OpenAIClient",
    # 适配器
    "ModelAdapter",
    "BaseModelAdapter",
    "GLMAdapter",
    "get_adapter_for_model",
]

from .types import (
    Message,
    AssistantMessage,
    ToolDefinition,
    ResponseFormat,
    ToolCall,
    LLMCallOptions,
    StreamChunk,
    ModelConfig,
)

from .parsers import StreamingJSONParser, StreamResponseParser
from .clients import BaseLLMClient, OpenAIClient
from .config import ModelAdapter, BaseModelAdapter, GLMAdapter, get_adapter_for_model