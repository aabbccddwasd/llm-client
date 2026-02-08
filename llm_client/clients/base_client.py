# -*- coding: utf-8 -*-
"""LLM 客户端抽象基类 - 定义统一的客户端接口"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator

from ..types import ToolDefinition, JSONSchema, ClientError


class BaseLLMClient(ABC):
    """
    LLM 客户端抽象基类

    定义所有 LLM 客户端必须实现的接口。
    使用 ModelAdapter 获取模型特定的 API 参数。

    Example:
        >>> class MyClient(BaseLLMClient):
        ...     def chat(self, messages, model_name, stream=True, ...):
        ...         # 实现调用逻辑
        ...         return response_stream if stream else response_text
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        stream: bool = False,
        enable_thinking: bool = False,
        clear_thinking: bool = True,
        tools: Optional[List[ToolDefinition]] = None,
        json_schema: Optional[JSONSchema] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[str, Iterator]:
        """
        调用 LLM 进行对话

        Args:
            messages: 聊天消息列表
            model_name: 模型名称
            stream: 是否流式输出
            enable_thinking: 是否开启思考模式
            clear_thinking: 是否清空之前的思考
            tools: 工具定义列表
            json_schema: JSON Schema 定义（用于 vLLM 结构化输出）
            max_tokens: 最大 token 数

        Returns:
            如果 stream=True，返回原始流迭代器
            如果 stream=False，返回生成的文本字符串

        Raises:
            ClientError: 当调用失败时
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        获取客户端使用的模型名称

        Returns:
            模型名称字符串
        """
        pass

    @abstractmethod
    def embed(
        self,
        input_data: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]],
        model_name: str,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Embedding 调用

        Args:
            input_data: 输入数据（文本字符串、文本列表、或图文混合结构）
            model_name: 模型名称
            extra_body: 额外的 API 参数（用于传图片）

        Returns:
            embedding 向量或向量列表

        Raises:
            ClientError: 当调用失败时
        """
        pass