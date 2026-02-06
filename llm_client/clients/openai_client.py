# -*- coding: utf-8 -*-
"""OpenAI 客户端实现 - 基于 OpenAI SDK 和 ModelAdapter"""

from typing import List, Dict, Any, Optional, Union, Iterator, TYPE_CHECKING

from openai import OpenAI

from .base_client import BaseLLMClient
from ..types import ToolDefinition, JSONSchema, ClientError
from ..config import get_adapter_for_model

if TYPE_CHECKING:
    import logging

logger = None  # 将在初始化时注入


class OpenAIClient(BaseLLMClient):
    """
    OpenAI 兼容客户端实现

    使用 OpenAI SDK 进行 API 调用，通过 ModelAdapter 获取模型特定的参数。

    Attributes:
        client: OpenAI 客户端实例
        model_name: 模型名称
        label: 日志标识符
        logger: 注入的 logger 实例

    Example:
        >>> client = OpenAIClient(api_key="xxx", base_url="http://...", model_name="GLM-4.7", logger=get_logger)
        >>> response = client.chat(messages, model_name="GLM-4.7", stream=False)
        >>> for chunk in client.chat(messages, model_name="GLM-4.7", stream=True):
        ...     print(chunk.choices[0].delta.content, end="")
    """

    def __init__(self, api_key: str, base_url: str, model_name: str, label: str, logger=None):
        """
        初始化 OpenAI 客户端

        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model_name: 模型名称
            label: 日志标识符
            logger: 可选的 logger 实例，默认使用标准 logging
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.label = label
        self.adapter = get_adapter_for_model(model_name)

        # 注入或创建 logger
        if logger is not None:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger(__name__)

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
        try:
            # 获取基础参数
            base_params = self.adapter.get_base_params(messages, model_name, stream)

            # 获取模型特定参数
            model_params = self.adapter.get_model_specific_params(
                enable_thinking=enable_thinking,
                clear_thinking=clear_thinking,
                stream=stream,
                tools=tools,
                json_schema=json_schema,
                max_tokens=max_tokens,
            )

            # 合并参数
            request_params = {**base_params, **model_params}

            self.logger.debug(f"Calling LLM [{self.label}]: model={model_name}, stream={stream}, thinking={enable_thinking}")

            # 调用 API
            response = self.client.chat.completions.create(**request_params)

            # 流式模式返回原始流，非流式返回文本
            if stream:
                return response
            else:
                content = response.choices[0].message.content or ""
                return content

        except Exception as e:
            self.logger.error(f"LLM client error [{self.label}]: {str(e)}", exc_info=True)
            raise ClientError(f"LLM API error: {str(e)}") from e

    def get_model_name(self) -> str:
        """获取客户端使用的模型名称"""
        return self.model_name