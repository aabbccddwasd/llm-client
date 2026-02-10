# -*- coding: utf-8 -*-
"""工具处理器 - 流式工具调用处理器"""

from typing import List, Dict, Any, Optional, Iterator, TYPE_CHECKING

from ..types import StreamChunk, ToolDefinition, ModelNotFoundError, ClientError
from ..clients.base_client import BaseLLMClient
from ..parsers.stream_parser import StreamResponseParser

if TYPE_CHECKING:
    import logging

logger = None  # 将在初始化时注入


class ToolHandler:
    """
    流式工具调用处理器

    专注处理支持工具调用的流式 LLM 调用，生成统一的 StreamChunk 格式。

    Attributes:
        clients: 模型名称到客户端实例的映射
        default_client_name: 默认客户端名称
        logger: 注入的 logger 实例

    Example:
        >>> handler = ToolHandler(clients, default_client_name="main", logger=get_logger)
        >>> tools = [{"type": "function", "function": {...}}]
        >>> for chunk in handler.handle(
        ...     messages=[{"role": "user", "content": "What's the weather?"}],
        ...     tools=tools,
        ...     enable_thinking=True
        ... ):
        ...     if "tool_call" in chunk:
        ...         print(chunk["tool_call"])
    """

    def __init__(self, clients: Dict[str, BaseLLMClient], default_client_name: Optional[str] = None, logger=None):
        """
        初始化工具处理器

        Args:
            clients: 模型调用名称到客户端实例的映射
            default_client_name: 默认客户端名称
            logger: 可选的 logger 实例，默认使用标准 logging
        """
        self.clients = clients
        self.default_client_name = default_client_name

        if logger is not None:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger(__name__)

    def handle(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        clear_thinking: bool = False,  # 工具调用场景通常不清除 thinking
    ) -> Iterator[StreamChunk]:
        """
        处理流式工具调用请求

        Args:
            messages: 聊天上下文
            tools: 可供模型调用的工具列表
            model_name: 模型调用名称
            max_tokens: 最大生成 token 数
            enable_thinking: 是否开启思考模式（工具调用场景建议开启）
            clear_thinking: 是否清空之前的思考

        Yields:
            StreamChunk: 统一格式的流块：
                - content_stream: 文本或 reasoning 内容
                - tool_call: 工具调用增量
                - complete: 最终完整消息

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 LLM 调用失败时
        """
        client_name = model_name or self.default_client_name
        client = self.clients.get(client_name)

        if not client:
            raise ModelNotFoundError(f"Unknown model call name: {client_name}")

        self.logger.info(f"ToolHandler: calling {client_name}, thinking={enable_thinking}, tools={len(tools)}")

        try:
            # 获取原始流（启用了工具流式调用）
            stream = client.chat(
                messages=messages,
                model_name=client.get_model_name(),
                stream=True,
                enable_thinking=enable_thinking,
                clear_thinking=clear_thinking,
                tools=tools,
                max_tokens=max_tokens,
            )

            # 使用解析器处理流（包括工具调用增量）
            parser = StreamResponseParser(logger=self.logger)
            for chunk in parser.parse(stream):
                yield chunk

            self.logger.info("ToolHandler: completed successfully")

        except ClientError:
            raise
        except Exception as e:
            self.logger.error(f"ToolHandler error: {str(e)}", exc_info=True)
            raise ClientError(f"Tool handling failed: {str(e)}") from e