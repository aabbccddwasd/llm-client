# -*- coding: utf-8 -*-
"""流处理器 - 流式聊天处理器"""

from typing import List, Dict, Any, Optional, Iterator, TYPE_CHECKING

from ..types import StreamChunk, ModelNotFoundError, ClientError
from ..clients.base_client import BaseLLMClient
from ..parsers.stream_parser import StreamResponseParser

if TYPE_CHECKING:
    import logging

logger = None  # 将在初始化时注入


class StreamHandler:
    """
    流式聊天处理器

    专注处理流式 LLM 调用，生成统一的 StreamChunk 格式。

    Attributes:
        clients: 模型名称到客户端实例的映射
        default_client_name: 默认客户端名称
        logger: 注入的 logger 实例

    Example:
        >>> handler = StreamHandler(clients, default_client_name="main", logger=get_logger)
        >>> for chunk in handler.handle(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     enable_thinking=True
        ... ):
        ...     if "content_stream" in chunk:
        ...         print(chunk["content_stream"]["content"], end="")
    """

    def __init__(self, clients: Dict[str, BaseLLMClient], default_client_name: Optional[str] = None, logger=None):
        """
        初始化流处理器

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
        model_name: Optional[str] = None,
        enable_thinking: bool = False,
        clear_thinking: bool = True,
        max_tokens: Optional[int] = None,
    ) -> Iterator[StreamChunk]:
        """
        处理流式聊天请求

        Args:
            messages: 聊天消息列表
            model_name: 模型调用名称，如果为 None 则使用默认模型
            enable_thinking: 是否开启思考模式
            clear_thinking: 是否清空之前的思考
            max_tokens: 最大 token 数

        Yields:
            StreamChunk: 统一格式的流块：
                - content_stream: {"id": ..., "content": ...} 或 reasoning 内容
                - complete: 最终完整消息

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 LLM 调用失败时
        """
        client_name = model_name or self.default_client_name
        client = self.clients.get(client_name)

        if not client:
            raise ModelNotFoundError(f"Unknown model call name: {client_name}")

        self.logger.debug(f"StreamHandler: calling {client_name}, messages={len(messages)}")

        try:
            # 获取原始流
            stream = client.chat(
                messages=messages,
                model_name=client.get_model_name(),
                stream=True,
                enable_thinking=enable_thinking,
                clear_thinking=clear_thinking,
                tools=None,
                max_tokens=max_tokens,
            )

            # 使用解析器处理流
            parser = StreamResponseParser(logger=self.logger)
            for chunk in parser.parse(stream):
                yield chunk

            self.logger.debug(f"StreamHandler: completed successfully")

        except ClientError:
            raise
        except Exception as e:
            self.logger.error(f"StreamHandler error: {str(e)}", exc_info=True)
            raise ClientError(f"Stream handling failed: {str(e)}") from e