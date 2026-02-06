# -*- coding: utf-8 -*-
"""聊天处理器 - 非流式聊天处理器"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING

from ..types import ToolDefinition, JSONSchema, ModelNotFoundError, ClientError
from ..clients.base_client import BaseLLMClient

if TYPE_CHECKING:
    import logging

logger = None  # 将在初始化时注入


class ChatHandler:
    """
    非流式聊天处理器

    专注处理非流式 LLM 调用，返回完整文本响应。

    Attributes:
        clients: 模型名称到客户端实例的映射
        default_client_name: 默认客户端名称
        logger: 注入的 logger 实例

    Example:
        >>> handler = ChatHandler(clients, default_client_name="main", logger=get_logger)
        >>> response = handler.handle(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     enable_thinking=True,
        ...     max_tokens=1000
        ... )
        >>> print(response)
    """

    def __init__(self, clients: Dict[str, BaseLLMClient], default_client_name: Optional[str] = None, logger=None):
        """
        初始化聊天处理器

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
        json_schema: Optional[JSONSchema] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        处理非流式聊天请求

        Args:
            messages: 聊天消息列表
            model_name: 模型调用名称，如果为 None 则使用默认模型
            enable_thinking: 是否开启思考模式
            clear_thinking: 是否清空之前的思考
            json_schema: JSON Schema 定义（用于 vLLM 结构化输出）
            max_tokens: 最大 token 数

        Returns:
            LLM 返回的完整文本内容

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 LLM 调用失败时
        """
        client_name = model_name or self.default_client_name
        client = self.clients.get(client_name)

        if not client:
            raise ModelNotFoundError(f"Unknown model call name: {client_name}")

        self.logger.debug(f"ChatHandler: calling {client_name}, messages={len(messages)}")

        try:
            # 使用客户端进行非流式调用
            response = client.chat(
                messages=messages,
                model_name=client.get_model_name(),
                stream=False,
                enable_thinking=enable_thinking,
                clear_thinking=clear_thinking,
                tools=None,
                json_schema=json_schema,
                max_tokens=max_tokens,
            )

            self.logger.debug(f"ChatHandler: response length={len(response)}")
            return response

        except ClientError:
            raise
        except Exception as e:
            self.logger.error(f"ChatHandler error: {str(e)}", exc_info=True)
            raise ClientError(f"Chat handling failed: {str(e)}") from e