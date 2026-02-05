# -*- coding: utf-8 -*-
"""批量处理器 - 批量并发处理器"""

import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

from ..types import ToolDefinition, ResponseFormat, ModelNotFoundError, ClientError
from ..clients.base_client import BaseLLMClient

if TYPE_CHECKING:
    import logging

logger = None  # 将在初始化时注入


class BatchHandler:
    """
    批量并发处理器

    使用线程池并发批量调用 LLM，保持顺序与输入一致。

    Attributes:
        clients: 模型名称到客户端实例的映射
        default_client_name: 默认客户端名称
        logger: 注入的 logger 实例

    Example:
        >>> handler = BatchHandler(clients, default_client_name="main", logger=get_logger)
        >>> messages_list = [
        ...     [{"role": "user", "content": "Hello 1"}],
        ...     [{"role": "user", "content": "Hello 2"}]
        ... ]
        >>> results = handler.handle(
        ...     messages_list=messages_list,
        ...     max_workers=4,
        ...     enable_thinking=True
        ... )
        >>> print(len(results))  # 2
    """

    def __init__(self, clients: Dict[str, BaseLLMClient], default_client_name: Optional[str] = None, logger=None):
        """
        初始化批量处理器

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
        messages_list: List[List[Dict[str, Any]]],
        model_name: Optional[str] = None,
        max_workers: int = 4,
        enable_thinking: bool = False,
        clear_thinking: bool = True,
        response_format: Optional[ResponseFormat] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        处理批量并发请求

        Args:
            messages_list: 包含多个消息列表的列表
            model_name: 模型调用名称
            max_workers: 最大并发线程数
            enable_thinking: 是否开启思考模式
            clear_thinking: 是否清空之前的思考
            response_format: 响应格式
            max_tokens: 最大 token 数

        Returns:
            包含所有 LLM 响应的列表，顺序与输入一致

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
        """
        client_name = model_name or self.default_client_name
        client = self.clients.get(client_name)

        if not client:
            raise ModelNotFoundError(f"Unknown model call name: {client_name}")

        if len(messages_list) == 0:
            return []

        self.logger.info(f"BatchHandler: processing {len(messages_list)} requests with {max_workers} workers")

        results = [None] * len(messages_list)

        def process_single(index: int, messages: List[Dict[str, Any]]) -> Tuple[int, str]:
            """处理单个请求"""
            try:
                # 批量调用通常不需要流式输出，强制 stream=False
                result = client.chat(
                    messages=messages,
                    model_name=client.get_model_name(),
                    stream=False,
                    enable_thinking=enable_thinking,
                    clear_thinking=clear_thinking,
                    tools=None,
                    response_format=response_format,
                    max_tokens=max_tokens,
                )
                return index, result
            except Exception as e:
                self.logger.error(f"Batch request {index} failed: {str(e)}", exc_info=True)
                return index, str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_single, i, messages): i
                for i, messages in enumerate(messages_list)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index, response = future.result()
                results[index] = response

        self.logger.debug(f"BatchHandler: completed {len(results)} requests")
        return results