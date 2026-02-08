# -*- coding: utf-8 -*-
"""Embedding 处理器 - 统一处理文本和多模态 embedding"""

import concurrent.futures
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from ..types import OpenAIMessageBlock, ModelNotFoundError, ClientError
from ..clients.base_client import BaseLLMClient

if TYPE_CHECKING:
    import logging

logger = None  # 将在初始化时注入


class EmbeddingHandler:
    """
    Embedding 处理器

    统一处理文本 embedding 和多模态（图文）embedding。

    Attributes:
        clients: 模型名称到客户端实例的映射
        default_client_name: 默认客户端名称
        logger: 注入的 logger 实例

    Example:
        >>> handler = EmbeddingHandler(clients, default_client_name="embedding", logger=get_logger)
        >>> vec = handler.handle_text("Hello world")
        >>> vecs = handler.handle_text_batch(["text1", "text2", "text3"])
        >>> msg = {"role": "user", "content": [{"type": "text", "text": "..."}]}
        >>> vec = handler.handle_multimodal(msg)
    """

    def __init__(self, clients: Dict[str, BaseLLMClient], default_client_name: Optional[str] = None, logger=None):
        """
        初始化 Embedding 处理器

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

    def _get_client(self, model_name: Optional[str]) -> BaseLLMClient:
        """获取客户端实例"""
        client_name = model_name or self.default_client_name
        client = self.clients.get(client_name)
        if not client:
            raise ModelNotFoundError(f"Unknown model call name: {client_name}")
        return client

    def handle_text(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """
        处理单文本 embedding

        Args:
            text: 文本字符串
            model_name: 模型调用名称

        Returns:
            embedding 向量

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 API 调用失败时
        """
        client = self._get_client(model_name)
        self.logger.debug(f"EmbeddingHandler: text embedding, text_len={len(text)}")

        result = client.embed(text, client.get_model_name())
        # 确保返回单个 embedding
        return result[0] if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) else result

    def handle_text_batch(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        处理批量文本 embedding

        Args:
            texts: 文本列表
            model_name: 模型调用名称

        Returns:
            embedding 向量列表，顺序与输入一致

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 API 调用失败时
        """
        client = self._get_client(model_name)
        self.logger.debug(f"EmbeddingHandler: batch text embedding, count={len(texts)}")

        return client.embed(texts, client.get_model_name())

    def handle_multimodal(self, msg_block: OpenAIMessageBlock, model_name: Optional[str] = None) -> List[float]:
        """
        处理图文混合 embedding

        提取 OpenAI 消息块中的文本和图片，调用 embedding API。

        Args:
            msg_block: OpenAI 格式的消息块
            model_name: 模型调用名称

        Returns:
            embedding 向量

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 API 调用失败时
        """
        client = self._get_client(model_name)

        # 处理 OpenAI 消息块格式
        text_parts = []
        images = []

        for item in msg_block.get("content", []):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url:
                    images.append(url)
            elif item.get("type") == "image_pil":
                # PIL 图片会在 extra_body 中处理
                pass  # 留给 embed() 方法处理

        # 准备输入参数
        input_text = " ".join(text_parts) or ""
        extra_body = {"image": images} if images else None

        self.logger.debug(f"EmbeddingHandler: multimodal embedding, text_len={len(input_text)}, image_count={len(images)}")

        # 调用 embed
        result = client.embed(input_text, client.get_model_name(), extra_body=extra_body)
        # 确保返回单个 embedding
        return result[0] if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) else result

    def handle_multimodal_batch(
        self,
        msg_blocks: List[OpenAIMessageBlock],
        model_name: Optional[str] = None,
        max_workers: int = 4,
    ) -> List[List[float]]:
        """
        处理批量图文混合 embedding

        使用线程池并发处理多个消息块，保持结果顺序与输入一致。

        Args:
            msg_blocks: OpenAI 格式的消息块列表
            model_name: 模型调用名称
            max_workers: 最大并发线程数

        Returns:
            embedding 向量列表，顺序与输入一致

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
        """
        client = self._get_client(model_name)
        self.logger.info(f"EmbeddingHandler: batch multimodal embedding, count={len(msg_blocks)}, workers={max_workers}")

        if len(msg_blocks) == 0:
            return []

        results = [None] * len(msg_blocks)

        def process_single(index: int, msg_block: OpenAIMessageBlock):
            """处理单个消息块"""
            try:
                embedding = self.handle_multimodal(msg_block, None)
                return index, embedding
            except Exception as e:
                self.logger.error(f"Multimodal embedding {index} failed: {e}", exc_info=True)
                # 返回空向量作为错误标记
                return index, []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_single, i, msg): i
                for i, msg in enumerate(msg_blocks)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index, embedding = future.result()
                results[index] = embedding

        self.logger.debug(f"EmbeddingHandler: completed {len(msg_blocks)} batch multimodal embeddings")
        return results
