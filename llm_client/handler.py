# -*- coding: utf-8 -*-
"""LLM 处理器统一入口 - 重构后的 LLMHandler 类"""

from typing import List, Dict, Any, Optional, Iterator, TYPE_CHECKING, Union

from .types import (
    ToolDefinition, JSONSchema, StreamChunk,
    ModelNotFoundError, ClientError, OpenAIMessageBlock,
)
from .clients.openai_client import OpenAIClient
from .handlers.chat_handler import ChatHandler
from .handlers.stream_handler import StreamHandler
from .handlers.tool_handler import ToolHandler
from .handlers.batch_handler import BatchHandler
from .handlers.embedding_handler import EmbeddingHandler

if TYPE_CHECKING:
    import logging

logger = None  # 延迟初始化，避免循环依赖


class LLMHandler:
    """
    LLM 处理器统一入口 - 重构后的模块化设计

    组合各个专用处理器，提供统一的公共 API。

    Attributes:
        clients: 模型调用名称到客户端实例的映射
        default_client_name: 默认模型调用名称
        chat_handler: 非流式聊天处理器
        stream_handler: 流式处理器
        tool_handler: 工具调用处理器
        batch_handler: 批量处理器
        logger: 注入的 logger 实例

    Example:
        >>> from llm_client import LLMHandler
        >>> # 配置日志（可选）
        >>> import logging
        >>> logging.basicConfig(level=logging.INFO)
        >>>
        >>> models_config = [
        ...     {
        ...         "call_name": "main",
        ...         "name": "GLM-4.7",
        ...         "api_key": "xxx",
        ...         "api_base": "http://localhost:8000/v1"
        ...     }
        ... ]
        >>> handler = LLMHandler(models_config)
        >>>
        >>> # 非流式聊天
        >>> response = handler.call_llm(messages, model_name="main", enable_thinking=True)
        >>> print(response)
        >>>
        >>> # 流式聊天
        >>> for chunk in handler.call_llm(messages, stream=True, enable_thinking=True):
        ...     if "content_stream" in chunk:
        ...         print(chunk["content_stream"]["content"], end="")
        >>>
        >>> # 流式工具调用
        >>> for chunk in handler.call_llm(messages, stream=True, tools=tools):
        ...     if "tool_call" in chunk:
        ...         print(chunk["tool_call"])
        >>>
        >>> # 批量调用
        >>> results = handler.batch_llm(messages_list, max_workers=4)
    """

    def __init__(self, models_config: List[Dict[str, Any]], logger=None):
        """
        初始化 LLM 处理器

        加载配置，创建客户端实例，初始化各个专用处理器。

        Args:
            models_config: 模型配置列表，格式如：
                [
                    {
                        "call_name": "main",      # 调用名称
                        "name": "GLM-4.7",        # 模型名称
                        "api_key": "xxx",         # API 密钥
                        "api_base": "http://..."  # API 基础 URL
                    },
                    ...
                ]
            logger: 可选的 logger 实例，默认使用标准 logging
        """
        # 初始化 logger
        if logger is not None:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger(__name__)

        self.clients: Dict[str, OpenAIClient] = {}
        self.default_client_name: Optional[str] = None

        # 初始化所有模型
        for config in models_config:
            call_name = config["call_name"]
            label = config["name"]  # 用于日志标识

            self.clients[call_name] = OpenAIClient(
                api_key=config["api_key"],
                base_url=config["api_base"],
                model_name=config["name"],
                label=label,
                logger=self.logger,
            )

            # 设置第一个模型为默认模型
            if self.default_client_name is None:
                self.default_client_name = call_name

        self.logger.info(f"LLMHandler initialized with {len(self.clients)} models, default={self.default_client_name}")

        # 初始化各个专用处理器，使用同一个 logger
        self.chat_handler = ChatHandler(self.clients, self.default_client_name, logger=self.logger)
        self.stream_handler = StreamHandler(self.clients, self.default_client_name, logger=self.logger)
        self.tool_handler = ToolHandler(self.clients, self.default_client_name, logger=self.logger)
        self.batch_handler = BatchHandler(self.clients, self.default_client_name, logger=self.logger)
        self.embedding_handler = EmbeddingHandler(self.clients, self.default_client_name, logger=self.logger)

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        tools: Optional[List[ToolDefinition]] = None,
        model_name: Optional[str] = None,
        enable_thinking: bool = False,
        clear_thinking: bool = True,
        json_schema: Optional[JSONSchema] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """
        统一的 LLM 调用接口

        根据 stream 和 tools 参数路由到对应的处理器：
        - stream=True: 返回 StreamChunk 迭代器
        - stream=False: 返回字符串
        - tools 不为 None: 启用工具调用

        Args:
            messages: 聊天消息列表
            stream: 是否开启流式传输
            tools: 工具定义列表（可选）
            model_name: 模型调用名称（可选，默认使用默认模型）
            enable_thinking: 是否开启思考模式
            clear_thinking: 是否清空之前的思考
            json_schema: JSON Schema 定义（用于 vLLM 结构化输出）
            max_tokens: 最大生成 token 数

        Returns:
            如果 stream=True，返回 StreamChunk 迭代器
            如果 stream=False，返回字符串

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 LLM 调用失败时
        """
        # 根据参数路由到对应处理器
        if stream:
            if tools is not None:
                # 流式 + 工具调用
                self.logger.debug(f"call_llm: streaming with tools, model={model_name or self.default_client_name}")
                return self.tool_handler.handle(
                    messages=messages,
                    tools=tools,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    clear_thinking=clear_thinking,
                )
            else:
                # 流式普通聊天
                self.logger.debug(f"call_llm: streaming chat, model={model_name or self.default_client_name}")
                return self.stream_handler.handle(
                    messages=messages,
                    model_name=model_name,
                    enable_thinking=enable_thinking,
                    clear_thinking=clear_thinking,
                    max_tokens=max_tokens,
                )
        else:
            # 非流式（不支持工具调用）
            self.logger.debug(f"call_llm: non-streaming chat, model={model_name or self.default_client_name}")
            return self.chat_handler.handle(
                messages=messages,
                model_name=model_name,
                enable_thinking=enable_thinking,
                clear_thinking=clear_thinking,
                json_schema=json_schema,
                max_tokens=max_tokens,
            )

    def batch_llm(
        self,
        messages_list: List[List[Dict[str, Any]]],
        model_name: Optional[str] = None,
        max_workers: int = 4,
        enable_thinking: bool = False,
        clear_thinking: bool = True,
        json_schema: Optional[JSONSchema] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        批量并发调用 LLM

        Args:
            messages_list: 包含多个消息列表的列表
            model_name: 模型调用名称
            max_workers: 最大并发线程数
            enable_thinking: 是否开启思考模式
            clear_thinking: 是否清空之前的思考
            json_schema: JSON Schema 定义（用于 vLLM 结构化输出）
            max_tokens: 最大 token 数

        Returns:
            包含所有 LLM 响应的列表，顺序与输入一致

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
        """
        self.logger.debug(f"batch_llm: processing {len(messages_list)} requests")

        return self.batch_handler.handle(
            messages_list=messages_list,
            model_name=model_name,
            max_workers=max_workers,
            enable_thinking=enable_thinking,
            clear_thinking=clear_thinking,
            json_schema=json_schema,
            max_tokens=max_tokens,
        )

    @property
    def models(self) -> Dict[str, Dict[str, Any]]:
        """
        获取模型配置（向后兼容属性）

        Returns:
            模型名称到配置的映射
        """
        return {name: {"client": client, "model_name": client.get_model_name()}
                for name, client in self.clients.items()}

    @property
    def model_names(self) -> List[str]:
        """
        获取所有可用的模型调用名称

        Returns:
            模型调用名称列表
        """
        return list(self.clients.keys())

    def embed_text(
        self,
        text: str,
        model_name: Optional[str] = "embedding",
    ) -> List[float]:
        """
        单文本 embedding

        Args:
            text: 文本字符串
            model_name: 模型调用名称（可选，默认使用 "embedding"）

        Returns:
            embedding 向量

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 API 调用失败时

        Example:
            >>> vec = handler.embed_text("Hello world")
            >>> print(len(vec))  # 4096
        """
        return self.embedding_handler.handle_text(text, model_name=model_name)

    def batch_embed_text(
        self,
        texts: List[str],
        model_name: Optional[str] = "embedding",
    ) -> List[List[float]]:
        """
        批量文本 embedding

        Args:
            texts: 文本列表
            model_name: 模型调用名称（可选，默认使用 "embedding"）

        Returns:
            embedding 向量列表，顺序与输入一致

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 API 调用失败时

        Example:
            >>> vecs = handler.batch_embed_text(["text1", "text2", "text3"])
            >>> print(len(vecs))  # 3
        """
        return self.embedding_handler.handle_text_batch(texts, model_name=model_name)

    def embed_multimodal(
        self,
        msg_block: OpenAIMessageBlock,
        model_name: Optional[str] = "embedding",
    ) -> List[float]:
        """
        图文混合 embedding

        支持 text、image_url、image_pil 类型。

        Args:
            msg_block: OpenAI 格式的消息块
            model_name: 模型调用名称（可选，默认使用 "embedding"）

        Returns:
            embedding 向量

        Raises:
            ModelNotFoundError: 当模型调用名称无效时
            ClientError: 当 API 调用失败时

        Example:
            >>> msg = {
            ...     "role": "user",
            ...     "content": [
            ...         {"type": "text", "text": "What's in this image?"},
            ...         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ...     ]
            ... }
            >>> vec = handler.embed_multimodal(msg)
        """
        return self.embedding_handler.handle_multimodal(msg_block, model_name=model_name)

    def batch_embed_multimodal(
        self,
        msg_blocks: List[OpenAIMessageBlock],
        model_name: Optional[str] = "embedding",
        max_workers: int = 4,
    ) -> List[List[float]]:
        """
        批量图文混合 embedding

        使用线程池并发处理多个消息块。

        Args:
            msg_blocks: OpenAI 格式的消息块列表
            model_name: 模型调用名称（可选，默认使用 "embedding"）
            max_workers: 最大并发线程数

        Returns:
            embedding 向量列表，顺序与输入一致

        Raises:
            ModelNotFoundError: 当模型调用名称无效时

        Example:
            >>> vecs = handler.batch_embed_multimodal([msg1, msg2, msg3], max_workers=4)
            >>> print(len(vecs))  # 3
        """
        return self.embedding_handler.handle_multimodal_batch(msg_blocks, model_name=model_name, max_workers=max_workers)