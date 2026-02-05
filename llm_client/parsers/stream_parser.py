# -*- coding: utf-8 -*-
"""流响应解析器 - 解析 OpenAI 原始流并生成统一的 StreamChunk 格式"""

from typing import Iterator, Dict, Any, Optional, List, TYPE_CHECKING

from .json_parser import StreamingJSONParser
from ..types import StreamChunk, AssistantMessage

if TYPE_CHECKING:
    import logging

logger = None  # 将在初始化时注入


class StreamResponseParser:
    """
    解析 OpenAI 原始流响应

    处理：
    - 推理内容（reasoning 字段）
    - 普通文本内容
    - 工具调用增量（流式 JSON 解析）
    - 生成统一格式的 StreamChunk

    Attributes:
        logger: 注入的 logger 实例

    Example:
        >>> parser = StreamResponseParser(logger=get_logger)
        >>> for chunk in parser.parse(stream):
        ...     if "content_stream" in chunk:
        ...         print(chunk["content_stream"]["content"], end="")
        ...     elif "tool_call" in chunk:
        ...         print(chunk["tool_call"])
    """

    def __init__(self, logger=None):
        """
        初始化解析器

        Args:
            logger: 可选的 logger 实例，默认使用标准 logging
        """
        if logger is not None:
            self.logger = logger
        else:
            import logging
            self.logger = logging.getLogger(__name__)

        self.reasoning_content: str = ""
        self.content: str = ""
        self.tool_calls: Dict[int, Dict[str, Any]] = {}

    def parse(self, stream: Iterator) -> Iterator[StreamChunk]:
        """
        解析原始流并生成 StreamChunk

        Args:
            stream: OpenAI 返回的原始流迭代器

        Yields:
            StreamChunk: 统一格式的流块：
                - content_stream: {"id": ..., "content": ...} 或 {"id": ..., "content": ...} (reasoning)
                - tool_call: {"id": ..., "function": {"name": ..., "arguments": ...}}
                - complete: 最终完整消息

        Raises:
            StreamParsingError: 当流解析失败时
        """
        try:
            for chunk in stream:
                # 验证 chunk 结构
                if not chunk.choices or not chunk.choices[0]:
                    self.logger.warning("Invalid chunk: empty choices")
                    continue

                yield from self._process_chunk(chunk)

            # 发送最终完成消息
            yield {"complete": self._build_final_message()}

        except Exception as e:
            self.logger.error(f"Stream processing error: {str(e)}", exc_info=True)
            from ..types import StreamParsingError
            raise StreamParsingError(f"Stream parsing failed: {str(e)}") from e

    def _process_chunk(self, chunk: Any) -> Iterator[StreamChunk]:
        """
        处理单个 chunk

        Args:
            chunk: OpenAI 流式的单个 chunk

        Yields:
            StreamChunk: 解析后的块
        """
        delta = chunk.choices[0].delta
        chunk_id = chunk.id

        # 处理思考内容 - 新版本 GLM-4.7 使用 'reasoning' 字段
        if hasattr(delta, "reasoning") and delta.reasoning:
            self.reasoning_content += delta.reasoning
            yield {"content_stream": {"id": chunk_id, "content": delta.reasoning}}

        # 处理普通文本内容
        if delta.content:
            self.content += delta.content
            yield {"content_stream": {"id": chunk_id, "content": delta.content}}

        # 处理工具调用
        if delta.tool_calls:
            yield from self._process_tool_calls(delta.tool_calls, chunk)

    def _process_tool_calls(self, tool_calls: Any, chunk: Any) -> Iterator[StreamChunk]:
        """
        处理工具调用增量

        Args:
            tool_calls: delta 中的工具调用列表
            chunk: 原始 chunk，用于获取完成状态

        Yields:
            StreamChunk: 工具调用相关的块
        """
        # 最后一个块可能包含完整 arguments，直接丢弃（前面碎片已拼完）
        is_final_chunk = chunk.choices[0].finish_reason == 'tool_calls'
        if is_final_chunk:
            return

        for tool_call in tool_calls:
            idx = tool_call.index
            if idx not in self.tool_calls:
                self.tool_calls[idx] = {
                    "id": tool_call.id,
                    "function": {"name": "", "arguments": "", "json_parser": StreamingJSONParser()}
                }

            # 处理工具名称
            if tool_call.function and tool_call.function.name:
                self.tool_calls[idx]["function"]["name"] = tool_call.function.name
                yield {
                    "tool_call": {
                        "id": self.tool_calls[idx]["id"],
                        "function": {"name": tool_call.function.name, "arguments": ""}
                    }
                }

            # 处理工具参数
            if tool_call.function and tool_call.function.arguments:
                # 保持原始累积（用于最终消息兼容性）
                self.tool_calls[idx]["function"]["arguments"] += tool_call.function.arguments

                # 使用流式 JSON 解析器
                parser = self.tool_calls[idx]["function"]["json_parser"]
                deltas = parser.feed(tool_call.function.arguments)

                # 如果有增量 delta，输出解析后的对象
                if deltas:
                    yield {
                        "tool_call": {
                            "id": self.tool_calls[idx]["id"],
                            "function": {
                                "name": self.tool_calls[idx]["function"]["name"],
                                "arguments": deltas
                            }
                        }
                    }

    def _build_final_message(self) -> AssistantMessage:
        """
        构建最终完整消息

        Returns:
            AssistantMessage: 完整的助手消息
        """
        # 清理 tool_calls 中的 json_parser（仅用于内部状态）
        clean_tool_calls = None
        if self.tool_calls:
            clean_tool_calls = [
                {
                    "id": tc["id"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                }
                for tc in self.tool_calls.values()
            ]

        return {
            "role": "assistant",
            "content": self.content if self.content else None,
            "reasoning_content": self.reasoning_content if self.reasoning_content else None,
            "tool_calls": clean_tool_calls
        }

    def reset(self) -> None:
        """重置解析器状态，用于新的解析任务"""
        self.reasoning_content = ""
        self.content = ""
        self.tool_calls = {}