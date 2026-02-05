# -*- coding: utf-8 -*-
"""流式 JSON 解析器 - 增量解析简单 {"key": "value"} 格式的 JSON"""

from typing import Optional, Dict, List


class StreamingJSONParser:
    """
    增量 JSON 解析器，只处理简单的 {"key": "value"} 格式

    用于流式工具调用场景，逐步解析工具参数 JSON。
    支持转义字符和 Unicode 转义序列处理。

    Attributes:
        buffer: 完整累积的 JSON 字符串
        prev_buffer_len: 上一次 feed 时的 buffer 长度
        current_key: 当前正在解析的 key
        in_key: 是否在解析 key
        in_value: 是否在解析 value
        escape_next: 是否处理转义字符
        unicode_chars: Unicode 转义序列缓存（用于 \\uXXXX）
        current_delta: 当前 key 的增量缓存

    Example:
        >>> parser = StreamingJSONParser()
        >>> delta = parser.feed('{"na')
        >>> print(delta)
        None
        >>> delta = parser.feed('me": "John')
        >>> print(delta)
        {'name': 'John'}
    """

    # JSON 转义序列映射
    ESCAPE_MAP = {
        '"': '"',
        '\\': '\\',
        'n': '\n',
        't': '\t',
        'r': '\r',
        'b': '\b',
        'f': '\f',
    }

    def __init__(self):
        """初始化解析器状态"""
        self.buffer: str = ""
        self.prev_buffer_len: int = 0
        self.current_key: Optional[str] = None
        self.in_key: bool = False
        self.in_value: bool = False
        self.escape_next: bool = False
        self.unicode_chars: List[str] = []
        self.current_delta: str = ""

    def feed(self, chunk: str) -> Optional[Dict[str, str]]:
        """
        处理新的 chunk，返回增量 key-value 对

        Args:
            chunk: 新增的 JSON 片段

        Returns:
            返回格式: {"key": "delta_value"}，如果没有增量则返回 None
        """
        self.prev_buffer_len = len(self.buffer)
        self.buffer += chunk
        self.current_delta = ""

        i = self.prev_buffer_len
        while i < len(self.buffer):
            char = self.buffer[i]

            # 处理 Unicode 转义序列 \\uXXXX
            if self.unicode_chars:
                # 正在收集 Unicode 的 4 个十六进制字符
                self.unicode_chars.append(char)
                if len(self.unicode_chars) == 4:
                    # Unicode 序列完成，转换为字符
                    hex_str = ''.join(self.unicode_chars)
                    try:
                        unicode_char = chr(int(hex_str, 16))
                    except ValueError:
                        unicode_char = hex_str  # 转换失败，保持原样
                    if self.in_value and self.current_key:
                        self.current_delta += unicode_char
                    self.unicode_chars = []
                i += 1
                continue

            # 处理转义字符（当前字节在转义序列中）
            if self.escape_next:
                if self.in_value and self.current_key:
                    if char == 'u':
                        # Unicode 转义序列开始，收集后续 4 个字符
                        self.unicode_chars = []
                    else:
                        # 将转义序列转换为实际字符
                        escaped_char = self.ESCAPE_MAP.get(char, char)
                        self.current_delta += escaped_char
                self.escape_next = False
                i += 1
                continue

            if char == "\\" and self.in_value:
                # 遇到转义符号，标记下一个字符需要转义处理
                self.escape_next = True
                i += 1
                continue

            # 状态机逻辑
            # 跳过非关键字符，只关注 key 和 value 的字符串内容
            if char == '"':
                if not self.in_key and not self.in_value:
                    # 字符串开始，判断是 key 还是 value
                    if self.current_key is None:
                        self.in_key = True
                        self.current_key = ""
                    else:
                        self.in_value = True
                elif self.in_key:
                    # key 结束
                    self.in_key = False
                elif self.in_value:
                    # value 结束
                    self.in_value = False

            elif self.in_key:
                self.current_key += char
            elif self.in_value and self.current_key:
                # 只输出本次新增的部分
                self.current_delta += char

            i += 1

        return {self.current_key: self.current_delta} if self.current_key and self.current_delta else None

    def reset(self) -> None:
        """重置解析器以用于新的解析任务"""
        self.buffer = ""
        self.prev_buffer_len = 0
        self.current_key = None
        self.in_key = False
        self.in_value = False
        self.escape_next = False
        self.unicode_chars = []
        self.current_delta = ""