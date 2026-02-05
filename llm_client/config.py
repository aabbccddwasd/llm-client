# -*- coding: utf-8 -*-
"""模型适配器模块 - 抽象不同模型特性的配置差异"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from .types import ToolDefinition, ResponseFormat


class ModelAdapter(ABC):
    """模型适配器抽象接口"""

    @abstractmethod
    def get_model_specific_params(
        self,
        enable_thinking: bool,
        clear_thinking: bool,
        stream: bool,
        tools: Optional[List[ToolDefinition]],
        response_format: Optional[ResponseFormat],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """
        获取模型特定的 API 参数

        Args:
            enable_thinking: 是否开启思考模式
            clear_thinking: 是否清空之前的思考
            stream: 是否流式输出
            tools: 工具定义列表
            response_format: 响应格式
            max_tokens: 最大 token 数

        Returns:
            包含所有需要传递给 API 参数的字典
        """
        pass

    @abstractmethod
    def get_base_params(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        stream: bool,
    ) -> Dict[str, Any]:
        """
        获取基础 API 参数

        Args:
            messages: 消息列表
            model_name: 模型名称
            stream: 是否流式输出

        Returns:
            基础 API 参数字典
        """
        pass


class BaseModelAdapter(ModelAdapter):
    """标准 OpenAI 兼容模型适配器"""

    def get_model_specific_params(
        self,
        enable_thinking: bool,
        clear_thinking: bool,
        stream: bool,
        tools: Optional[List[ToolDefinition]],
        response_format: Optional[ResponseFormat],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """标准 OpenAI 模型参数"""
        params = {}

        if response_format is not None:
            params["response_format"] = response_format

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if tools is not None:
            params["tools"] = tools

        return params

    def get_base_params(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        stream: bool,
    ) -> Dict[str, Any]:
        """标准基础参数"""
        return {
            "messages": messages,
            "model": model_name,
            "stream": stream,
        }


class GLMAdapter(ModelAdapter):
    """GLM-4.7 特定适配器 - 支持 thinking 模式和工具流式调用"""

    def get_model_specific_params(
        self,
        enable_thinking: bool,
        clear_thinking: bool,
        stream: bool,
        tools: Optional[List[ToolDefinition]],
        response_format: Optional[ResponseFormat],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """
        GLM-4.7 特定参数配置

        使用 extra_body 来传递 GLM 特有的参数：
        - chat_template_kwargs: 控制 thinking 模式
        - parallel_tool_calls: 启用并行工具调用
        """
        params = {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                    "clear_thinking": clear_thinking,
                },
                "parallel_tool_calls": True,
            }
        }

        if response_format is not None:
            params["response_format"] = response_format

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if tools is not None:
            params["tools"] = tools

        return params

    def get_base_params(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        stream: bool,
    ) -> Dict[str, Any]:
        """GLM 基础参数"""
        return {
            "messages": messages,
            "model": model_name,
            "stream": stream,
        }


# 模型名称到适配器的映射
MODEL_ADAPTER_MAP: Dict[str, ModelAdapter] = {
    # 标准适配器用于任何 OpenAI 兼容的模型
    "base": BaseModelAdapter(),
    # GLM-4.7 特定适配器
    "glm": GLMAdapter(),
}


def get_adapter_for_model(model_name: str) -> ModelAdapter:
    """
    根据模型名称获取对应的适配器

    Args:
        model_name: 模型名称

    Returns:
        对应的模型适配器实例

    Examples:
        >>> adapter = get_adapter_for_model("GLM-4.7-GPTQ-Int4-Int8Mix")
        >>> isinstance(adapter, GLMAdapter)
        True

        >>> adapter = get_adapter_for_model("Qwen3-VL-30B")
        >>> isinstance(adapter, BaseModelAdapter)
        True
    """
    # GLM 模型使用 GLM 适配器
    if "GLM" in model_name.upper():
        return MODEL_ADAPTER_MAP["glm"]

    # 其他模型使用标准适配器
    return MODEL_ADAPTER_MAP["base"]