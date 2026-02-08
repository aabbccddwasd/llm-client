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

    def embed(
        self,
        input_data: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]],
        model_name: str,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        实现 embedding 调用，处理图片编码

        Args:
            input_data: 输入数据（文本字符串、文本列表、或图文混合结构）
            model_name: 模型名称
            extra_body: 额外的 API 参数（用于传图片）

        Returns:
            embedding 向量或向量列表

        Raises:
            ClientError: 当调用失败时
        """
        try:
            # 处理 extra_body 中的 PIL 图片和 OpenAIMessageBlock 格式的图片
            processed_extra = self._process_images_in_extra_body(extra_body) if extra_body else None

            # 调用 API
            request_params = {"model": model_name, "input": input_data}
            if processed_extra:
                request_params["extra_body"] = processed_extra

            self.logger.debug(f"Calling embedding API [{self.label}]: model={model_name}")

            response = self.client.embeddings.create(**request_params)

            # 返回结果（单输入返回 list[float]，批量输入返回 list[list[float]]）
            if len(response.data) == 1:
                return response.data[0].embedding
            return [item.embedding for item in response.data]

        except Exception as e:
            self.logger.error(f"Embedding client error [{self.label}]: {str(e)}", exc_info=True)
            raise ClientError(f"Embedding API error: {str(e)}") from e

    def _process_images_in_extra_body(self, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 extra_body 中的图片，将 PIL/Image 和其他格式统一转为 base64

        支持的图片格式：
        - PIL.Image 对象
        - data:image/png;base64,... (已编码)
        - http://... 或 https://... URL 字符串

        Args:
            extra_body: 包含 image 参数的字典

        Returns:
            处理后的 extra_body
        """
        images = extra_body.get("image")
        if not images:
            return extra_body

        # 统一处理为列表
        if not isinstance(images, list):
            images = [images]

        processed = []
        for img in images:
            processed.extend(self._encode_single_image(img))

        return {**extra_body, "image": processed if len(processed) > 1 else processed[0]}

    def _encode_single_image(self, image_input: Any) -> List[str]:
        """
        编码单个图片或图片列表

        Args:
            image_input: PIL.Image、图片路径URL、base64 字符串，或它们的列表

        Returns:
            base64 图片 URL 列表
        """
        import base64
        import io

        processed = []

        # 如果输入是列表，递归处理
        if isinstance(image_input, list):
            for item in image_input:
                processed.extend(self._encode_single_image(item))
            return processed

        # 检查是否是 PIL Image
        try:
            # 使用不导入的方式检查，避免导入失败时中断
            image_module = __import__("PIL", fromlist=["Image"])
            if isinstance(image_input, image_module.Image.Image):
                # PIL.Image 转为 base64（避免显示：直接读取 bytes）
                buffered = io.BytesIO()
                image_input.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return [f"data:image/png;base64,{img_base64}"]
        except ImportError:
            # PIL 不可用时跳过
            pass
        except (AttributeError, TypeError):
            pass

        # 如果是字符串，检查格式
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                # 已经是 base64 格式
                return [image_input]
            elif image_input.startswith(("http://", "https://")):
                # URL 格式，直接返回（vLLM 可能支持）
                return [image_input]
            else:
                # 假设是纯 base64 数据，添加前缀
                return [f"data:image/png;base64,{image_input}"]

        # 其他情况，直接返回
        return [image_input]