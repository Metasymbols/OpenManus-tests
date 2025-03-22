import math
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # 假设您的应用中设置了记录器
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)


REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # Default values when dimensions aren't available or detail level is unknown
        if detail == "high":
            # Default to a 1024x1024 image calculation for high detail
            return self._calculate_high_detail_tokens(1024, 1024)  # 765 tokens
        elif detail == "medium":
            # Default to a medium-sized image for medium detail
            return 1024  # This matches the original default
        else:
            # For unknown detail levels, use medium as default
            return 1024

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    """
    LLM客户端封装类，实现多实例管理、token计算和请求重试机制

    特性：
    - 支持Azure/OpenAI/Ollama等多种API类型
    - 自动化的token用量跟踪与限制检查
    - 消息格式标准化与验证
    - 指数退避重试策略
    """

    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        """
        单例模式构造函数

        参数：
            config_name: 配置名称，对应config.toml中的配置节
            llm_config: 可选的直接配置对象，优先级高于配置文件

        返回：
            LLM实例
        """
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        """
        初始化LLM客户端配置

        参数：
            config_name: 配置名称，对应config.toml中的配置节
            llm_config: 可选的直接配置对象，优先级高于配置文件

        抛出：
            ValueError: 当必需配置项缺失时
        """
        if not hasattr(self, "client"):  # 只有初始化的初始化
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # 添加令牌计数相关属性
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # 初始化令牌
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # 如果模型不在tiktoken的预设中，请使用cl100k_base默认
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                self.client = BedrockClient()
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)

    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量

        参数：
            text (str): 需要计算的文本内容

        返回：
            int: 文本对应的token数量
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        """
        计算消息列表的总token数量

        参数：
            messages (List[dict]): 消息字典列表，格式需符合OpenAI标准

        返回：
            int: 消息列表的总token数

        注意：
            计算规则包含角色、内容、工具调用等元素的token总和
        """
        token_count = 0
        for message in messages:
            # 每个消息的基本令牌计数（根据OpenAI的计算方法）
            token_count += 4  # 每个消息的基础令牌计数

            # 计算该角色的令牌
            if "role" in message:
                token_count += self.count_tokens(message["role"])

            # 计算内容的令牌
            if "content" in message and message["content"]:
                token_count += self.count_tokens(message["content"])

            # 计算工具调用的令牌
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    if "function" in tool_call:
                        # 功能名称
                        if "name" in tool_call["function"]:
                            token_count += self.count_tokens(
                                tool_call["function"]["name"]
                            )
                        # 函数参数
                        if "arguments" in tool_call["function"]:
                            token_count += self.count_tokens(
                                tool_call["function"]["arguments"]
                            )

            # 计算工具响应的令牌
            if "name" in message and message["name"]:
                token_count += self.count_tokens(message["name"])

            if "tool_call_id" in message and message["tool_call_id"]:
                token_count += self.count_tokens(message["tool_call_id"])

        # 为消息格式添加额外的令牌
        token_count += 2  # 消息格式的额外令牌

        return token_count

    def update_token_count(self, input_tokens: int) -> None:
        """Update token counts"""
        # 仅设置MAX_INPUT_TOKENS的轨道令牌
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """
        检查token用量是否超过限制

        参数：
            input_tokens (int): 当前请求的预估输入token数

        返回：
            bool: True表示未超限，False表示已超限
        """
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # 如果未设置max_input_tokens，请始终返回true
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # 如果消息是dict，请确保它具有所需字段
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else：不包括消息
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # 验证所有消息所需的字段
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # 不要重试tokenlimitexceed
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        发送请求到LLM并获取响应

        参数：
            messages: 对话消息列表
            system_msgs: 可选系统消息列表
            stream: 是否启用流式响应
            temperature: 采样温度值

        返回：
            str: 生成的文本响应

        抛出：
            TokenLimitExceeded: token用量超过配置限制
            ValueError: 消息格式错误或响应为空
            OpenAIError: API调用失败
        """
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # 格式系统和用户消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入令牌计数
            input_tokens = self.count_message_tokens(messages)

            # 检查是否超出了令牌限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 提出一个不会重述的特殊例外
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # 非流程请求
                params["stream"] = False

                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # 更新令牌计数
                self.update_token_count(response.usage.prompt_tokens)

                return response.choices[0].message.content

            # 流媒体请求，用于流媒体，更新估计的令牌计数，然后提出请求
            self.update_token_count(input_tokens)

            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content if isinstance(content, list) else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # 流媒体后的新线
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            # 重新汇总令牌限制错误而无需记录
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # 不要重试tokenlimitexceed
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # 类型：忽略
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        使用工具/函数调用方式请求LLM并返回响应

        参数：
            messages: 对话消息列表
            system_msgs: 可选系统消息列表
            timeout: 请求超时时间（秒）
            tools: 可用工具列表
            tool_choice: 工具选择策略
            temperature: 采样温度值
            **kwargs: 额外请求参数

        返回：
            ChatCompletionMessage: 模型响应对象

        抛出：
            TokenLimitExceeded: token用量超过配置限制
            ValueError: 消息格式错误或工具参数无效
            OpenAIError: API调用失败
        """
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # 验证tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # 格式消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入令牌计数
            input_tokens = self.count_message_tokens(messages)

            # 如果有工具，请计算工具描述的令牌计数
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # 检查是否超出了令牌限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 提出一个不会重述的特殊例外
                raise TokenLimitExceeded(error_message)

            # 验证工具如果提供
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # 设置完成请求
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            response: ChatCompletion = await self.client.chat.completions.create(
                **params, stream=False
            )

            # 检查响应是否有效
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # 更新令牌计数
            self.update_token_count(response.usage.prompt_tokens)

            return response.choices[0].message

        except TokenLimitExceeded:
            # 重新汇总令牌限制错误而无需记录
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
