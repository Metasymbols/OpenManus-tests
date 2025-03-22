import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """工具调用代理（基于ReAct框架的具体实现）

    核心功能：
    - 实现复杂的工具调用流程管理
    - 支持多种工具选择模式（自动/强制/禁用）
    - 提供特殊工具（如终止工具）的定制处理
    """

    # Basic configuration
    name: str = "toolcall"  # Agent name (fixed value)
    description: str = "an agent that can execute tool calls."  # Function description

    # Prompt word template
    system_prompt: str = SYSTEM_PROMPT  # System-level instruction templates
    next_step_prompt: str = NEXT_STEP_PROMPT  # Step decision prompt template

    # Tool management related
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )  # Collection of available tools (with chat and termination tools by default)
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore  # 工具选择模式（auto/required/none）
    special_tool_names: List[str] = Field(
        default_factory=lambda: [Terminate().name]
    )  # Special tools whitelist

    # Runtime status
    tool_calls: List[ToolCall] = Field(
        default_factory=list
    )  # List of currently pending tool calls
    max_steps: int = (
        30  # Maximum execution steps (override the base class default value)
    )
    max_observe: Optional[Union[int, bool]] = (
        None  # The result is truncated length (none means no truncated)
    )

    async def think(self) -> bool:
        """思考阶段实现（处理LLM响应并准备工具调用）

        执行流程：
        1. 调用LLM获取带工具调用的响应
        2. 处理token超限等异常情况
        3. 根据工具选择模式验证响应有效性
        4. 准备工具调用消息存入记忆
        """
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # 通过工具选项获取响应
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # 检查这是否是包含Tokenlimitexceeded的重试
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # 日志响应信息
        logger.info(f"✨ {self.name}'s thoughts: {response.content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            # 处理不同的工具_CHOICES模式
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # 创建并添加助手消息
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # 将在ACT（）中处理

            # 对于“自动”模式，如果没有命令，则继续使用内容
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """行动阶段实现（执行所有待处理工具调用）

        执行流程：
        1. 遍历执行所有已准备的工具调用
        2. 截断超长结果（根据max_observe配置）
        3. 记录工具执行结果到记忆
        4. 返回所有结果的拼接字符串
        """
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # 如果没有工具调用，请返回最后一条消息内容
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # 将工具响应添加到内存
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """单工具执行器（含完整错误处理）

        实现要点：
        - 参数解析使用JSON格式
        - 自动处理特殊工具（如终止工具）
        - 统一错误消息格式
        - 详细记录执行日志
        """
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # 解析论点
            args = json.loads(command.function.arguments or "{}")

            # 执行工具
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 格式结果显示
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            # 处理特殊工具，例如``完成''
            await self._handle_special_tool(name=name, result=result)

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """特殊工具后处理
        功能：
        - 检查是否为特殊工具（通过白名单）
        - 触发代理状态变更（如执行终止工具后设为FINISHED）
        """
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # 设置代理状态完成
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """终止条件判断（默认总是终止）
        子类可重写此方法实现自定义终止逻辑
        """
        return True

    def _is_special_tool(self, name: str) -> bool:
        """特殊工具校验
        通过名称大小写不敏感匹配白名单
        """
        return name.lower() in [n.lower() for n in self.special_tool_names]
