import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall_zh import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "需要工具调用但未提供"


class ToolCallAgent(ReActAgent):
    """工具调用代理（基于ReAct框架的具体实现）

    核心功能：
    - 实现复杂的工具调用流程管理
    - 支持多种工具选择模式（自动/强制/禁用）
    - 提供特殊工具（如终止工具）的定制处理
    """

    # 基础配置
    name: str = "toolcall"  # 代理名称（固定值）
    description: str = "一个可以执行工具调用的代理。"  # 功能描述

    # 提示词模板
    system_prompt: str = SYSTEM_PROMPT  # 系统级指令模板
    next_step_prompt: str = NEXT_STEP_PROMPT  # 步骤决策提示模板

    # Tool management related
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )  # Collection of available tools (chain and terminating tools are used by default)
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
        None  # The result is the truncated length (no truncated representation)
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
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # 检查这是否是一个包含Tokenlimitexceeded的重试
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(f"🚨 令牌限制错误（来自重试错误）：{token_limit_error}")
                self.memory.add_message(
                    Message.assistant_message(
                        f"已达到最大令牌限制，无法继续执行：{str(token_limit_error)}"
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
        logger.info(f"✨ {self.name}的思考：{response.content}")
        logger.info(
            f"🛠️ {self.name}选择使用了{len(tool_calls) if tool_calls else 0}个工具"
        )
        if tool_calls:
            logger.info(
                f"🧰 正在准备的工具：{[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 工具参数：{tool_calls[0].function.arguments}")

        try:
            # 处理不同的工具选择模式
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(f"🤔 嗯，{self.name}尝试使用了不可用的工具！")
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
                return True  # 将在ACT（）中进行处理

            # 对于“自动”模式，如果没有命令，请继续使用内容
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 糟糕！{self.name}的思考过程遇到了问题：{e}")
            self.memory.add_message(
                Message.assistant_message(f"处理过程中遇到错误：{str(e)}")
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

            # 如果没有工具调用，请返回最后一条消息
            return self.messages[-1].content or "没有内容或命令可执行"

        results = []
        for command in self.tool_calls:
            # Reset the base64 image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(f"🎯 工具'{command.function.name}'完成了任务！结果：{result}")

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
        - 增强参数验证机制
        - 完善错误处理流程
        - 优化结果格式化输出
        - 支持工具生命周期管理
        """
        # 基础验证
        if not command or not command.function or not command.function.name:
            logger.error("Invalid command format: missing required fields")
            return "Error: Invalid command format - missing required fields"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            logger.error(f"Unknown tool requested: {name}")
            return f"Error: Unknown tool '{name}' - tool not registered"

        try:
            # 参数解析与验证
            args = {}
            if command.function.arguments:
                try:
                    args = json.loads(command.function.arguments)
                    if not isinstance(args, dict):
                        raise ValueError("Arguments must be a JSON object")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {str(e)}")
                    return f"Error: Invalid JSON format in arguments - {str(e)}"
                except ValueError as e:
                    logger.error(f"Arguments validation error: {str(e)}")
                    return f"Error: {str(e)}"

            # 工具执行前处理
            tool = self.available_tools.tool_map[name]
            if hasattr(tool, "pre_execute") and callable(getattr(tool, "pre_execute")):
                await tool.pre_execute()

            # 执行工具
            logger.info(f"🔧 Activating tool: '{name}' with validated arguments")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 结果格式化
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                result_str = str(result)

            observation = (
                f"Observed output of cmd `{name}` executed:\n{result_str}"
                if result_str
                else f"Cmd `{name}` completed with no output"
            )

            # 工具执行后处理
            if hasattr(tool, "post_execute") and callable(
                getattr(tool, "post_execute")
            ):
                await tool.post_execute(result)

            # 特殊工具处理
            await self._handle_special_tool(name=name, result=result)

            return observation

        except json.JSONDecodeError as e:
            error_msg = f"Error parsing arguments for {name}: {str(e)}"
            logger.error(f"📝 JSON parse error: {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' execution failed: {str(e)}"
            logger.error(f"Tool execution error: {error_msg}")
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """特殊工具后处理
        功能：
        - 增强特殊工具生命周期管理
        - 支持工具状态跟踪和清理
        - 提供可扩展的终止条件判断
        """
        if not self._is_special_tool(name):
            return

        try:
            # 获取工具实例
            tool = self.available_tools.tool_map[name]

            # 工具状态跟踪
            if hasattr(tool, "status") and callable(getattr(tool, "status")):
                tool_status = await tool.status()
                logger.info(f"Special tool '{name}' status: {tool_status}")

            # 工具资源清理
            if hasattr(tool, "cleanup") and callable(getattr(tool, "cleanup")):
                await tool.cleanup()
                logger.info(f"Special tool '{name}' resources cleaned up")

            # 终止条件判断
            if self._should_finish_execution(
                name=name, result=result, tool=tool, **kwargs
            ):
                logger.info(f"🏁 Special tool '{name}' has completed the task!")
                self.state = AgentState.FINISHED

        except Exception as e:
            logger.error(f"Error handling special tool '{name}': {str(e)}")

    def _should_finish_execution(
        self, name: str, result: Any, tool: Any = None, **kwargs
    ) -> bool:
        """终止条件判断（支持自定义终止逻辑）
        功能：
        - 支持基于工具状态的终止判断
        - 允许子类扩展终止条件
        - 提供默认终止行为
        """
        # 检查工具是否定义了自己的终止条件
        if (
            tool
            and hasattr(tool, "should_terminate")
            and callable(getattr(tool, "should_terminate"))
        ):
            return tool.should_terminate(result)

        # 默认终止行为
        return True

    def _is_special_tool(self, name: str) -> bool:
        """特殊工具校验（增强版）
        功能：
        - 支持大小写不敏感的名称匹配
        - 验证工具实例的有效性
        """
        if not name:
            return False

        name_lower = name.lower()
        return (
            name_lower in [n.lower() for n in self.special_tool_names]
            and name in self.available_tools.tool_map
        )
