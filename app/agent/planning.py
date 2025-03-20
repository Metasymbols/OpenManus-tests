import time
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.planning_zh import NEXT_STEP_PROMPT, PLANNING_SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, Message, ToolCall, ToolChoice
from app.tool import PlanningTool, Terminate, ToolCollection


class PlanningAgent(ToolCallAgent):
    """计划管理代理（继承自ToolCallAgent）

    核心功能：
    - 创建和管理结构化任务计划
    - 跟踪计划步骤执行状态
    - 与计划工具（PlanningTool）深度集成实现进度跟踪
    """

    # 代理ID
    name: str = "planning"  # 代理名称（固定值）
    description: str = (
        "An agent that creates and manages plans to solve tasks"  # 功能描述（用英语保留）
    )

    # 提示单词模板
    system_prompt: str = PLANNING_SYSTEM_PROMPT  # 特殊系统提示计划管理
    next_step_prompt: str = NEXT_STEP_PROMPT  # 步骤决策提示模板

    # 工具配置
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(PlanningTool(), Terminate())
    )  # 计划工具和终止工具中包含默认值
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore  # 工具选择模式
    special_tool_names: List[str] = Field(
        default_factory=lambda: [Terminate().name]
    )  # 专用工具白名单

    # 计划状态跟踪
    tool_calls: List[ToolCall] = Field(default_factory=list)  # 要执行的工具调用列表
    active_plan_id: Optional[str] = Field(default=None)  # 当前活动计划ID
    step_execution_tracker: Dict[str, Dict] = Field(
        default_factory=dict,
        description="步骤执行跟踪器（格式：{tool_call_id: {step_index, tool_name, status}}）",
    )  # 执行状态跟踪的步骤
    current_step_index: Optional[int] = None  # 当前的处理步骤索引

    # 执行控制
    max_steps: int = 20  # 覆盖父类的默认步骤限制

    @model_validator(mode="after")
    def initialize_plan_and_verify_tools(self) -> "PlanningAgent":
        """初始化验证器
        功能：
        - 生成唯一计划ID
        - 确保计划工具已注册
        """
        self.active_plan_id = f"plan_{int(time.time())}"
        if "planning" not in self.available_tools.tool_map:
            self.available_tools.add_tool(PlanningTool())
        return self

    async def think(self) -> bool:
        """计划感知的思考阶段
        增强功能：
        - 自动注入当前计划状态到提示词
        - 记录当前步骤索引用于后续跟踪
        """
        prompt = (
            f"CURRENT PLAN STATUS:\n{await self.get_plan()}\n\n{self.next_step_prompt}"
            if self.active_plan_id
            else self.next_step_prompt
        )
        self.messages.append(Message.user_message(prompt))

        # Get the current step index before thinking
        self.current_step_index = await self._get_current_step_index()

        result = await super().think()

        # After thinking, if we decide to execute tools, rather than planning tools or special tools,
        # Associate it with the current tracking step
        if result and self.tool_calls:
            latest_tool_call = self.tool_calls[0]  # Get the latest tool phone
            if (
                latest_tool_call.function.name != "planning"
                and latest_tool_call.function.name not in self.special_tool_names
                and self.current_step_index is not None
            ):
                self.step_execution_tracker[latest_tool_call.id] = {
                    "step_index": self.current_step_index,
                    "tool_name": latest_tool_call.function.name,
                    "status": "pending",  # Updates will be performed after execution
                }

        return result

    async def act(self) -> str:
        """计划感知的执行阶段
        增强功能：
        - 自动更新步骤执行状态
        - 同步计划进度到存储
        """
        result = await super().act()

        # After executing the tool, update the plan status
        if self.tool_calls:
            latest_tool_call = self.tool_calls[0]

            # Update execution status to completed
            if latest_tool_call.id in self.step_execution_tracker:
                self.step_execution_tracker[latest_tool_call.id]["status"] = "completed"
                self.step_execution_tracker[latest_tool_call.id]["result"] = result

                # If this is a non-planned, non-special tool, update the plan status
                if (
                    latest_tool_call.function.name != "planning"
                    and latest_tool_call.function.name not in self.special_tool_names
                ):
                    await self.update_plan_status(latest_tool_call.id)

        return result

    async def get_plan(self) -> str:
        """获取当前计划状态
        返回：
           格式化后的计划文本，包含所有步骤及其状态
        """
        if not self.active_plan_id:
            return "No active plan. Please create a plan first."

        result = await self.available_tools.execute(
            name="planning",
            tool_input={"command": "get", "plan_id": self.active_plan_id},
        )
        return result.output if hasattr(result, "output") else str(result)

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with an optional initial request."""
        if request:
            await self.create_initial_plan(request)
        return await super().run()

    async def update_plan_status(self, tool_call_id: str) -> None:
        """更新计划进度
        逻辑：
        1. 仅当工具调用成功完成时更新
        2. 通过PlanningTool标记步骤状态
        3. 记录操作日志
        """
        if not self.active_plan_id:
            return

        if tool_call_id not in self.step_execution_tracker:
            logger.warning(f"No step tracking found for tool call {tool_call_id}")
            return

        tracker = self.step_execution_tracker[tool_call_id]
        if tracker["status"] != "completed":
            logger.warning(f"Tool call {tool_call_id} has not completed successfully")
            return

        step_index = tracker["step_index"]

        try:
            # Mark the step as completed
            await self.available_tools.execute(
                name="planning",
                tool_input={
                    "command": "mark_step",
                    "plan_id": self.active_plan_id,
                    "step_index": step_index,
                    "step_status": "completed",
                },
            )
            logger.info(
                f"Marked step {step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")

    async def _get_current_step_index(self) -> Optional[int]:
        """
        Parse the current plan to identify the first non-completed step's index.
        Returns None if no active step is found.
        """
        if not self.active_plan_id:
            return None

        plan = await self.get_plan()

        try:
            plan_lines = plan.splitlines()
            steps_index = -1

            # Find the index of the "Step:" line
            for i, line in enumerate(plan_lines):
                if line.strip() == "Steps:":
                    steps_index = i
                    break

            if steps_index == -1:
                return None

            # Find the first unfinished step
            for i, line in enumerate(plan_lines[steps_index + 1 :], start=0):
                if "[ ]" in line or "[→]" in line:  # not_started或in_progress
                    # 将当前步骤标记为in_progress
                    await self.available_tools.execute(
                        name="planning",
                        tool_input={
                            "command": "mark_step",
                            "plan_id": self.active_plan_id,
                            "step_index": i,
                            "step_status": "in_progress",
                        },
                    )
                    return i

            return None  # No active steps found
        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None

    async def create_initial_plan(self, request: str) -> None:
        """Create an initial plan based on the request."""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        messages = [
            Message.user_message(
                f"Analyze the request and create a plan with ID {self.active_plan_id}: {request}"
            )
        ]
        self.memory.add_messages(messages)
        response = await self.llm.ask_tool(
            messages=messages,
            system_msgs=[Message.system_message(self.system_prompt)],
            tools=self.available_tools.to_params(),
            tool_choice=ToolChoice.AUTO,
        )
        assistant_msg = Message.from_tool_calls(
            content=response.content, tool_calls=response.tool_calls
        )

        self.memory.add_message(assistant_msg)

        plan_created = False
        for tool_call in response.tool_calls:
            if tool_call.function.name == "planning":
                result = await self.execute_tool(tool_call)
                logger.info(
                    f"Executed tool {tool_call.function.name} with result: {result}"
                )

                # Add tool response to memory
                tool_msg = Message.tool_message(
                    content=result,
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                )
                self.memory.add_message(tool_msg)
                plan_created = True
                break

        if not plan_created:
            logger.warning("No plan created from initial request")
            tool_msg = Message.assistant_message(
                "Error: Parameter `plan_id` is required for command: create"
            )
            self.memory.add_message(tool_msg)


async def main():
    # Configure and run the agent
    agent = PlanningAgent(available_tools=ToolCollection(PlanningTool(), Terminate()))
    result = await agent.run("Help me plan a trip to the moon")
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
