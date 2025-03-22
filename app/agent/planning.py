import time
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.planning import NEXT_STEP_PROMPT, PLANNING_SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, Message, ToolCall, ToolChoice
from app.tool import PlanningTool, Terminate, ToolCollection


class PlanningAgent(ToolCallAgent):
    """计划管理代理（继承自ToolCallAgent）

    核心功能：
    - 创建和管理结构化任务计划
    - 跟踪计划步骤执行状态
    - 与计划工具（PlanningTool）深度集成实现进度跟踪
    """

    # Agent ID
    name: str = "planning"  # Agent name (fixed value)
    description: str = (
        "An agent that creates and manages plans to solve tasks"  # Function description (keep in English)
    )

    # Prompt word template
    system_prompt: str = (
        PLANNING_SYSTEM_PROMPT  # Special system prompts for planning management
    )
    next_step_prompt: str = NEXT_STEP_PROMPT  # Step decision prompt template

    # Tool configuration
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(PlanningTool(), Terminate())
    )  # The default is included with the Schedule Tool and the Termination Tool
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore  # 工具选择模式
    special_tool_names: List[str] = Field(
        default_factory=lambda: [Terminate().name]
    )  # Special tools whitelist

    # Plan status tracking
    tool_calls: List[ToolCall] = Field(
        default_factory=list
    )  # List of tool calls to be executed
    active_plan_id: Optional[str] = Field(default=None)  # Current activity plan id
    step_execution_tracker: Dict[str, Dict] = Field(
        default_factory=dict,
        description="步骤执行跟踪器（格式：{tool_call_id: {step_index, tool_name, status}}）",
    )  # Steps to perform status tracking
    current_step_index: Optional[int] = None  # Current Processing Step Index

    # Execute control
    max_steps: int = 20  # Override the default step limit of parent class

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

        # 在思考之前获取当前的步骤索引
        self.current_step_index = await self._get_current_step_index()

        result = await super().think()

        # 经过思考，如果我们决定执行工具，而不是计划工具或特殊工具，
        # 将其与当前跟踪步骤相关联
        if result and self.tool_calls:
            latest_tool_call = self.tool_calls[0]  # 获取最新的工具电话
            if (
                latest_tool_call.function.name != "planning"
                and latest_tool_call.function.name not in self.special_tool_names
                and self.current_step_index is not None
            ):
                self.step_execution_tracker[latest_tool_call.id] = {
                    "step_index": self.current_step_index,
                    "tool_name": latest_tool_call.function.name,
                    "status": "pending",  # 执行后将进行更新
                }

        return result

    async def act(self) -> str:
        """计划感知的执行阶段
        增强功能：
        - 自动更新步骤执行状态
        - 同步计划进度到存储
        """
        result = await super().act()

        # 执行工具后，更新计划状态
        if self.tool_calls:
            latest_tool_call = self.tool_calls[0]

            # 将执行状态更新到已完成
            if latest_tool_call.id in self.step_execution_tracker:
                self.step_execution_tracker[latest_tool_call.id]["status"] = "completed"
                self.step_execution_tracker[latest_tool_call.id]["result"] = result

                # 如果这是一个非计划的非特殊工具，则更新计划状态
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
            # 将步骤标记为已完成
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

            # 找到“步骤：”行的索引
            for i, line in enumerate(plan_lines):
                if line.strip() == "Steps:":
                    steps_index = i
                    break

            if steps_index == -1:
                return None

            # 找到第一个未完成的步骤
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

            return None  # 找不到主动步骤
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

                # 将工具响应添加到内存
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
    # 配置并运行代理
    agent = PlanningAgent(available_tools=ToolCollection(PlanningTool(), Terminate()))
    result = await agent.run("Help me plan a trip to the moon")
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
