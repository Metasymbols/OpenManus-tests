import json
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanStepStatus(str, Enum):
    """计划步骤状态枚举

    定义了计划步骤可能的状态值。

    Attributes:
        NOT_STARTED: 未开始的步骤
        IN_PROGRESS: 正在执行的步骤
        COMPLETED: 已完成的步骤
        BLOCKED: 被阻塞的步骤
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """获取所有可能的步骤状态值

        Returns:
            包含所有状态值的列表
        """
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """获取表示活动状态的值列表

        Returns:
            包含未开始和进行中状态的列表
        """
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """获取状态到标记符号的映射

        Returns:
            状态到对应标记符号的字典映射
        """
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """规划流程类

    管理任务的规划和执行的流程类，使用代理来完成具体任务。

    Attributes:
        llm: 语言模型实例
        planning_tool: 规划工具实例
        executor_keys: 执行器代理的标识键列表
        active_plan_id: 当前活动计划的ID
        current_step_index: 当前执行步骤的索引
    """

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # 处理executors键
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # 设置计划ID（如果提供）
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 初始化计划工具（如果不提供）
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # 用处理的数据调用parents.__init__
        super().__init__(agents, **data)

        # 如果未指定
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """获取当前步骤的合适执行器代理

        根据步骤类型选择合适的执行器代理。此方法可以根据步骤类型或需求进行扩展。

        Args:
            step_type: 步骤类型，可选

        Returns:
            选择的执行器代理实例
        """
        # 如果提供了步骤类型并与代理密钥匹配，请使用该代理
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 否则，请使用第一个可用的执行人或倒回主代理
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 向主要代理的后备
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """执行规划流程

        使用代理执行规划流程，包括创建初始计划和执行各个步骤。

        Args:
            input_text: 输入的任务文本

        Returns:
            执行结果文本

        Raises:
            ValueError: 当没有可用的主要代理时抛出
        """
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # 如果提供输入，请创建初始计划
            if input_text:
                await self._create_initial_plan(input_text)

                # 验证计划已成功创建
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"计划创建失败。在规划工具中未找到计划ID {self.active_plan_id}。"
                    )
                    return f"未能为以下任务创建计划：{input_text}"

            result = ""
            while True:
                # 获取当前执行的步骤
                self.current_step_index, step_info = await self._get_current_step_info()

                # 如果没有更多的步骤或计划，请退出
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 使用适当的代理执行当前步骤
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += (step_result if step_result is not None else "") + "\n"

                # 检查代理是否要终止
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e) if e is not None else 'Unknown error'}"

    async def _create_initial_plan(self, request: str) -> None:
        """创建初始计划

        使用流程的语言模型和规划工具根据请求创建初始计划。

        Args:
            request: 用户请求文本

        Returns:
            None

        Note:
            如果计划创建失败，会创建一个包含基本步骤的默认计划
        """
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        # 为计划创建创建系统消息
        system_message = Message.system_message(
            "您是一个规划助手。请创建简洁、可执行的计划，明确步骤。"
            "重点关注关键里程碑而非详细子步骤。"
            "优化方案的清晰度和效率。"
        )

        # 用请求创建用户消息
        user_message = Message.user_message(
            f"请为以下任务创建一个合理的计划，明确步骤：{request}"
        )

        # 致电LLM与Plansphertool致电
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 处理工具调用如果存在
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析论点
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {args}")
                            continue

                    # 确保正确设置plan_id并执行工具
                    args["plan_id"] = self.active_plan_id

                    # 通过ToolCollection而不是直接执行工具
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"Plan creation result: {str(result)}")
                    return

        # 如果在此处执行执行，请创建一个默认计划
        logger.warning("Creating default plan")

        # 使用工具收集创建默认计划
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """获取当前步骤信息

        解析当前计划以识别第一个未完成步骤的索引和信息。

        Returns:
            包含步骤索引和信息的元组，如果没有活动步骤则返回(None, None)

        Note:
            步骤信息字典包含步骤文本和可选的步骤类型
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            # 直接访问计划工具存储的计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤类型/类别如果可用
                    step_info = {"text": step}

                    # 尝试从文本中提取步骤类型（例如[搜索]或[代码]）
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 将当前步骤标记为in_progress
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        # 如果需要，直接更新步骤状态
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # 找不到主动步骤

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """执行当前步骤

        使用指定的代理执行当前步骤，通过agent.run()方法实现。

        Args:
            executor: 执行步骤的代理实例
            step_info: 步骤信息字典

        Returns:
            步骤执行结果文本

        Note:
            执行成功后会自动将步骤标记为已完成
        """
        # 为具有当前计划状态的代理准备上下文
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # 为代理执行当前步骤创建提示
        step_prompt = f"""
        当前计划状态:
        {plan_status}

        当前任务:
        正在执行第 {self.current_step_index} 步: "{step_text}"

        请使用合适的工具执行该步骤。完成后请提供本次步骤的执行总结。
        """

        # 使用Agent.run（）执行步骤
        try:
            step_result = await executor.run(step_prompt)

            # 成功执行后将步骤标记为完成
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error executing step {self.current_step_index}: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """标记当前步骤为已完成

        将当前步骤的状态更新为已完成。如果更新失败，会尝试直接修改计划工具存储中的状态。

        Note:
            此方法会同时更新计划工具中的步骤状态和内存中的状态记录
        """
        if self.current_step_index is None:
            return

        try:
            # 将步骤标记为已完成
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")
            # 直接在计划工具存储中更新步骤状态
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # 确保step_statuses列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """获取当前计划的格式化文本

        Returns:
            格式化的计划文本

        Note:
            如果通过计划工具获取失败，会尝试直接从存储生成文本
        """
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """从存储生成计划文本

        当计划工具获取失败时，直接从存储中生成计划文本。

        Returns:
            生成的计划文本

        Note:
            包含计划标题、步骤列表和每个步骤的状态
        """
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"错误：使用ID计划 {self.active_plan_id} 未找到"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # 确保step_statuses和step_notes匹配步骤数
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 计算状态的步骤
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"计划: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += f"进步: {completed}/{total} 步骤完成({progress:.1f}%)\n"
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # 使用状态标记表示步骤状态
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"错误：无法使用ID检索计划 {self.active_plan_id}"

    async def _finalize_plan(self) -> str:
        """Finalize the plan and provide a summary using the flow's LLM directly."""
        plan_text = await self._get_plan_text()

        # 直接使用流的LLM创建摘要
        try:
            system_message = Message.system_message(
                "您是一个规划助手。您的任务是总结已完成的计划。"
            )

            user_message = Message.user_message(
                f"计划已完成。以下是最终计划状态：\n\n{plan_text}\n\n请提供所完成内容的摘要以及最终想法。"
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"计划完成:\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")

            # 退缩到使用代理进行摘要
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                计划已完成。以下是最终计划状态：

                {plan_text}

                请提供所完成内容的摘要以及最终想法。
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"Error finalizing plan with agent: {e2}")
                return "计划完成。错误生成摘要。"
