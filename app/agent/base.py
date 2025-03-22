from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """代理抽象基类，用于管理代理状态和执行流程

    提供状态转换、记忆管理的基础功能，以及基于步骤的执行循环。
    子类必须实现 step 方法来实现具体业务逻辑。
    """

    # 核心属性
    name: str = Field(..., description="代理的唯一名称")
    description: Optional[str] = Field(None, description="代理的详细描述（可选）")

    # 提示词配置
    system_prompt: Optional[str] = Field(
        None, description="系统级指令提示词，用于指导代理的基础行为"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="下一步行动决策提示词模板"
    )

    # 依赖组件
    llm: LLM = Field(default_factory=LLM, description="语言模型实例")
    memory: Memory = Field(default_factory=Memory, description="记忆存储模块")
    state: AgentState = Field(
        default=AgentState.IDLE, description="当前运行状态，详见AgentState枚举"
    )

    # 执行控制参数
    max_steps: int = Field(default=10, description="最大执行步骤数，防止无限循环")
    current_step: int = Field(default=0, description="当前执行步骤计数器")
    duplicate_threshold: int = 2  # 重复内容检测阈值，达到该值视为卡死状态

    class Config:
        arbitrary_types_allowed = True  # 允许非Pydantic类型
        extra = "allow"  # 允许子类扩展字段

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """组件初始化校验器"""
        # 确保LLM实例正确初始化
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())  # 使用小写的代理名作为配置名

        # 初始化记忆存储
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """状态管理上下文（线程安全）

        用法：
            async with self.state_context(AgentState.RUNNING):
                # 在此代码块中代理将保持RUNNING状态
                await do_something()

        异常处理：
            当代码块内发生异常时，状态会被标记为ERROR
            退出上下文后自动恢复原始状态
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """记忆更新方法
        参数：
            role - 消息角色（user/system/assistant/tool）
            content - 消息内容
            kwargs - 工具消息的额外参数（如tool_call_id）
        """
        # 消息工厂映射（使用Message类的静态方法创建消息对象）
        message_map = {
            "user": Message.user_message,  # 用户消息
            "system": Message.system_message,  # 系统消息
            "assistant": Message.assistant_message,  # 代理消息
            "tool": lambda content, **kw: Message.tool_message(
                content, **kw
            ),  # 工具调用消息
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """代理主运行循环
        执行流程：
        1. 检查初始状态必须为IDLE
        2. 如果有初始请求，存入记忆
        3. 进入RUNNING状态循环执行step()
        4. 达到最大步数或FINISHED状态时终止

        返回：
            包含所有步骤结果的字符串（按步骤换行）
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                # 记录步骤执行日志
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()  # 核心步骤调用

                # 卡死状态检测与处理
                if self.is_stuck():
                    self.handle_stuck_state()  # 自动追加提示词

                results.append(f"Step {self.current_step}: {step_result}")

            # 最大步数保护机制
            if self.current_step >= self.max_steps:
                self.current_step = 0  # 重置步数计数器
                self.state = AgentState.IDLE  # 回归空闲状态
                results.append(f"Terminated: Reached max steps ({self.max_steps})")

    def is_stuck(self) -> bool:
        """卡死状态检测逻辑
        实现原理：
        1. 检查最近两条消息
        2. 统计历史消息中相同assistant消息的出现次数
        3. 当重复次数达到阈值时返回True

        注意：
            只检测assistant角色的消息内容重复
            工具消息和用户消息不计入检测
        """
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """记忆消息访问器（属性方式访问memory.messages）"""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """记忆消息设置器（提供直接赋值接口）"""
        self.memory.messages = value

    @abstractmethod
    async def step(self) -> str:
        """单步执行抽象方法（必须由子类实现）

        实现要求：
        - 每个step()应完成一个完整的工作单元
        - 返回该步骤的执行结果摘要
        - 更新记忆或状态需通过对应方法操作
        """

    def handle_stuck_state(self):
        """卡死状态处理策略
        实现方式：
        - 在提示词前追加防重复策略说明
        - 通过日志记录卡死事件
        - 保持原有提示词内容不变，仅做前置拼接
        """
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = (
            f"{stuck_prompt}\n{self.next_step_prompt}"  # 前置拼接新提示
        )
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")
