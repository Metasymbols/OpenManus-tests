from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    """ReAct框架具体实现（继承自BaseAgent）

    特性：
    - 实现classical thinking-action cycle（Think-Act Cycle）
    - 提供默认的step()方法实现
    - 要求子类必须实现think()和act()方法
    """

    # 核心属性（继承自BaseAgent）
    name: str  # 必须设置的代理标识
    description: Optional[str] = None  # 代理功能描述

    # 提示词配置
    system_prompt: Optional[str] = None  # ReAct系统提示模板
    next_step_prompt: Optional[str] = None  # 下一步决策提示模板

    # 依赖组件
    llm: Optional[LLM] = Field(default_factory=LLM)  # 语言模型实例
    memory: Memory = Field(default_factory=Memory)  # 记忆存储
    state: AgentState = AgentState.IDLE  # 初始状态为IDLE

    # 执行控制
    max_steps: int = 10  # 最大迭代次数
    current_step: int = 0  # 当前步骤计数器

    @abstractmethod
    async def think(self) -> bool:
        """思考阶段抽象方法（必须实现）

        实现要求：
        - 分析当前记忆和环境状态
        - 返回布尔值表示是否需要执行行动
        - 可通过修改next_step_prompt影响决策
        """

    @abstractmethod
    async def act(self) -> str:
        """行动阶段抽象方法（必须实现）

        实现要求：
        - 执行具体的工具调用或外部操作
        - 返回行动结果摘要
        - 需自行处理异常情况
        """

    async def step(self) -> str:
        """执行完整的思考-行动循环单步
        执行流程：
        1. 调用think()决定是否需要行动
        2. 需要行动时调用act()执行
        3. 返回阶段结果摘要
        """
        should_act = await self.think()
        if not should_act:
            return "思考阶段完成，无需执行行动"  # 中文日志输出
        return await self.act()
