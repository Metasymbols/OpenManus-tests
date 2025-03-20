from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from app.agent.base import BaseAgent


class BaseFlow(BaseModel, ABC):
    """执行流程基类

    支持多代理的执行流程基类，提供了代理管理和流程执行的基本功能。

    Attributes:
        agents: 代理实例字典，键为代理标识
        tools: 可选的工具列表
        primary_agent_key: 主要代理的标识键
    """

    agents: Dict[str, BaseAgent]
    tools: Optional[List] = None
    primary_agent_key: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # 处理提供代理商的不同方式
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}
        else:
            agents_dict = agents

        # 如果未指定主要代理，请使用第一代理
        primary_key = data.get("primary_agent_key")
        if not primary_key and agents_dict:
            primary_key = next(iter(agents_dict))
            data["primary_agent_key"] = primary_key

        # 设置代理词典
        data["agents"] = agents_dict

        # 使用basemodel的init初始化
        super().__init__(**data)

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        """获取流程的主要代理

        Returns:
            主要代理实例，如果未设置则返回None
        """
        return self.agents.get(self.primary_agent_key)

    def get_agent(self, key: str) -> Optional[BaseAgent]:
        """通过键获取特定代理

        Args:
            key: 代理标识键

        Returns:
            指定的代理实例，如果不存在则返回None
        """
        return self.agents.get(key)

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """向流程添加新代理

        Args:
            key: 代理标识键
            agent: 要添加的代理实例
        """
        self.agents[key] = agent

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """执行流程

        Args:
            input_text: 输入文本

        Returns:
            执行结果文本
        """
