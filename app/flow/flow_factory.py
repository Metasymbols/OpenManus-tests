from enum import Enum
from typing import Dict, List, Union

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.flow.planning import PlanningFlow


class FlowType(str, Enum):
    """流程类型枚举

    定义了系统支持的不同类型的执行流程。

    Attributes:
        PLANNING: 规划型流程，用于任务分解和执行规划
    """

    PLANNING = "planning"


class FlowFactory:
    """流程工厂类

    用于创建不同类型的执行流程，支持多代理配置。
    该工厂类负责根据指定的流程类型实例化相应的流程对象。
    """

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        """创建指定类型的执行流程实例

        Args:
            flow_type: 要创建的流程类型
            agents: 流程使用的代理配置，支持以下格式：
                - 单个代理实例
                - 代理实例列表
                - 代理实例字典(键为代理标识)
            **kwargs: 额外的流程配置参数

        Returns:
            创建的流程实例

        Raises:
            ValueError: 当指定了未知的流程类型时抛出
        """
        flows = {
            FlowType.PLANNING: PlanningFlow,
        }

        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"Unknown flow type: {flow_type}")

        return flow_class(agents, **kwargs)
