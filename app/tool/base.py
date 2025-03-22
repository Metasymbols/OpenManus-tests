from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    """基础工具类抽象基类

    Attributes:
        name: 工具名称（需全局唯一）
        description: 工具功能描述（用于LLM识别）
        parameters: 工具参数规范（遵循JSON Schema格式）
    """

    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """工具执行入口

        Args:
            **kwargs: 动态接收工具参数

        Returns:
            Any: 工具执行原始结果

        Raises:
            ToolExecutionError: 工具执行过程中出现错误时抛出
        """
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """抽象执行方法（需子类实现具体逻辑）

        Args:
            **kwargs: 工具参数键值对

        Note:
            子类应在此方法中实现具体的工具业务逻辑，并处理参数验证
        """

    def to_param(self) -> Dict:
        """将工具转换为函数调用格式

        Returns:
            Dict: 符合OpenAI Function Calling规范的函数描述

        Example:
            {"type": "function", "function": {"name": ...}}
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """工具执行结果模型

    Attributes:
        output: 原始执行结果（任意类型）
        error: 错误信息（执行失败时设置）
        system: 系统级附加信息（调试/日志用途）
    """

    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        return f"Error: {self.error}" if self.error else self.output

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        # return self.copy(update=kwargs)
        return type(self)(**{**self.dict(), **kwargs})


class CLIResult(ToolResult):
    """命令行渲染专用结果类型

    特征：
    - 自动格式化输出内容
    - 支持ANSI颜色代码
    - 优化控制台显示效果
    """


class ToolFailure(ToolResult):
    """工具执行失败标识类型

    特征：
    - error字段必须包含错误描述
    - 自动记录错误堆栈到system字段
    - 触发告警监控系统
    """


class AgentAwareTool:
    """支持代理绑定的工具基类

    属性:
        agent: 绑定的代理实例（运行时自动注入）

    功能:
    - 提供工具访问代理上下文的能力
    - 支持在工具中调用其他工具
    - 允许访问共享内存空间
    """

    agent: Optional = None
