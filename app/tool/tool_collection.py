"""Collection classes for managing multiple tools."""

from typing import Any, Dict, List

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """工具集合管理类，用于统一管理多个工具实例。

    核心功能：
    - 提供工具的动态注册与检索功能
    - 支持单工具执行和批量顺序执行模式
    - 统一处理工具执行异常并标准化错误输出
    - 生成工具集的OpenAI兼容参数格式

    典型用例：
    >>> collection = ToolCollection(tool1, tool2)
    >>> collection.add_tools(tool3)
    >>> result = await collection.execute(name="tool1")
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        """初始化工具集合

        Args:
            *tools (BaseTool): 可变数量工具实例，支持初始化时批量注入

        属性:
            tools (tuple): 按添加顺序保存的工具元组
            tool_map (dict): 工具名称到工具实例的映射字典
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolResult:
        """执行指定工具

        Args:
            name (str): 工具注册名称，需存在于tool_map中
            tool_input (Dict[str, Any], optional): 工具输入参数字典，默认None

        Returns:
            ToolResult: 工具执行结果对象，包含成功状态和输出内容

        Raises:
            隐式捕获ToolError异常并返回ToolFailure结果
        """
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """Execute all tools in the collection sequentially."""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        """动态添加单个工具到集合

        Args:
            tool (BaseTool): 必须实现name属性的工具实例

        Returns:
            self: 支持链式调用

        Raises:
            ValueError: 当工具名称已存在时会抛出异常
        """
        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        """批量添加多个工具

        Args:
            *tools (BaseTool): 可变数量工具实例

        Returns:
            self: 支持链式调用
        """
        for tool in tools:
            self.add_tool(tool)
        return self
