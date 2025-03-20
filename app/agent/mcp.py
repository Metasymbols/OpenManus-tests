from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.mcp_zh import (
    MULTIMEDIA_RESPONSE_PROMPT,
    NEXT_STEP_PROMPT,
    SYSTEM_PROMPT,
)
from app.schema import AgentState, Message
from app.tool.base import ToolResult
from app.tool.mcp import MCPClients


class MCPAgent(ToolCallAgent):
    """MCP服务器交互代理（基于ToolCallAgent）

    核心功能：
    - 支持SSE和stdio两种连接方式
    - 动态管理服务器工具集合
    - 自动刷新工具列表和状态
    - 提供完整的生命周期管理
    """

    name: str = "mcp_agent"
    description: str = "An agent that connects to an MCP server and uses its tools."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 初始化MCP工具集合
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    available_tools: MCPClients = None  # 将在initialize（）中设置

    max_steps: int = 20
    connection_type: str = "stdio"  # “ STDIO”或“ SSE”

    # 跟踪工具模式以检测更改
    tool_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _refresh_tools_interval: int = 5  # 刷新工具每个n步骤

    # 应触发终止的特殊工具名称
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])

    async def initialize(
        self,
        connection_type: Optional[str] = None,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> None:
        """初始化MCP连接

        参数说明：
        - connection_type: 连接类型（"stdio"或"sse"）
        - server_url: MCP服务器URL（SSE连接必需）
        - command: 要执行的命令（stdio连接必需）
        - args: 命令参数列表（stdio连接可选）

        执行流程：
        1. 根据连接类型建立服务器连接
        2. 初始化可用工具集合
        3. 添加系统提示和工具信息
        """
        if connection_type:
            self.connection_type = connection_type

        # 基于连接类型连接到MCP服务器
        if self.connection_type == "sse":
            if not server_url:
                raise ValueError("Server URL is required for SSE connection")
            await self.mcp_clients.connect_sse(server_url=server_url)
        elif self.connection_type == "stdio":
            if not command:
                raise ValueError("Command is required for stdio connection")
            await self.mcp_clients.connect_stdio(command=command, args=args or [])
        else:
            raise ValueError(f"Unsupported connection type: {self.connection_type}")

        # 将可用的_tool设置为我们的MCP实例
        self.available_tools = self.mcp_clients

        # 存储初始工具模式
        await self._refresh_tools()

        # 添加有关可用工具的系统消息
        tool_names = list(self.mcp_clients.tool_map.keys())
        tools_info = ", ".join(tool_names)

        # 添加系统提示和可用工具信息
        self.memory.add_message(
            Message.system_message(
                f"{self.system_prompt}\n\nAvailable MCP tools: {tools_info}"
            )
        )

    async def _refresh_tools(self) -> Tuple[List[str], List[str]]:
        """刷新MCP服务器工具列表

        功能：
        - 获取服务器最新工具列表
        - 比较工具变更（新增/移除）
        - 更新本地工具集合状态

        返回：
        tuple: 包含(新增工具列表, 移除工具列表)
        """
        if not self.mcp_clients.session:
            return [], []

        # 直接从服务器获取当前工具架构
        response = await self.mcp_clients.session.list_tools()
        current_tools = {tool.name: tool.inputSchema for tool in response.tools}

        # 确定添加，删除和更改工具
        current_names = set(current_tools.keys())
        previous_names = set(self.tool_schemas.keys())

        added_tools = list(current_names - previous_names)
        removed_tools = list(previous_names - current_names)

        # 检查现有工具中的模式更改
        changed_tools = []
        for name in current_names.intersection(previous_names):
            if current_tools[name] != self.tool_schemas.get(name):
                changed_tools.append(name)

        # 更新存储的模式
        self.tool_schemas = current_tools

        # 日志并通知更改
        if added_tools:
            logger.info(f"新增MCP工具: {added_tools}")
            self.memory.add_message(
                Message.system_message(f"New tools available: {', '.join(added_tools)}")
            )
        if removed_tools:
            logger.info(f"移除MCP工具: {removed_tools}")
            self.memory.add_message(
                Message.system_message(
                    f"Tools no longer available: {', '.join(removed_tools)}"
                )
            )
        if changed_tools:
            logger.info(f"变更MCP工具: {changed_tools}")

        return added_tools, removed_tools

    async def think(self) -> bool:
        """Process current state and decide next action."""
        # 检查MCP会话和工具可用性
        if not self.mcp_clients.session or not self.mcp_clients.tool_map:
            logger.info("MCP服务不可用，结束交互")
            self.state = AgentState.FINISHED
            return False

        # 定期刷新工具
        if self.current_step % self._refresh_tools_interval == 0:
            await self._refresh_tools()
            # 删除的所有工具表示关闭
            if not self.mcp_clients.tool_map:
                logger.info("MCP服务已关闭，结束交互")
                self.state = AgentState.FINISHED
                return False

        # 使用父班的思考方法
        return await super().think()

    async def _handle_special_tool(self, name: str, result: Any, **kwargs) -> None:
        """Handle special tool execution and state changes"""
        # 父母处理程序的第一个过程
        await super()._handle_special_tool(name, result, **kwargs)

        # 处理多媒体响应
        if isinstance(result, ToolResult) and result.base64_image:
            self.memory.add_message(
                Message.system_message(
                    MULTIMEDIA_RESPONSE_PROMPT.format(tool_name=name)
                )
            )

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        # 如果工具名称为“终止”，终止
        return name.lower() == "terminate"

    async def cleanup(self) -> None:
        """Clean up MCP connection when done."""
        if self.mcp_clients.session:
            await self.mcp_clients.disconnect()
            logger.info("MCP连接已关闭")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            result = await super().run(request)
            return result
        finally:
            # 确保清理发生，即使有错误
            await self.cleanup()
