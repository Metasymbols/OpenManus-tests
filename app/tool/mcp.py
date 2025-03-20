from contextlib import AsyncExitStack
from typing import List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


class MCPClientTool(BaseTool):
    """表示可以在客户端调用MCP服务器端工具的代理类。

    该类继承自BaseTool，作为客户端和服务器之间的桥梁，
    负责将客户端的工具调用请求转发到服务器端执行。

    属性:
        session: 客户端会话对象，用于与服务器通信
    """

    session: Optional[ClientSession] = None

    async def execute(self, **kwargs) -> ToolResult:
        """执行工具调用。

        通过客户端会话向服务器发送工具调用请求，并处理返回结果。

        Args:
            **kwargs: 工具调用的参数字典

        Returns:
            ToolResult: 包含执行结果或错误信息的结果对象
        """
        if not self.session:
            return ToolResult(error="未连接到MCP服务器")

        try:
            result = await self.session.call_tool(self.name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")


class MCPClients(ToolCollection):
    """
    MCP客户端工具集合，负责连接MCP服务器并通过模型上下文协议管理可用工具。

    该类继承自ToolCollection，提供了与MCP服务器建立连接、
    管理工具列表以及处理会话生命周期的功能。

    属性:
        session: 客户端会话对象，用于与服务器通信
        exit_stack: 异步上下文管理器栈，用于管理资源的生命周期
        description: 工具集合的描述信息
    """

    session: Optional[ClientSession] = None
    exit_stack: AsyncExitStack = None
    description: str = "MCP客户端工具集，用于服务器交互"

    def __init__(self):
        super().__init__()  # Initialize with empty tools list
        self.name = "mcp"  # Keep name for backward compatibility
        self.exit_stack = AsyncExitStack()

    async def connect_sse(self, server_url: str) -> None:
        """使用SSE传输方式连接到MCP服务器。

        Args:
            server_url: 服务器的URL地址

        Raises:
            ValueError: 当服务器URL为空时抛出
        """
        if not server_url:
            raise ValueError("Server URL is required.")
        if self.session:
            await self.disconnect()

        streams_context = sse_client(url=server_url)
        streams = await self.exit_stack.enter_async_context(streams_context)
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*streams)
        )

        await self._initialize_and_list_tools()

    async def connect_stdio(self, command: str, args: List[str]) -> None:
        """使用标准输入输出方式连接到MCP服务器。

        Args:
            command: 服务器命令
            args: 命令参数列表

        Raises:
            ValueError: 当服务器命令为空时抛出
        """
        if not command:
            raise ValueError("Server command is required.")
        if self.session:
            await self.disconnect()

        server_params = StdioServerParameters(command=command, args=args)
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self._initialize_and_list_tools()

    async def _initialize_and_list_tools(self) -> None:
        """初始化会话并填充工具映射。

        该方法会初始化客户端会话，获取服务器端可用的工具列表，
        并为每个工具创建对应的MCPClientTool实例。

        Raises:
            RuntimeError: 当会话未初始化时抛出
        """
        if not self.session:
            raise RuntimeError("Session not initialized.")

        await self.session.initialize()
        response = await self.session.list_tools()

        # Clear existing tools
        self.tools = tuple()
        self.tool_map = {}

        # Create proper tool objects for each server tool
        for tool in response.tools:
            server_tool = MCPClientTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=self.session,
            )
            self.tool_map[tool.name] = server_tool

        self.tools = tuple(self.tool_map.values())
        logger.info(
            f"Connected to server with tools: {[tool.name for tool in response.tools]}"
        )

    async def disconnect(self) -> None:
        """断开与MCP服务器的连接并清理资源。

        关闭客户端会话，清理工具列表和映射，释放相关资源。
        """
        if self.session and self.exit_stack:
            await self.exit_stack.aclose()
            self.session = None
            self.tools = tuple()
            self.tool_map = {}
            logger.info("Disconnected from MCP server")
