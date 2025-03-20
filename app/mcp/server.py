import logging
import sys


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])

import argparse
import asyncio
import atexit
import json
from inspect import Parameter, Signature
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from app.logger import logger
from app.tool.base import BaseTool
from app.tool.bash import Bash
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate


class MCPServer:
    """MCP服务器实现，负责工具注册和管理。

    该类提供了一个基于FastMCP的服务器实现，支持工具的动态注册和管理。
    服务器可以处理工具的参数验证、文档生成和执行调用等功能。

    属性:
        server: FastMCP实例，用于处理底层的MCP协议通信
        tools: 存储已注册工具的字典，键为工具名称，值为工具实例
    """

    def __init__(self, name: str = "openmanus"):
        """初始化MCP服务器。

        Args:
            name: 服务器名称，默认为'openmanus'
        """
        self.server = FastMCP(name)
        self.tools: Dict[str, BaseTool] = {}

        # 初始化标准工具
        self.tools["bash"] = Bash()
        self.tools["browser"] = BrowserUseTool()
        self.tools["editor"] = StrReplaceEditor()
        self.tools["terminate"] = Terminate()

    def register_tool(self, tool: BaseTool, method_name: Optional[str] = None) -> None:
        """注册工具，包含参数验证和文档生成。

        该方法将工具注册到服务器，并生成相应的参数验证和文档。
        注册过程包括：
        1. 创建工具的异步执行方法
        2. 生成工具的文档字符串
        3. 构建工具的参数签名
        4. 存储工具的参数模式

        Args:
            tool: 要注册的工具实例，必须是BaseTool的子类
            method_name: 可选的方法名称，如果不提供则使用工具的name属性
        """
        tool_name = method_name or tool.name
        tool_param = tool.to_param()
        tool_function = tool_param["function"]

        # Define the async function to be registered
        async def tool_method(**kwargs):
            logger.info(f"Executing {tool_name}: {kwargs}")
            result = await tool.execute(**kwargs)

            logger.info(f"Result of {tool_name}: {result}")

            # Handle different types of results (match original logic)
            if hasattr(result, "model_dump"):
                return json.dumps(result.model_dump())
            elif isinstance(result, dict):
                return json.dumps(result)
            return result

        # Set method metadata
        tool_method.__name__ = tool_name
        tool_method.__doc__ = self._build_docstring(tool_function)
        tool_method.__signature__ = self._build_signature(tool_function)

        # Store parameter schema (important for tools that access it programmatically)
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        tool_method._parameter_schema = {
            param_name: {
                "description": param_details.get("description", ""),
                "type": param_details.get("type", "any"),
                "required": param_name in required_params,
            }
            for param_name, param_details in param_props.items()
        }

        # Register with server
        self.server.tool()(tool_method)
        logger.info(f"Registered tool: {tool_name}")

    def _build_docstring(self, tool_function: dict) -> str:
        """从工具函数元数据构建格式化的文档字符串。

        Args:
            tool_function: 包含工具函数元数据的字典，包括描述和参数信息

        Returns:
            格式化的文档字符串，包含工具描述和参数说明
        """
        description = tool_function.get("description", "")
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])

        # Build docstring (match original format)
        docstring = description
        if param_props:
            docstring += "\n\nParameters:\n"
            for param_name, param_details in param_props.items():
                required_str = (
                    "(required)" if param_name in required_params else "(optional)"
                )
                param_type = param_details.get("type", "any")
                param_desc = param_details.get("description", "")
                docstring += (
                    f"    {param_name} ({param_type}) {required_str}: {param_desc}\n"
                )

        return docstring

    def _build_signature(self, tool_function: dict) -> Signature:
        """从工具函数元数据构建函数签名。

        将JSON Schema类型映射到Python类型，并创建相应的参数签名。

        Args:
            tool_function: 包含工具函数元数据的字典，包括参数类型和必需性

        Returns:
            函数的参数签名对象
        """
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])

        parameters = []

        # Follow original type mapping
        for param_name, param_details in param_props.items():
            param_type = param_details.get("type", "")
            default = Parameter.empty if param_name in required_params else None

            # Map JSON Schema types to Python types (same as original)
            annotation = Any
            if param_type == "string":
                annotation = str
            elif param_type == "integer":
                annotation = int
            elif param_type == "number":
                annotation = float
            elif param_type == "boolean":
                annotation = bool
            elif param_type == "object":
                annotation = dict
            elif param_type == "array":
                annotation = list

            # Create parameter with same structure as original
            param = Parameter(
                name=param_name,
                kind=Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
            parameters.append(param)

        return Signature(parameters=parameters)

    async def cleanup(self) -> None:
        """清理服务器资源。

        主要用于清理浏览器工具等需要特殊处理的资源。
        该方法在服务器关闭时自动调用。
        """
        logger.info("Cleaning up resources")
        # Follow original cleanup logic - only clean browser tool
        if "browser" in self.tools and hasattr(self.tools["browser"], "cleanup"):
            await self.tools["browser"].cleanup()

    def register_all_tools(self) -> None:
        """注册所有已添加的工具到服务器。

        遍历tools字典中的所有工具实例，并调用register_tool方法进行注册。
        """
        for tool in self.tools.values():
            self.register_tool(tool)

    def run(self, transport: str = "stdio") -> None:
        """运行MCP服务器。

        Args:
            transport: 传输方式，默认为'stdio'，目前仅支持标准输入输出传输
        """
        # Register all tools
        self.register_all_tools()

        # Register cleanup function (match original behavior)
        atexit.register(lambda: asyncio.run(self.cleanup()))

        # Start server (with same logging as original)
        logger.info(f"Starting OpenManus server ({transport} mode)")
        self.server.run(transport=transport)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        包含解析后的命令行参数的Namespace对象
    """
    parser = argparse.ArgumentParser(description="OpenManus MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Communication method: stdio or http (default: stdio)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create and run server (maintaining original flow)
    server = MCPServer()
    server.run(transport=args.transport)
