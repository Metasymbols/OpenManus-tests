from typing import Any

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.manus_zh import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.python_execute import PythonExecute
from app.tool.web_search import WebSearch


class Manus(ToolCallAgent):
    """全能型代理（继承自ToolCallAgent）

    核心功能：
    - 集成Python执行、网页浏览、文件操作等多样化工具
    - 支持复杂任务的多步骤规划与执行
    - 自动管理浏览器工具等资源的生命周期
    """

    # 代理标识
    name: str = "Manus"  # 代理名称（固定值）
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"  # 功能描述（保持英文）
    )

    # 提示词模板
    system_prompt: str = SYSTEM_PROMPT  # 通用系统提示模板
    next_step_prompt: str = NEXT_STEP_PROMPT  # 通用步骤决策提示

    # 执行控制参数
    max_observe: int = 2000  # 结果截断长度
    max_steps: int = 20  # 覆盖父类默认步数限制

    # 工具集合配置
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),  # Python执行工具
            WebSearch(),  # 网页搜索工具
            BrowserUseTool(),  # 浏览器操作工具
            FileSaver(),  # 文件保存工具
            Terminate(),  # 终止工具
        )
    )

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """特殊工具后处理
        增强功能：
        - 执行浏览器工具的清理操作
        - 调用父类基础处理逻辑
        """
        if not self._is_special_tool(name):
            return
        else:
            # 清理浏览器工具资源
            await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
            # 调用父类处理逻辑
            await super()._handle_special_tool(name, result, **kwargs)
