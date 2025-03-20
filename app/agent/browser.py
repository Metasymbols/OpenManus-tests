import json
from typing import Any, Optional

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.browser_zh import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message, ToolChoice
from app.tool import BrowserUseTool, Terminate, ToolCollection


class BrowserAgent(ToolCallAgent):
    """
    浏览器代理（继承自ToolCallAgent）

    核心功能：
    - 使用browser_use库控制浏览器行为
    - 支持网页导航、元素交互、表单填写等操作
    - 能够提取网页内容并执行复杂的浏览任务
    - 自动管理浏览器状态和截图功能
    """

    name: str = "browser"
    description: str = (
        "A browser agent that can control a browser to accomplish tasks"  # Keep English in compatible tool calls
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # 配置可用工具
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(BrowserUseTool(), Terminate())
    )

    # 使用自动选择工具以允许工具使用和自由形式响应
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    _current_base64_image: Optional[str] = None

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return
        else:
            await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
            await super()._handle_special_tool(name, result, **kwargs)

    async def get_browser_state(self) -> Optional[dict]:
        """获取当前浏览器状态用于上下文

        返回：
            dict或None：包含URL、标题、标签页等信息的字典，获取失败时返回None
        """
        browser_tool = self.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool:
            return None

        try:
            # 直接从工具中获取浏览器状态
            result = await browser_tool.get_current_state()

            if result.error:
                logger.debug(f"浏览器状态获取错误: {result.error}")
                return None

            # 存储屏幕截图如果可用
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image

            # 解析状态信息
            return json.loads(result.output)

        except Exception as e:
            logger.debug(f"获取浏览器状态失败: {str(e)}")
            return None

    async def think(self) -> bool:
        """处理当前状态并决定下一步操作

        增强功能：
        - 自动注入浏览器状态到上下文
        - 支持标签页信息展示
        - 记录页面滚动位置
        - 自动处理截图附件

        返回：
            bool：思考阶段的执行结果
        """
        # 将浏览器状态添加到上下文
        browser_state = await self.get_browser_state()

        # 初始化占位符值
        url_info = ""
        tabs_info = ""
        content_above_info = ""
        content_below_info = ""
        results_info = ""

        if browser_state and not browser_state.get("error"):
            # URL和标题信息
            url_info = f"\n   URL: {browser_state.get('url', 'N/A')}\n   Title: {browser_state.get('title', 'N/A')}"

            # 选项卡信息
            if "tabs" in browser_state:
                tabs = browser_state.get("tabs", [])
                if tabs:
                    tabs_info = f"\n   {len(tabs)} tab(s) available"

            # 内容上方/下方的内容
            pixels_above = browser_state.get("pixels_above", 0)
            pixels_below = browser_state.get("pixels_below", 0)

            if pixels_above > 0:
                content_above_info = f" ({pixels_above} pixels)"

            if pixels_below > 0:
                content_below_info = f" ({pixels_below} pixels)"

            # 如果可用
            if self._current_base64_image:
                # 使用图像附件创建消息
                image_message = Message.user_message(
                    content="Current browser screenshot:",
                    base64_image=self._current_base64_image,
                )
                self.memory.add_message(image_message)

        # 用实际的浏览器状态信息代替占位符
        self.next_step_prompt = NEXT_STEP_PROMPT.format(
            url_placeholder=url_info,
            tabs_placeholder=tabs_info,
            content_above_placeholder=content_above_info,
            content_below_placeholder=content_below_info,
            results_placeholder=results_info,
        )

        # 调用父级实现
        result = await super().think()

        # 将next_step_prompt重置为原始状态
        self.next_step_prompt = NEXT_STEP_PROMPT

        return result
