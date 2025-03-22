import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch


_BROWSER_DESCRIPTION = """
浏览器交互工具，支持导航、元素操作、内容提取和标签页管理。主要功能包括：
- 'navigate'：导航到指定网址
- 'click'：通过索引点击元素
- 'input_text'：在元素中输入文本
- 'screenshot'：截取网页截图
- 'get_html'：获取页面HTML内容
- 'get_text'：获取页面文本内容
- 'read_links'：读取页面所有链接
- 'execute_js'：执行JavaScript代码
- 'scroll'：滚动页面
- 'switch_tab'：切换浏览器标签页
- 'new_tab'：新建标签页
- 'close_tab'：关闭当前标签页
- 'refresh'：刷新当前页面
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "要执行的浏览器操作类型",
            },
            "url": {
                "type": "string",
                "description": "用于'navigate'或'new_tab'操作的目标网址",
            },
            "index": {
                "type": "integer",
                "description": "用于'click'或'input_text'操作的元素索引",
            },
            "text": {"type": "string", "description": "用于'input_text'操作的输入文本"},
            "script": {
                "type": "string",
                "description": "用于'execute_js'操作的JavaScript代码",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "用于'scroll'操作的滚动像素数（正数向下滚动，负数向上滚动）",
            },
            "tab_id": {
                "type": "integer",
                "description": "用于'switch_tab'操作的标签页ID",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # 通用功能的上下文
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        try:
            if not isinstance(v, dict):
                raise ValueError("参数必须是一个有效的JSON对象")
            if not v:
                raise ValueError("参数不能为空")
            return v
        except Exception as e:
            raise ValueError(f"参数验证错误: {str(e)}")

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # 处理代理设置。
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # 如果配置中有上下文配置，请使用它。
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                # 从配置获取最大内容长度
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # 导航动作
                if action == "go_to_url":
                    if not url:
                        return ToolResult(error="导航操作需要提供URL")

                    # 自动补全协议前缀
                    if not url.startswith(("http://", "https://")):
                        url = f"http://{url}"

                    # 验证URL格式
                    try:
                        result = urlparse(url)
                        if not all([result.scheme, result.netloc]):
                            raise ValueError
                    except:
                        return ToolResult(error=f"无效的URL格式: {url}")

                    await context.navigate_to(url)
                    return ToolResult(output=f"已导航至 {url}")

                elif action == "go_back":
                    await context.go_back()
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    search_results = await self.web_search_tool.execute(query)

                    if search_results:
                        # 导航到第一个搜索结果
                        first_result = search_results[0]
                        if isinstance(first_result, dict) and "url" in first_result:
                            url_to_navigate = first_result["url"]
                        elif isinstance(first_result, str):
                            url_to_navigate = first_result
                        else:
                            return ToolResult(
                                error=f"Invalid search result format: {first_result}"
                            )

                        page = await context.get_current_page()
                        response = await page.goto(url_to_navigate)

                        if response.status >= 400:
                            raise Exception(
                                f"HTTP {response.status} error loading {url_to_navigate}"
                            )

                        # 处理百度搜索的特殊重定向
                        if "baidu.com/link?" in url_to_navigate:
                            await page.wait_for_timeout(1500)  # 等待重定向完成
                            final_url = page.url
                            if final_url == "https://www.baidu.com/":
                                raise Exception("Baidu search result link expired")

                        await page.wait_for_load_state("networkidle", timeout=10000)

                        return ToolResult(
                            output=f"Searched for '{query}' and navigated to first result: {url_to_navigate}\nAll results:"
                            + "\n".join([str(r) for r in search_results])
                        )
                    else:
                        return ToolResult(
                            error=f"No search results found for '{query}'"
                        )

                # 元素交互操作
                elif action == "click_element":
                    if index is None:
                        return ToolResult(error="点击操作需要提供元素索引")
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"未找到索引为 {index} 的元素")
                    download_path = await context._click_element_node(element)
                    output = f"已点击索引为 {index} 的元素"
                    if download_path:
                        output += f" - 文件已下载至 {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="输入文本操作需要提供元素索引和文本内容"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"未找到索引为 {index} 的元素")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"已在索引为 {index} 的元素中输入文本 '{text}'"
                    )

                elif action == "screenshot":
                    screenshot = await context.take_screenshot(full_page=True)
                    return ToolResult(
                        output=f"已捕获截图（base64长度: {len(screenshot)}）",
                        system=screenshot,
                    )

                elif action == "get_html":
                    html = await context.get_page_html()
                    truncated = (
                        html[:MAX_LENGTH] + "..." if len(html) > MAX_LENGTH else html
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "execute_js":
                    if not script:
                        return ToolResult(error="执行JavaScript操作需要提供脚本代码")
                    result = await context.execute_javascript(script)
                    return ToolResult(output=str(result))

                elif action == "scroll":
                    if scroll_amount is None:
                        return ToolResult(error="滚动操作需要提供滚动像素数")
                    await context.execute_javascript(
                        f"window.scrollBy(0, {scroll_amount});"
                    )
                    direction = "向下" if scroll_amount > 0 else "向上"
                    return ToolResult(
                        output=f"已{direction}滚动 {abs(scroll_amount)} 像素"
                    )

                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(error="切换标签页操作需要提供标签页ID")
                    await context.switch_to_tab(tab_id)
                    return ToolResult(output=f"已切换至标签页 {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="新建标签页操作需要提供URL")

                    # 复用导航页的验证逻辑
                    validation_result = await self.execute("navigate", url=url)
                    if validation_result.error:
                        return validation_result

                    await context.create_new_tab(url)
                    return ToolResult(output=f"已在新标签页打开 {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="已关闭当前标签页")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="已刷新当前页面")

                else:
                    return ToolResult(error=f"未知的操作类型: {action}")

            except Exception as e:
                return ToolResult(error=f"浏览器操作 '{action}' 失败: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # 使用提供的上下文或回到self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # 如果不存在，请创建fiewport_info词典
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # 为州屏幕截图
            page = await ctx.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state()

            screenshot = await page.screenshot(
                full_page=True, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # 使用所有必需字段构建状态信息
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc.,表示与列出的元素相对应的可点击索引。单击这些索引将导航到或与它们背后的各个内容进行交互。",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
