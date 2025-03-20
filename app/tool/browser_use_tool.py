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
通过浏览器执行导航、元素交互、内容提取和标签页管理等操作。本工具提供以下浏览器自动化能力：

导航：
- 'go_to_url': 在当前标签页访问指定URL
- 'go_back': 后退
- 'refresh': 刷新当前页面
- 'web_search': 在当前标签页进行网页搜索，查询词应具体明确，类似人类常用的搜索关键词，避免模糊或过长

元素交互：
- 'click_element': 通过索引点击元素
- 'input_text': 在表单元素中输入文本
- 'scroll_down'/'scroll_up': 滚动页面（可指定像素量）
- 'scroll_to_text': 如果找不到想交互的内容，滚动到指定文本位置
- 'send_keys': 发送特殊按键组合（如 Escape、Backspace、Insert、PageDown、Delete、Enter），支持`Control+o`、`Control+Shift+T`等快捷键，通过keyboard.press实现
- 'get_dropdown_options': 获取下拉框所有选项
- 'select_dropdown_option': 根据选项文本为指定元素选择下拉框选项

内容提取：
- 'extract_content': 提取页面内容获取特定信息，例如：所有公司名称、特定描述、结构化数据格式的公司链接等

标签页管理：
- 'switch_tab': 切换到指定标签页
- 'open_tab': 在新标签页打开URL
- 'close_tab': 关闭当前标签页

实用功能：
- 'wait': 等待指定秒数
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
                "description": "要执行的浏览器动作",
            },
            "url": {
                "type": "string",
                "description": "url for'go_to_url'或'Open_tab'操作",
            },
            "index": {
                "type": "integer",
                "description": "'click_element'，'input_text'，'get_dropdown_options'或'select_dropdown_option'actions'ancy'的元素索引",
            },
            "text": {
                "type": "string",
                "description": "'input_text'，'scroll_to_text'或'select_dropdown_option'操作的文本",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "'scroll_down'或'scroll_up'操作",
            },
            "tab_id": {
                "type": "integer",
                "description": "'switch_tab'动作的选项卡ID",
            },
            "query": {
                "type": "string",
                "description": "搜索查询'Web_search'动作",
            },
            "goal": {
                "type": "string",
                "description": "提取目标'extract_content'动作",
            },
            "keys": {
                "type": "string",
                "description": "发送'send_keys'动作的钥匙",
            },
            "seconds": {
                "type": "integer",
                "description": "待几秒钟'wait'行动",
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
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

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
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
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
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    await page.goto(url)
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Navigated to {url}")

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
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
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

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # 内容提取动作
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )
                    page = await context.get_current_page()
                    try:
                        # 获取页面内容并转换为Markdown以进行更好的处理
                        html_content = await page.content()

                        # 在这里导入降级以避免全局导入
                        try:
                            import markdownify

                            content = markdownify.markdownify(html_content)
                        except ImportError:
                            # 后备如果Markdownify不可用
                            content = html_content

                        # 为LLM创建提示
                        prompt_text = """
                            您的任务是提取页面的内容。您将获得一个页面和一个目标，您应该从页面上提取有关此目标的所有相关信息。如果目标含糊不清，请总结页面。响应JSON格式。
                            提取目标：{目标}
                           页面内容:
                            {page}
                            """
                        # 用目标和内容格式化提示
                        max_content_length = min(50000, len(content))
                        formatted_prompt = prompt_text.format(
                            goal=goal, page=content[:max_content_length]
                        )

                        # 为LLM创建适当的消息列表
                        from app.schema import Message

                        messages = [Message.user_message(formatted_prompt)]

                        # 定义工具的提取功能
                        extraction_function = {
                            "type": "function",
                            "function": {
                                "name": "extract_content",
                                "description": "根据目标从网页中提取特定信息",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "extracted_content": {
                                            "type": "object",
                                            "description": "根据目标从页面提取的内容",
                                            "properties": {
                                                "text": {
                                                    "type": "string",
                                                    "description": "从页面提取的文本内容",
                                                },
                                                "metadata": {
                                                    "type": "object",
                                                    "description": "关于提取的内容的其他元数据",
                                                    "properties": {
                                                        "source": {
                                                            "type": "string",
                                                            "description": "提取内容的来源",
                                                        }
                                                    },
                                                },
                                            },
                                        }
                                    },
                                    "required": ["extracted_content"],
                                },
                            },
                        }

                        # 使用LLM提取内容与所需功能调用的内容
                        response = await self.llm.ask_tool(
                            messages,
                            tools=[extraction_function],
                            tool_choice="required",
                        )

                        # 从功能呼叫响应中提取内容
                        if (
                            response
                            and response.tool_calls
                            and len(response.tool_calls) > 0
                        ):
                            # 获取第一个工具调用参数
                            tool_call = response.tool_calls[0]
                            # 解析JSON论点
                            try:
                                args = json.loads(tool_call.function.arguments)
                                extracted_content = args.get("extracted_content", {})
                                # 格式提取的内容作为JSON字符串
                                content_json = json.dumps(
                                    extracted_content, indent=2, ensure_ascii=False
                                )
                                msg = f"从页面提取:\n{content_json}\n"
                            except Exception as e:
                                msg = f"错误解析提取结果: {str(e)}\nRaw response: {tool_call.function.arguments}"
                        else:
                            msg = "没有从页面中提取内容。"

                        return ToolResult(output=msg)
                    except Exception as e:
                        # 提供更有帮助的错误消息
                        error_msg = f"无法提取内容: {str(e)}"
                        try:
                            # 尝试将页面内容的一部分返回作为后备
                            return ToolResult(
                                output=f"{error_msg}\n这是页面内容的一部分:\n{content[:2000]}..."
                            )
                        except Exception:
                            # 如果所有其他方法都失败，只需返回错误
                            return ToolResult(error=error_msg)

                # 标签管理操作
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # 公用事业动作
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

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
