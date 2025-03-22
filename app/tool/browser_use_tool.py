import asyncio
import json
from typing import Generic, Optional

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


MAX_LENGTH = 2000

_BROWSER_DESCRIPTION = """
网页浏览器交互工具，支持导航、元素操作、内容提取和标签管理。主要功能包括：
- 'navigate'：导航到指定URL
- 'click'：通过索引点击元素
- 'input_text'：在元素中输入文本
- 'screenshot'：截取屏幕截图
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
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
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

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """
        初始化浏览器实例和上下文环境

        返回:
            BrowserContext: 初始化完成的浏览器上下文对象

        实现流程:
        1. 检查浏览器实例是否存在，不存在时根据配置创建新实例
        2. 初始化浏览器上下文配置（包含代理设置、安全参数等）
        3. 创建DOM服务实例用于页面元素操作
        """
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
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

            # if there is context config in the config, use it.
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
        执行浏览器操作指令

        参数说明:
        - action: 要执行的操作类型（12种预定义操作）
        - url: 导航或新建标签页的目标地址
        - index: 元素索引（点击/输入操作用）
        - text: 需要输入的文本内容
        - script: 要执行的JavaScript代码
        - scroll_amount: 滚动像素数（正数向下，负数向上）
        - tab_id: 要切换的标签页ID

        返回值:
            ToolResult: 包含操作结果或错误信息的工具结果对象

        异常处理:
        - 捕获所有操作异常并返回错误信息
        - 通过互斥锁保证线程安全
        """

        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                if action == "navigate":
                    from urllib.parse import urlparse

                    if not url:
                        return ToolResult(error="URL is required for 'navigate' action")

                    # 自动补全协议前缀
                    if not url.startswith(("http://", "https://")):
                        url = f"http://{url}"

                    # 验证URL格式
                    try:
                        result = urlparse(url)
                        if not all([result.scheme, result.netloc]):
                            raise ValueError
                    except:
                        return ToolResult(error=f"Invalid URL format: {url}")

                    await context.navigate_to(url)
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
                        # Navigate to the first search result
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
                        await page.goto(url_to_navigate)
                        await page.wait_for_load_state()

                        return ToolResult(
                            output=f"Searched for '{query}' and navigated to first result: {url_to_navigate}\nAll results:"
                            + "\n".join([str(r) for r in search_results])
                        )
                    else:
                        return ToolResult(
                            error=f"No search results found for '{query}'"
                        )

                # Element interaction actions
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

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )
                    page = await context.get_current_page()
                    try:
                        # Get page content and convert to markdown for better processing
                        html_content = await page.content()

                        # Import markdownify here to avoid global import
                        try:
                            import markdownify

                            content = markdownify.markdownify(html_content)
                        except ImportError:
                            # Fallback if markdownify is not available
                            content = html_content

                        # Create prompt for LLM
                        prompt_text = """
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}

Page content:
{page}
"""
                        # Format the prompt with the goal and content
                        max_content_length = min(50000, len(content))
                        formatted_prompt = prompt_text.format(
                            goal=goal, page=content[:max_content_length]
                        )

                        # Create a proper message list for the LLM
                        from app.schema import Message

                        messages = [Message.user_message(formatted_prompt)]

                        # Define extraction function for the tool
                        extraction_function = {
                            "type": "function",
                            "function": {
                                "name": "extract_content",
                                "description": "Extract specific information from a webpage based on a goal",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "extracted_content": {
                                            "type": "object",
                                            "description": "The content extracted from the page according to the goal",
                                            "properties": {
                                                "text": {
                                                    "type": "string",
                                                    "description": "Text content extracted from the page",
                                                },
                                                "metadata": {
                                                    "type": "object",
                                                    "description": "Additional metadata about the extracted content",
                                                    "properties": {
                                                        "source": {
                                                            "type": "string",
                                                            "description": "Source of the extracted content",
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

                        # Use LLM to extract content with required function calling
                        response = await self.llm.ask_tool(
                            messages,
                            tools=[extraction_function],
                            tool_choice="required",
                        )

                        # Extract content from function call response
                        if (
                            response
                            and response.tool_calls
                            and len(response.tool_calls) > 0
                        ):
                            # Get the first tool call arguments
                            tool_call = response.tool_calls[0]
                            # Parse the JSON arguments
                            try:
                                args = json.loads(tool_call.function.arguments)
                                extracted_content = args.get("extracted_content", {})
                                # Format extracted content as JSON string
                                content_json = json.dumps(
                                    extracted_content, indent=2, ensure_ascii=False
                                )
                                msg = f"Extracted from page:\n{content_json}\n"
                            except Exception as e:
                                msg = f"Error parsing extraction result: {str(e)}\nRaw response: {tool_call.function.arguments}"
                        else:
                            msg = "No content was extracted from the page."

                        return ToolResult(output=msg)
                    except Exception as e:
                        # Provide a more helpful error message
                        error_msg = f"Failed to extract content: {str(e)}"
                        try:
                            # Try to return a portion of the page content as fallback
                            return ToolResult(
                                output=f"{error_msg}\nHere's a portion of the page content:\n{content[:2000]}..."
                            )
                        except:
                            # If all else fails, just return the error
                            return ToolResult(error=error_msg)

                # Tab management actions
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
                        return ToolResult(error="URL is required for 'new_tab' action")

                    # 复用导航页的验证逻辑
                    validation_result = await self.execute("navigate", url=url)
                    if validation_result.error:
                        return validation_result

                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(self) -> ToolResult:
        """
        获取当前浏览器状态

        返回:
            ToolResult: 包含浏览器状态信息的JSON字符串
                包含字段:
                - url: 当前页面地址
                - title: 页面标题
                - tabs: 所有标签页信息列表
                - interactive_elements: 可交互元素列表

        异常处理:
        - 捕获状态获取失败异常并返回错误信息
        """
        """Get the current browser state as a ToolResult."""
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()
                state = await context.get_state()
                state_info = {
                    "url": state.url,
                    "title": state.title,
                    "tabs": [tab.model_dump() for tab in state.tabs],
                    "interactive_elements": state.element_tree.clickable_elements_to_string(),
                }
                return ToolResult(output=json.dumps(state_info))
            except Exception as e:
                return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """
        清理浏览器资源

        执行流程:
        1. 关闭当前浏览器上下文
        2. 终止浏览器实例
        3. 重置实例引用为None
        """
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
