import asyncio
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import config
from app.logger import logger
from app.tool.base import BaseTool
from app.tool.search import (
    BaiduSearchEngine,
    BingSearchEngine,
    DuckDuckGoSearchEngine,
    GoogleSearchEngine,
    WebSearchEngine,
)


class WebSearch(BaseTool):
    """多搜索引擎集成工具类

    特性：
    - 支持Google/Baidu/DuckDuckGo多个搜索引擎
    - 配置驱动引擎优先级（通过config.search_config.engine指定）
    - 自动故障转移机制：主引擎失败时自动尝试备用引擎
    - 指数退避重试策略：单个引擎最多重试3次

    使用示例：
    >>> await WebSearch().execute("最新AI新闻")
    """

    name: str = "web_search"
    description: str = """执行网页搜索并返回相关链接列表
    该工具优先使用配置的主搜索引擎获取结果，当主引擎失败时
    会自动按预定义顺序尝试备用搜索引擎。"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the search engine.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    _search_engine: dict[str, WebSearchEngine] = {
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "duckduckgo": DuckDuckGoSearchEngine(),
        "bing": BingSearchEngine(),
    }

    async def execute(self, query: str, num_results: int = 10) -> List[str]:
        """
        执行网页搜索并返回结果链接

        Args:
            query (str): 搜索关键词或查询语句
            num_results (int, optional): 需要返回的结果数量，默认10条

        Returns:
            List[str]: 匹配的URL列表，按引擎优先级返回可用结果

        Raises:
            SearchEngineError: 当所有配置的搜索引擎均不可用时抛出
        """
        engine_order = self._get_engine_order()
        failed_engines = []

        for engine_name in engine_order:
            engine = self._search_engine[engine_name]
            try:
                logger.info(f"🔎 Attempting search with {engine_name.capitalize()}...")
                links = await self._perform_search_with_engine(
                    engine, query, num_results
                )
                if links:
                    if failed_engines:
                        logger.info(
                            f"Search successful with {engine_name.capitalize()} after trying: {', '.join(failed_engines)}"
                        )
                    return links
            except Exception as e:
                failed_engines.append(engine_name.capitalize())
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)

                if is_rate_limit:
                    logger.warning(
                        f"⚠️ {engine_name.capitalize()} search engine rate limit exceeded, trying next engine..."
                    )
                else:
                    logger.warning(
                        f"⚠️ {engine_name.capitalize()} search failed with error: {e}"
                    )

        if failed_engines:
            logger.error(f"All search engines failed: {', '.join(failed_engines)}")
        return []

    def _get_engine_order(self) -> List[str]:
        """
        Determines the order in which to try search engines.
        Preferred engine is first (based on configuration), followed by fallback engines,
        and then the remaining engines.

        Returns:
            List[str]: Ordered list of search engine names.
        """
        preferred = "google"
        fallbacks = []

        if config.search_config:
            if config.search_config.engine:
                preferred = config.search_config.engine.lower()
            if config.search_config.fallback_engines:
                fallbacks = [
                    engine.lower() for engine in config.search_config.fallback_engines
                ]

        engine_order = []
        # Add preferred engine first
        if preferred in self._search_engine:
            engine_order.append(preferred)

        # Add configured fallback engines in order
        for fallback in fallbacks:
            if fallback in self._search_engine and fallback not in engine_order:
                engine_order.append(fallback)

        return engine_order

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _perform_search_with_engine(
        self,
        engine: WebSearchEngine,
        query: str,
        num_results: int,
    ) -> List[str]:
        """
        [重试机制] 执行单个搜索引擎查询

        重试策略：
        - 最多重试3次
        - 指数退避等待：1s, 2s, 4s（最大10秒）
        - 仅捕获引擎级别的临时性错误
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: list(engine.perform_search(query, num_results=num_results))
        )
