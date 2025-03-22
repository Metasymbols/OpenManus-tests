import asyncio
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import config
from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool
from app.tool.search import (
    BaiduSearchEngine,
    BingSearchEngine,
    DuckDuckGoSearchEngine,
    GoogleSearchEngine,
    WebSearchEngine,
)


class SearchEngineError(ToolError):
    """搜索引擎执行错误

    当所有配置的搜索引擎均不可用时抛出
    """


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
                "description": "(必填) 要提交给搜索引擎的搜索查询。",
            },
            "num_results": {
                "type": "integer",
                "description": "(可选) 要返回的搜索结果数量，默认为10。",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    _search_engine: dict[str, WebSearchEngine] = {
        "duckduckgo": DuckDuckGoSearchEngine(),
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
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
        # 参数验证
        if not query or not isinstance(query, str) or query.strip() == "":
            raise ValueError("搜索查询不能为空")

        if not isinstance(num_results, int) or num_results <= 0:
            raise ValueError("结果数量必须为正整数")

        # 执行搜索
        engine_order = self._get_engine_order()
        errors = []

        for engine_name in engine_order:
            engine = self._search_engine[engine_name]
            try:
                logger.debug(f"尝试使用搜索引擎: {engine_name}")
                links = await self._perform_search_with_engine(
                    engine, query, num_results
                )
                if links:
                    logger.info(
                        f"搜索引擎 '{engine_name}' 成功返回 {len(links)} 条结果"
                    )
                    return links
                logger.warning(f"搜索引擎 '{engine_name}' 未返回任何结果")
            except Exception as e:
                error_msg = f"搜索引擎 '{engine_name}' 失败，错误信息: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            logger.error("所有搜索引擎均失败")
            raise SearchEngineError("所有配置的搜索引擎均不可用: " + "; ".join(errors))

        return []

    def _get_engine_order(self) -> List[str]:
        """
        确定尝试搜索引擎的顺序。
        首选引擎排在首位（基于配置），然后是其余引擎。

        Returns:
            List[str]: 搜索引擎名称的有序列表。
        """
        preferred = "bing"
        fallbacks = []

        if config.search_config:
            if config.search_config.engine:
                preferred = config.search_config.engine.lower()
            if config.search_config.fallback_engines:
                fallbacks = [
                    engine.lower() for engine in config.search_config.fallback_engines
                ]

        engine_order = []
        # 首先添加首选引擎
        if preferred in self._search_engine:
            engine_order.append(preferred)

        # 将配置的后备引擎添加到顺序
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

        Args:
            engine: 搜索引擎实例
            query: 搜索查询
            num_results: 结果数量

        Returns:
            List[str]: 搜索结果URL列表

        Raises:
            Exception: 搜索引擎执行失败时抛出的异常
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: list(engine.perform_search(query, num_results=num_results)),
            )
        except Exception as e:
            logger.debug(f"搜索引擎执行失败，准备重试: {str(e)}")
            raise
