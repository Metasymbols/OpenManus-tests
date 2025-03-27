from typing import List

import requests
from bs4 import BeautifulSoup

from app.logger import logger
from app.tool.search.base import WebSearchEngine


ABSTRACT_MAX_LENGTH = 300

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/49.0.2623.108 Chrome/49.0.2623.108 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; pt-BR) AppleWebKit/533.3 (KHTML, like Gecko) QtWeb Internet Browser/3.7 http://www.QtWeb.net",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.2 (KHTML, like Gecko) ChromePlus/4.0.222.3 Chrome/4.0.222.3 Safari/532.2",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.4pre) Gecko/20070404 K-Ninja/2.1.3",
    "Mozilla/5.0 (Future Star Technologies Corp.; Star-Blade OS; x86_64; U; en-US) iNet Browser 4.7",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.13) Gecko/20080414 Firefox/2.0.0.13 Pogo/2.0.0.13.6866",
]

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": USER_AGENTS[0],
    "Referer": "https://www.bing.com/",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

BING_HOST_URL = "https://www.bing.com"
BING_SEARCH_URL = "https://www.bing.com/search?q="


class BingSearchEngine(WebSearchEngine):
    session: requests.Session = None

    def __init__(self, **data):
        """Initialize the BingSearch tool with a requests session."""
        super().__init__(**data)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _search_sync(self, query: str, num_results: int = 10) -> List[str]:
        """
        Synchronous Bing search implementation to retrieve a list of URLs matching a query.

        Args:
            query (str): The search query to submit to Bing. Must not be empty.
            num_results (int, optional): The maximum number of URLs to return. Defaults to 10.

        Returns:
            List[str]: A list of URLs from the search results, capped at `num_results`.
                       Returns an empty list if the query is empty or no results are found.

        Notes:
            - Pagination is handled by incrementing the `first` parameter and following `next_url` links.
            - If fewer results than `num_results` are available, all found URLs are returned.
        """
        if not query:
            logger.warning("Bing搜索查询为空")
            return []

        logger.debug(f"开始Bing搜索，查询: '{query}'，请求结果数: {num_results}")
        list_result = []
        first = 1
        next_url = BING_SEARCH_URL + query
        logger.debug(f"初始Bing搜索URL: {next_url}")

        page_count = 0
        while len(list_result) < num_results:
            page_count += 1
            logger.debug(f"获取Bing搜索结果页 #{page_count}, URL: {next_url}")

            data, next_url = self._parse_html(
                next_url, rank_start=len(list_result), first=first
            )

            if data:
                new_urls = [item["url"] for item in data]
                logger.debug(f"从当前页面解析到 {len(new_urls)} 个结果")
                list_result.extend(new_urls)
            else:
                logger.debug("当前页面未解析到任何结果")

            if not next_url:
                logger.debug("没有下一页，搜索结束")
                break

            first += 10
            # 限制最大页数，避免无限循环
            if page_count >= 3:
                logger.debug(f"达到最大页数限制({page_count}页)，停止获取更多结果")
                break

        logger.info(f"Bing搜索完成，共获取到 {len(list_result)} 个结果")
        return list_result[:num_results]

    def _parse_html(self, url: str, rank_start: int = 0, first: int = 1) -> tuple:
        """
        Parse Bing search result HTML synchronously to extract search results and the next page URL.

        Args:
            url (str): The URL of the Bing search results page to parse.
            rank_start (int, optional): The starting rank for numbering the search results. Defaults to 0.
            first (int, optional): Unused parameter (possibly legacy). Defaults to 1.
        Returns:
            tuple: A tuple containing:
                - list: A list of dictionaries with keys 'title', 'abstract', 'url', and 'rank' for each result.
                - str or None: The URL of the next results page, or None if there is no next page.
        """
        try:
            res = self.session.get(url=url)
            res.encoding = "utf-8"
            root = BeautifulSoup(res.text, "lxml")

            list_data = []
            ol_results = root.find("ol", id="b_results")
            if not ol_results:
                logger.warning(f"Bing搜索未找到结果列表元素(b_results)，URL: {url}")
                # 检查是否有验证码或其他错误页面
                if "验证" in res.text or "captcha" in res.text.lower():
                    logger.error("Bing搜索可能需要验证码验证，请检查网络环境")
                return [], None

            for li in ol_results.find_all("li", class_="b_algo"):
                title = ""
                url = ""
                abstract = ""
                try:
                    h2 = li.find("h2")
                    if h2:
                        title = h2.text.strip()
                        url = h2.a["href"].strip()

                    p = li.find("p")
                    if p:
                        abstract = p.text.strip()

                    if ABSTRACT_MAX_LENGTH and len(abstract) > ABSTRACT_MAX_LENGTH:
                        abstract = abstract[:ABSTRACT_MAX_LENGTH]

                    rank_start += 1
                    list_data.append(
                        {
                            "title": title,
                            "abstract": abstract,
                            "url": url,
                            "rank": rank_start,
                        }
                    )
                except Exception:
                    continue

            next_btn = root.find("a", title="Next page")
            if not next_btn:
                return list_data, None

            next_url = BING_HOST_URL + next_btn["href"]
            return list_data, next_url
        except Exception as e:
            logger.warning(f"Error parsing HTML: {e}")
            return [], None

    def perform_search(self, query, num_results=10, *args, **kwargs):
        """Bing search engine."""
        return self._search_sync(query, num_results=num_results)
