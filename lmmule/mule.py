import asyncio
import logging
import aiohttp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Generator
from googlesearch import search
from html_to_markdown import convert_to_markdown
from lxml import html, etree

logging.basicConfig(filename="/tmp/mule.log", level=logging.INFO, filemode="a+")
OLLAMA_URL = "http://localhost:11434/api/chat"
ALLOWED_TAGS = {"div", "h1", "h2", "h3", "h4", "h5", "h6", "p", "code"}


class MuleLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        prefix = f"[{self.extra.get('mule_name', 'Mule')}] "
        return prefix + msg, kwargs


class Multils:
    @classmethod
    async def request(cls, method: str, url: str, *, payload=None) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.request(method.upper(), url, json=payload) as response:
                if response.status == 200:
                    try:
                        return await response.json(content_type=None)
                    except Exception:
                        return {"text": await response.text()}
        return {}

    @classmethod
    async def google(cls, query: str, num_results=10) -> Generator:
        return search(query, num_results=num_results, advanced=True)

    @classmethod
    async def scrape_page(cls, title: str, url: str) -> dict:
        page = (await Mule.request("GET", url)).get("text", "")

        tree = html.fromstring(page).find("body")
        for elem in tree.xpath(".//*"):
            if elem.tag not in ALLOWED_TAGS:  # strip unwanted page elements
                elem.drop_tag()

        return {
            "title": title,
            "url": url,
            "content": convert_to_markdown(
                etree.tostring(tree, encoding="unicode", pretty_print=True),
                parser="lxml",
            ),
        }


@dataclass
class Mule(ABC, Multils):
    mule_name: str
    model_name: str
    base_prompt: str = ""
    topics: str = ""
    output_format: dict = field(default_factory=dict)

    def __post_init__(self):
        self.log = MuleLoggerAdapter(
            logging.getLogger(__name__), {"mule_name": self.mule_name}
        )

    @abstractmethod
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        pass

    async def _llm_call(self, prompt: str) -> list[dict]:
        chat_history = [{"role": self.mule_name, "content": prompt}]
        reply = ""

        payload = {
            "model": self.model_name,
            "messages": chat_history,
            "stream": False,
            "format": self.output_format,
        }

        try:
            reply = (
                (await Mule.request("POST", OLLAMA_URL, payload=payload))
                .get("message", {})
                .get("content", "")
            )
        except Exception as e:
            self.log.error(f"Could not call Ollama | {e}\n{payload}")

        state = chat_history + [{"role": self.model_name, "content": reply}]
        self.log.info(f"ollama call result : {state}")
        return state

    async def _websearch(self, prompt: str) -> list:
        search_results = await Mule.google(prompt, num_results=3)
        return await asyncio.gather(
            *(Mule.scrape_page(r.title, r.url) for r in search_results)
        )
