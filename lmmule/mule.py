import json
import asyncio
import logging
import aiohttp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Generator
from googlesearch import search
from markdownify import markdownify as md
from lxml import html, etree

logging.basicConfig(filename="/tmp/mule.log", level=logging.INFO, filemode="a+")
OLLAMA_URL = "http://localhost:11434/api/chat"
ALLOWED_TAG_DEFAULT = {"div", "h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "code"}


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

                return {"error": await response.text()}

    @classmethod
    async def google(cls, query: str, num_results: int) -> Generator:
        return search(query, num_results=num_results, advanced=True)

    @classmethod
    async def scrape_page(cls, title: str, url: str, allowed_tags: set[str]) -> dict:
        page = (await cls.request("GET", url)).get("text", "")
        tree = html.fromstring(page).find("body")
        tmp = None

        for elem in tree.xpath(".//*"):
            if elem.tag not in allowed_tags:  # strip unwanted page elements
                elem.getparent().remove(elem)

        for elem in tree.xpath(".//*"):  # pick a 'main' div by density
            if tmp is None:
                tmp = elem
                continue

            if len(etree.tostring(elem, encoding="unicode")) > len(
                etree.tostring(tmp, encoding="unicode")
            ):
                tmp = elem

        return {
            "title": title,
            "url": url,
            "content": md(
                (
                    etree.tostring(tmp, encoding="unicode", pretty_print=True)
                    if tmp is not None
                    else ""
                ),
            ),
        }


@dataclass
class Mule(ABC, Multils):
    mule_name: str
    model_name: str
    base_prompt: str = ""
    topics: str = ""
    output_format: dict = field(default_factory=dict)
    search_num_res: int = 10
    search_allowed_tags: set[str] = field(default_factory=lambda: ALLOWED_TAG_DEFAULT)

    def __post_init__(self):
        self.log = MuleLoggerAdapter(
            logging.getLogger(__name__), {"mule_name": self.mule_name}
        )

    @abstractmethod
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        pass

    async def _llm_call(self, prompt: str) -> list[dict]:
        chat_history = [{"role": "user", "content": prompt}]
        resp = None
        payload = {
            "model": self.model_name,
            "stream": False,
            "format": self.output_format,
            "messages": chat_history,
        }

        try:
            resp = await Mule.request("POST", OLLAMA_URL, payload=payload)
            chat_history += [{"role": "system", "content": resp["message"]["content"]}]
            self.log.info(
                f"""Ollama call:
                \ninput: {json.dumps(chat_history[0], indent=2)}
                \noutput: {json.dumps(chat_history[1], indent=2)}"""
            )
        except Exception as e:
            self.log.error(
                f"""Could not call Ollama | {e}\n{resp}
                \npayload:{json.dumps(payload, indent=2)}"""
            )
        return chat_history

    async def _websearch(self, prompt: str) -> list:
        search_results = await Mule.google(prompt, num_results=self.search_num_res)
        return await asyncio.gather(
            *(
                Mule.scrape_page(r.title, r.url, self.search_allowed_tags)
                for r in search_results
            )
        )
