import json
import os
import sys
import asyncio
import logging
import aiohttp
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Generator
from markdownify import markdownify as md
from ddgs import DDGS
from lxml import html, etree

logging.basicConfig(filename="/tmp/mule.log", level=logging.INFO, filemode="a+")

args = None
USE_REMOTE = False
OLLAMA_URL = "http://localhost:11434/api/chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

BLOCKED_SOURCES = ["youtube.com", "google.com", "facebook.com"]
ALLOWED_TAG_DEFAULT = {
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "pre",
    "code",
}


class MuleLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        prefix = f"[{self.extra.get('mule_name', 'Mule')}] "
        return prefix + msg, kwargs


class Multils:
    @classmethod
    def init_args(cls):
        global args, USE_REMOTE

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--remote", action="store_true", help="Tries to use OpenRouter if provided"
        )
        parser.add_argument("--model", default="phi4-mini", help="LLM model name")

        args = parser.parse_args()
        USE_REMOTE = args.remote

    @classmethod
    async def request(
        cls, method: str, url: str, *, payload=None, headers=None
    ) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method.upper(), url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    try:
                        return await response.json(content_type=None)
                    except Exception:
                        try:
                            return {"text": await response.text()}
                        except Exception:
                            return {"text": ""}

                return {"error": await response.text()}

    @classmethod
    def ddg_search(cls, query: str, num_results: int) -> list:
        return DDGS().text(query, max_results=num_results, region="wt-wt")

    @classmethod
    async def scrape_page(cls, title: str, url: str, allowed_tags: set[str]) -> dict:
        if any(domain in url for domain in BLOCKED_SOURCES):
            return {}

        def score_content_density(elem) -> float:
            """Score element based on content density and quality indicators."""
            text = elem.text_content().strip()
            text_len = len(text)
            if text_len < 50:
                return 0

            # Count meaningful content elements
            paragraphs = len(elem.xpath(".//p"))
            headings = len(elem.xpath(".//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6"))
            lists = len(elem.xpath(".//ul | .//ol"))

            # Base score from text length (diminishing returns)
            score = min(text_len / 100, 20)

            # Bonus for content structure
            score += paragraphs * 2  # Paragraphs are good
            score += headings * 3  # Headings indicate article structure
            score += lists * 1  # Lists add value

            # Calculate text density (text vs HTML ratio)
            html_len = len(etree.tostring(elem, encoding="unicode"))
            density = text_len / max(html_len, 1)
            score += density * 10

            # Penalty for excessive links (likely navigation/sidebar)
            links = elem.xpath(".//a")
            if links:
                link_text_len = sum(
                    len((link.text_content() or "").strip()) for link in links
                )
                link_ratio = link_text_len / max(text_len, 1)
                if link_ratio > 0.4:  # More than 40% links is suspicious
                    score *= 0.5

            return score

        def find_content_heavy_div(tree):
            """Find the div/section with the highest content score."""
            # Check semantic elements first
            for tag in ["article", "main", "section"]:
                candidates = tree.xpath(f".//{tag}")
                if candidates:
                    scored = [
                        (elem, score_content_density(elem)) for elem in candidates
                    ]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    if scored[0][1] > 5:  # good enough score
                        return scored[0][0]

            # Fall back to divs with content indicators
            content_divs = []

            # Look for divs with content-related classes/IDs
            content_indicators = [
                "content",
                "article",
                "post",
                "story",
                "main",
                "body",
                "text",
                "entry",
            ]
            for indicator in content_indicators:
                divs = tree.xpath(
                    f'.//div[contains(@class, "{indicator}") or contains(@id, "{indicator}")]'
                )
                content_divs.extend(divs)

            # Add all divs as fallback
            all_divs = tree.xpath(".//div")
            content_divs.extend(all_divs)

            # Remove duplicates and score
            unique_divs = list(set(content_divs))
            scored_divs = [(div, score_content_density(div)) for div in unique_divs]
            scored_divs.sort(key=lambda x: x[1], reverse=True)

            # Return best scoring div with minimum threshold
            if scored_divs and scored_divs[0][1] > 3:
                return scored_divs[0][0]

            return tree

        page = (await cls.request("GET", url)).get("text", "")
        if not page or not page.strip():
            return {}

        tree = html.fromstring(page)
        tree = tree.find("body") if tree.find("body") is not None else tree

        primary_content_div = find_content_heavy_div(tree)

        all_tags = set([elem.tag for elem in tree.iter()])
        tags_to_remove = list(all_tags - allowed_tags)
        etree.strip_elements(primary_content_div, *tags_to_remove, with_tail=False)

        return {
            "title": title,
            "url": url,
            "content": md(
                (
                    etree.tostring(
                        primary_content_div, encoding="unicode", pretty_print=True
                    )
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
    search_allowed_tags: set[str] = field(default_factory=lambda: ALLOWED_TAG_DEFAULT)

    def __post_init__(self):
        self.log = MuleLoggerAdapter(
            logging.getLogger(__name__), {"mule_name": self.mule_name}
        )
        self.chat_history = []

    @abstractmethod
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        pass

    @classmethod
    async def websearch(
        cls, query: str, num_res: int, allowed_tags: set = ALLOWED_TAG_DEFAULT
    ) -> list:
        search_results = Mule.ddg_search(query, num_results=int(num_res * 2))

        sources: list[dict] = [
            item
            for item in await asyncio.gather(
                *(
                    Mule.scrape_page(r["title"], r["href"], allowed_tags)
                    for r in search_results
                    if r
                )
            )
            if item is not None
            and item.get("content")
            and len(item.get("content", "").split()) > 100
        ]

        # for src in sources:
        #     print(f">> {src['title']}\n({src['url']})\n")
        #     l = len(src["content"].split("\n"))
        #     print(f"{l}\n\n")
        # print(f"  - {len(sources)} pages")
        return sources[:num_res]

    async def _ollama_call(self) -> list[dict]:
        resp = None
        payload = {
            "model": self.model_name,
            "stream": False,
            "format": self.output_format,
            "messages": self.chat_history,
        }

        try:
            resp = await Mule.request("POST", OLLAMA_URL, payload=payload)
            self.chat_history += [
                {"role": "system", "content": resp["message"]["content"]}
            ]
            self.log.info(
                f"""Ollama call:
                \ninput: {json.dumps(self.chat_history[-2], indent=2)}
                \noutput: {json.dumps(self.chat_history[-1], indent=2)}"""
            )
        except Exception as e:
            self.log.error(
                f"""Could not call Ollama | {e}\n{resp}
                \npayload:{json.dumps(payload, indent=2)}"""
            )
        return self.chat_history

    async def _openrouter_call(self) -> list[dict]:

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key is None:
            print("Env variable 'OPENROUTER_API_KEY' not set, exiting..")
            sys.exit()

        resp = None
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": self.chat_history,
        }

        try:
            resp = await Mule.request(
                "POST", OPENROUTER_URL, payload=payload, headers=headers
            )
            self.chat_history += [
                {"role": "system", "content": resp["choices"][0]["message"]["content"]}
            ]
            self.log.info(
                f"""Openrouter call:
                \ninput: {json.dumps(self.chat_history[-2], indent=2)}
                \noutput: {json.dumps(self.chat_history[-1], indent=2)}"""
            )
        except Exception as e:
            self.log.error(
                f"""Could not call Openrouter | {e}\n{resp}
                \npayload:{json.dumps(payload, indent=2)}"""
            )
        return self.chat_history

    async def llm_call(self, prompt: str) -> list[dict]:
        self.chat_history += [{"role": "user", "content": prompt}]
        return (
            await self._openrouter_call() if USE_REMOTE else await self._ollama_call()
        )
