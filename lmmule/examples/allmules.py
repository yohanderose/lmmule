import sys
from typing import Awaitable
import asyncio
import itertools

from lmmule.mule import Mule


class Thinker(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        if depends_on:
            results = await asyncio.gather(*depends_on.values())
            self.chat_history = list(itertools.chain.from_iterable(results))
        return await self.llm_call(self.base_prompt)


class Critic(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        results = dict(
            zip(depends_on.keys(), (await asyncio.gather(*depends_on.values())))
        )
        most_recent_msg = results["prior1"][-1]["content"]
        return await self.llm_call(self.base_prompt.format(most_recent_msg))


class Researcher(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        results = await asyncio.gather(*depends_on.values())
        notes = "\n\n".join([r[-1]["content"] for r in results])
        return await self.llm_call(self.base_prompt.format(notes))
