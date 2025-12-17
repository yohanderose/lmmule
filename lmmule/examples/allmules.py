from typing import Awaitable
import asyncio

from lmmule.mule import Mule


class Thinker(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        return await self.llm_call(self.base_prompt)


class Critic(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        results = dict(
            zip(depends_on.keys(), (await asyncio.gather(*depends_on.values())))
        )
        chat_history = results["prior1"]
        most_recent_msg = chat_history[-1]["content"]
        return chat_history + await self.llm_call(
            self.base_prompt.format(most_recent_msg)
        )
