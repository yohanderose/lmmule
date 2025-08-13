import json
from typing import Awaitable
import asyncio

from lmmule.mule import Mule


class Thinker(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        return await self._llm_call(self.base_prompt)


class Critic(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        results = dict(
            zip(depends_on.keys(), (await asyncio.gather(*depends_on.values())))
        )
        chat_history = results["dep1"]
        most_recent_msg = chat_history[-1]["content"]
        return chat_history + await self._llm_call(
            self.base_prompt.format(most_recent_msg)
        )


async def main():
    task1 = Thinker(
        "mule1-bob",
        "phi4-mini",
        base_prompt="very concisely explain the meaning of life",
    )()
    task2 = await Critic(
        "mule2-jane",
        "phi4-mini",
        base_prompt="please evaluate this answer to the meaning of life: {}",
    )(dep1=task1)
    print(json.dumps(task2, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
