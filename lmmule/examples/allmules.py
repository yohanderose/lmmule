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


class ResearchTeam(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        sources = await self._websearch(self.topics)
        contents = [item["content"] for item in sources]

        grad_students = [
            Thinker(
                f"minimule-{i}",
                model_name=self.model_name,
                base_prompt=f"take considered notes on {self.topics} from the following source:\n{src}",
            )
            for i, src in enumerate(contents)
        ]

        results = await asyncio.gather(*(gs() for gs in grad_students))

        notes = "\n\n".join(
            [
                r[-1]["content"]
                for r in results
                if r
                and len(r) > 1
                and r[-1].get("role") == "system"
                and r[-1].get("content")
            ]
        )

        return await self.llm_call(
            f"{self.base_prompt}, refer to my notes below:\n{notes}"
        )


class Research(Mule):
    async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
        sources = await self._websearch(self.topics)
        content = "\n\n".join([item["content"] for item in sources])
        return await self.llm_call(
            f"{self.base_prompt}, refer to the following sources:\n{content}"
        )
