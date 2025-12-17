import asyncio
from datetime import datetime
from typing import Awaitable

from lmmule.mule import Mule
from lmmule.examples.allmules import *


async def research():
    class Research(Mule):
        async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
            sources = await self._websearch(self.topics)
            content = [item["content"] for item in sources]

            grad_students = [
                Thinker(
                    f"minimule-{i}",
                    "gemma3:4b",
                    base_prompt=f"take concise notes on {self.topics} from the following source:\n{src}",
                )
                for i, src in enumerate(content)
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

    class Research2(Mule):
        async def __call__(self, **depends_on: Awaitable[list[dict]]) -> list[dict]:
            sources = await self._websearch(self.topics)
            content = "\n\n".join([item["content"] for item in sources])
            return await self.llm_call(
                f"{self.base_prompt}, refer to the following sources:\n{content}"
            )

    def write_to_file(path, value):
        with open(path, "w+") as f:
            f.write(value)

    ts1 = datetime.now()
    task = await Research(
        "mule4-alice",
        "gemma3:4b",
        topics="flutter tts",
        base_prompt="code to simply implement tts in flutter application",
        search_num_res=6,
    )()
    if task[-1] and task[-1]["role"] == "system":
        write_to_file("agent.md", task[-1]["content"])
    else:
        write_to_file("agent.md", "")

    ts2 = datetime.now()
    task = await Research2(
        "mule4-paul",
        "gemma3:4b",
        topics="flutter tts",
        base_prompt="code to simply implement tts in flutter application",
        search_num_res=6,
    )()
    if task[-1] and task[-1]["role"] == "system":
        write_to_file("non-agent.md", task[-1]["content"])
    else:
        write_to_file("non-agent.md", "")

    d1 = (ts2 - ts1).seconds
    d2 = (datetime.now() - ts2).seconds
    print(f"  agentic took {int(d1 // 60)}m:{int(d1 % 60)}s")
    print(f"  non-agentic took {int(d2 // 60)}m:{int(d2 % 60)}s")


async def main():
    await asyncio.gather(research())


if __name__ == "__main__":
    asyncio.run(main())
