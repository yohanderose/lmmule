import asyncio
import sys
import json
import argparse
from datetime import datetime
from typing import Awaitable

from lmmule.mule import Mule
from lmmule.examples.allmules import Thinker, Researcher
import lmmule.mule

from rich.console import Console
from rich.markdown import Markdown

console = Console()


async def main():
    Mule.init_args()
    model_name = lmmule.mule.args.model
    topics = "index file DB in C for mcu application"

    sources = await Mule.websearch(topics, num_res=8)
    contents = [item["content"] for item in sources]

    ts1 = datetime.now()
    grad_students = {
        f"{i}": Thinker(
            f"minimule-{i}",
            model_name=model_name,
            base_prompt=f"take careful notes on {topics} from the following source:\n{src}",
        )()
        for i, src in enumerate(contents)
    }
    task1 = await Researcher(
        "mule4-alice",
        model_name=model_name,
        base_prompt="refer to the following notes and give code to implement \n\n{}",
    )(**grad_students)
    console.print(Markdown(task1[-1]["content"]))

    notes = "\n".join(contents)
    ts2 = datetime.now()
    task2 = await Thinker(
        "mule4-paul",
        model_name=model_name,
        base_prompt=f"refer to the following notes and give code to implement \n\n{notes}",
    )()
    console.print(Markdown(task2[-1]["content"]))

    d1 = (ts2 - ts1).seconds
    d2 = (datetime.now() - ts2).seconds

    task3 = await Thinker(
        "mule6-jim",
        model_name=model_name,
        base_prompt=f"carefully analyse and compare the below 2 guides separated by >><<, and explain which is better, first or second. \n{task1[-1]['content']}\n>><<\n{task2[-1]['content']}",
    )()
    console.print(Markdown(task3[-1]["content"]))

    print(f"  agentic took {int(d1 // 60)}m:{int(d1 % 60)}s")
    print(f"  non-agentic took {int(d2 // 60)}m:{int(d2 % 60)}s")


if __name__ == "__main__":
    asyncio.run(main())
