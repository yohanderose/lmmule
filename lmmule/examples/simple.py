import asyncio
import json
import argparse

from lmmule.mule import Mule
from lmmule.examples.allmules import Thinker, Critic
import lmmule.mule


async def main():
    Mule.init_args()
    model_name = lmmule.mule.args.model

    task1 = Thinker(
        "mule1-bob",
        model_name=model_name,
        base_prompt="very concisely explain the meaning of life",
    )()
    task2 = await Critic(
        "mule2-jane",
        model_name=model_name,
        base_prompt="evaluate this answer to the meaning of life: {}",
    )(prior1=task1)
    print(json.dumps(task2, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
