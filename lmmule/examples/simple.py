import asyncio
import json
import argparse

from lmmule.examples.allmules import Thinker, Critic
import lmmule.mule


async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--model", default="phi4-mini", help="LLM model name")
    args = parser.parse_args()

    lmmule.mule.USE_REMOTE = args.remote

    task1 = Thinker(
        "mule1-bob",
        model_name=args.model,
        base_prompt="very concisely explain the meaning of life",
    )()
    task2 = await Critic(
        "mule2-jane",
        model_name=args.model,
        base_prompt="evaluate this answer to the meaning of life: {}",
    )(prior1=task1)
    print(json.dumps(task2, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
