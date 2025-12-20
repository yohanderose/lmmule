import asyncio
import json

from lmmule.mule import Mule
from lmmule.examples.allmules import *
import lmmule.mule


async def main():
    Mule.init_args()
    model_name = lmmule.mule.args.model

    t1 = Thinker(
        "mule12-eve",
        model_name=model_name,
        base_prompt="who was albert einstein?",
    )()
    t2 = Thinker(
        "mule12-ben",
        model_name=model_name,
        base_prompt="was he American?",
    )(prior=t1)

    t3 = Thinker(
        "mule12-sally",
        model_name=model_name,
        base_prompt="who was newton?",
    )()

    t4 = await Thinker(
        "mule12-charles",
        model_name=model_name,
        base_prompt="who made bigger contributions?",
    )(prior1=t2, prior2=t3)

    print(json.dumps(t4, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
