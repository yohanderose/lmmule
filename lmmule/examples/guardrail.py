import asyncio
import argparse
import json

from lmmule.examples.allmules import Thinker
import lmmule.mule


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--model", default="phi4-mini", help="LLM model name")
    args = parser.parse_args()

    lmmule.mule.USE_REMOTE = args.remote
    model_name = args.model

    messages = [
        "what is the meaning of life",
        "where can i buy a car for cheap",
        "will you be my ai girlfriend",
        "remedies for arthritis",
    ]

    for i, m in enumerate(messages):
        t = await Thinker(
            f"mule{i}-veda",
            model_name=model_name,
            output_format={  # Ollama uses this
                "type": "object",
                "properties": {"result": {"type": "number"}},
                "required": ["result"],
            },
            base_prompt=f"""
            You are an assistant to an Ayurveda practitioner.
            You are to classify the following client message as either 1 or 0.
            Return 1 if the message is strictly related to health and wellness, diet
            and lifestyle; within the realm of holistic medicine.
            Return 0 if irrelevant or inappropriate.
            The message is contained between $$ lines.

            $$
            {m}
            $$
            """,
        )()

        print(
            bool(t[-1]["content"])
            if lmmule.mule.USE_REMOTE
            else bool(json.loads(t[-1]["content"])["result"])
        )


if __name__ == "__main__":
    asyncio.run(main())
