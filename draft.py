import asyncio
import sys
import json
import argparse
from datetime import datetime
from typing import Awaitable

from lmmule.mule import Mule
from lmmule.examples.allmules import *
import lmmule.mule


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--model", default="phi4-mini", help="LLM model name")
    args = parser.parse_args()

    lmmule.mule.USE_REMOTE = args.remote
    model_name = args.model


if __name__ == "__main__":
    asyncio.run(main())
