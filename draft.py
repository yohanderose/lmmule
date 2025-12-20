import asyncio
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Awaitable

from lmmule.rag import Rag, OllamaEmbedding
from lmmule.mule import Mule
from lmmule.examples.allmules import *
import lmmule.mule


async def main():
    Mule.init_args()
    model_name = lmmule.mule.args.model


if __name__ == "__main__":
    asyncio.run(main())
