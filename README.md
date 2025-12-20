<table width="100%" style="table-layout: fixed;">
  <tr>
    <td align="left" valign="middle" style="white-space: nowrap;">
      <img src="https://raw.githubusercontent.com/yohanderose/lmmule/master/docs/logo.png" alt="Llama avec ciggy" width="140" style="height:auto; max-width: 100%;" />
    </td>
    <td align="right" valign="middle" style="white-space: nowrap;">
      <h1>LM.Mule</h1>
      <strong>Lightweight framework for complex agentic orchestration.</strong>
    </td>
  </tr>
</table>

Create custom Mules to break down, research and prepare solutions for you in the background.

## Setup

```bash
git clone https://github.com/yohanderose/lmmule.git
cd lmmule
python -m pip install .
```

Ollama setup https://docs.ollama.com/quickstart

To enable remote llm calls, sign up to OpenRouter, create an API key and ensure the `OPENROUTER_API_KEY` environment variable is set https://openrouter.ai/docs/quickstart

## Usage

Please check out [`lmmule/examples`](https://github.com/yohanderose/lmmule/tree/master/lmmule/examples) for complete (and concise) examples.

**Running examples**

```bash
python lmmule/examples/simple.py # Using local Ollama instance

python lmmule/examples/simple.py --remote --model "xiaomi/mimo-v2-flash:free" # With Openrouter
```

**High level roll-your-own**

[`lmmule/examples/simple.py`](https://github.com/yohanderose/lmmule/blob/master/lmmule/examples/simple.py)

```python
import asyncio
import json

from lmmule.examples.allmules import Thinker, Critic


async def main():
    task1 = Thinker(
        "mule1-bob",
        "phi4-mini",
        base_prompt="very concisely explain the meaning of life",
    )()
    task2 = await Critic(
        "mule2-jane",
        "phi4-mini",
        base_prompt="evaluate this answer to the meaning of life: {}",
    )(prior1=task1)
    print(json.dumps(task2, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

![Example output](https://raw.githubusercontent.com/yohanderose/lmmule/master/docs/lmmule-output.png)

## Agentic Flows

I've included an example comparing agentic vs non-agentic approaches. As demonstrated by many others, combining the efforts of distributed task specific workers yields much higher quality and more relevant output. This is often at the cost of speed, as the aggregating `Researcher` Mule can take 1-3x the time of `Thinker` alone.

[`lmmule/examples/agentic_bench.py`](https://github.com/yohanderose/lmmule/blob/master/lmmule/examples/agentic_bench.py)

![Example output](https://raw.githubusercontent.com/yohanderose/lmmule/master/docs/agentic-eval.png)

## TODO

- base tools in abstract class
  - fast API
    - doc processing pipe
    - agentic rag flow
  - remote / local fallback
  - websearch tool
    - !!research papers
    - cache visited pages with ttl
    - selenium driverless fallback on simple request fail
- auto pull ollama model/config if not exists
- manage local resources and concurrent tasks, mayb w cap suggestions
- mypy type force and docs
