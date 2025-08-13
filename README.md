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

## Usage

Please check out `lmmule/example.py` for the complete (and concise) example.

```python
import json
import asyncio

from lmmule.example import Thinker, Critic

async def main():
    task1 = Thinker(
        "mule1-bob",
        "phi4-mini",
        base_prompt="very concisely explain the meaning of life",
    )()
    task2 = await Critic(
        "mule2-jane",
        "phi4-mini",
        base_prompt="please evaluate this answer to the meaning of life: {}",
    )(dep1=task1)
    print(json.dumps(task2, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

![Example output](https://raw.githubusercontent.com/yohanderose/lmmule/master/docs/lmmule-output.png)

## TODO

- base tools in abstract class
  - enhance ollama call (full payload options)
  - websearch tool
    - cache visited pages with ttl
    - selenium driverless fallback on simple request fail
- auto pull ollama model/config if not exists
- manage local resources and concurrent tasks, mayb w cap suggestions
- mypy type force and docs
