

lsimple:
	python lmmule/examples/simple.py

rsimple:
	python lmmule/examples/simple.py --remote --model "xiaomi/mimo-v2-flash:free"

abench:
	python lmmule/examples/agentic_bench.py --remote --model "xiaomi/mimo-v2-flash:free"

dd:
	python draft.py --remote --model "xiaomi/mimo-v2-flash:free"
