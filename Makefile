.PHONY: start start-dev test

start:
	uv run thaumaturgy

start-dev:
	uv run python -m thaumaturgy.main

test:
	uv run pytest
