.PHONY: start start-dev

start:
	uv run thaumaturgy

start-dev:
	uv run python -m thaumaturgy.main
