run:
	uv run evaluation.py --task hearsay --model gpt-5-nano --mode all --debug

fmt:
	uv run ruff format src/ *.py pyproject.toml
