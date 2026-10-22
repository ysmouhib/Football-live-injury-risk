.PHONY: check lint fmtcheck fmt type test test-all dist

check: lint fmtcheck type test

lint:
	ruff check src tests

fmtcheck:
	ruff format --check src tests

fmt:
	ruff format src tests

type:
	mypy src tests

test:
	pytest -m "not pending"

test-all:
	pytest

# Source archive of the last commit (tracked files only)
dist:
	mkdir -p dist
	git archive --format=zip --prefix=live-injury-risk/ -o dist/live-injury-risk-$$(git describe --always).zip HEAD
