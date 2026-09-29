# derived from https://github.com/pydantic/pydantic
.DEFAULT_GOAL := help
# `tests` is dropped from the list when the directory is not present
sources = src tools $(wildcard tests)
.ONESHELL:

# .venv/ holds the hand-built per-arch environments (.venv/lincir, .venv/magiclens, ...) that
# every run activates. uv treats the project environment as its own directory and will
# replace it wholesale, taking those with it -- which is exactly what happened on 2026-08-22.
# Point uv's own environment somewhere else so no target here can do that again.
# NOTE: this only covers uv invoked through make. A bare `uv run`/`uv sync` typed in the repo
# root still targets .venv/ unless UV_PROJECT_ENVIRONMENT is exported in your shell too.
export UV_PROJECT_ENVIRONMENT := .venv-uv

.PHONY: .uv  ## Check that uv is installed
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit  ## Check that pre-commit is installed
.pre-commit: .uv
	@uv run pre-commit -V || uv pip install pre-commit

.PHONY: install  ## Install all dependencies and pre-commit hooks
install: .uv
	uv sync --all-groups
	uv run pre-commit install --install-hooks

.PHONY: format  ## Auto-format source files with ruff
format: .uv
	uv run ruff check --fix $(sources)
	uv run ruff format $(sources)

.PHONY: lint  ## Lint source files with ruff
lint: .uv
	uv run ruff check $(sources)
	uv run ruff format --check $(sources)

# Tests need torch/transformers/peft, which live in the hand-built per-arch venvs rather than
# in .venv-uv. VENV overrides which one is used: make test VENV=.venv/magiclens
VENV ?= .venv/lincir

.PHONY: test  ## Run tests (CPU only, no checkpoints needed) in the lincir venv
test:
	PYTHONPATH=src:. $(VENV)/bin/python -m pytest -s --durations=10

.PHONY: coverage  ## Run tests with a coverage report
coverage:
	PYTHONPATH=src:. $(VENV)/bin/python -m coverage run -m pytest --durations=10
	PYTHONPATH=src:. $(VENV)/bin/python -m coverage xml
	PYTHONPATH=src:. $(VENV)/bin/python -m coverage html

.PHONY: all  ## Run lint and tests
all: lint test

.PHONY: clean  ## Clear local caches and build artifacts
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage .coverage.*
	rm -rf build dist site
	rm -f coverage.xml

.PHONY: release  ## Bump version, commit, tag and push (BUMP=major|minor|patch)
release: .uv
ifndef BUMP
	$(error BUMP is not set. Usage: make release BUMP=major|minor|patch)
endif
	@echo "Current version: $$(uv version)"
	@uv version --bump $(BUMP)
	@NEW_VERSION=$$(uv version --short)
	@echo "New version: v$$NEW_VERSION"
	@git add pyproject.toml
	@git commit -m "release: version v$$NEW_VERSION"
	@git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION"
	@git push
	@git push origin tag v$$NEW_VERSION
	@echo "version bumped to $$NEW_VERSION — run 'uv publish' if needed"

.PHONY: help  ## Display this help
help:
	@grep -E \
		'^.PHONY: .*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-19s\033[0m %s\n", $$2, $$3}'

.PHONY: requirements  ## Install the pinned runtime dependencies into the active venv
requirements:
	pip install -r requirements.txt

start_jup:
	jupyter notebook --no-browser --port 40005 --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'