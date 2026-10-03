# Developer workflow. Windows users without `make`: run the raw commands from
# each target (see README "Local Setup").

PYTHON ?= .venv/Scripts/python.exe
ifeq ($(OS),Windows_NT)
    PYTHON ?= .venv/Scripts/python.exe
else
    PYTHON ?= .venv/bin/python
endif

.PHONY: install dev lint format typecheck test test-unit test-integration eval migrate seed docker-up docker-down docker-build load-test infra-up infra-down clean

install:            ## create venv + install deps
	python -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

dev:                ## run API locally (reload)
	$(PYTHON) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

lint:               ## ruff lint + format check
	$(PYTHON) -m ruff check app tests scripts migrations
	$(PYTHON) -m ruff format --check app tests scripts migrations

format:             ## ruff auto-format
	$(PYTHON) -m ruff format app tests scripts migrations
	$(PYTHON) -m ruff check --fix app tests scripts migrations

typecheck:          ## mypy
	$(PYTHON) -m mypy app

test: test-unit test-integration

test-unit:          ## fast unit tests (no services)
	$(PYTHON) -m pytest tests/unit -v

test-integration:   ## integration/e2e/security (needs Postgres :5434 + Redis :6390)
	$(PYTHON) -m pytest tests/integration tests/e2e tests/security -v

eval:               ## AI evaluation (heuristic router, CI-safe)
	$(PYTHON) scripts/generate_eval_dataset.py
	$(PYTHON) scripts/run_evaluation.py --router heuristic

migrate:            ## apply migrations
	$(PYTHON) -m alembic -c migrations/alembic.ini upgrade head

seed:               ## seed local dev data + admin user
	$(PYTHON) scripts/seed_local.py

docker-build:       ## build production image
	docker build -f docker/Dockerfile -t multi-agent-orchestrator:latest .

docker-up:          ## full stack: api + postgres + redis + prometheus + grafana
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down

infra-up:           ## just the dev databases (postgres :5434, redis :6390)
	docker run -d --name mao-postgres -e POSTGRES_PASSWORD=orchestrator -e POSTGRES_USER=orchestrator -e POSTGRES_DB=orchestrator -p 5434:5432 postgres:16-alpine
	docker run -d --name mao-redis -p 6390:6379 redis:7-alpine

infra-down:
	docker rm -f mao-postgres mao-redis 2>/dev/null || true

load-test:          ## needs running API + LOADTEST_API_KEY env var
	LOADTEST_API_KEY=$(LOADTEST_API_KEY) $(PYTHON) -m locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 --headless -u 50 -r 5 -t 60s --csv docs/performance/run50

security-scan:
	$(PYTHON) -m bandit -c pyproject.toml -r app -q
	$(PYTHON) -m pip_audit --skip-editable

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
