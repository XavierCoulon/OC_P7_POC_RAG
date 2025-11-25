# Development commands
.PHONY: up down rebuild precommit test coverage eval-mistral eval-huggingface

# Precommit commands
precommit:
	pre-commit run --all-files

# Docker commands
up:
	docker compose up -d

down:
	docker compose down

rebuild:
	docker compose down
	docker compose build
	docker compose up -d

# Run tests
test:
	pytest -v

# covergage report
coverage:
	pytest --cov=app --cov-report=term-missing --cov-report=html

# Ragas evaluation commands
eval-mistral:
	@echo "🚀 Evaluating RAG with Mistral embeddings..."
	python scripts/ragas_eval.py --provider mistral

eval-huggingface:
	@echo "🚀 Evaluating RAG with HuggingFace embeddings..."
	python scripts/ragas_eval.py --provider huggingface
