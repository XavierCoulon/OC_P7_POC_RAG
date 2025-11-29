# Development commands
.PHONY: up down rebuild precommit test coverage eval-mistral eval-huggingface rebuild-index rebuild-index-mistral rebuild-index-huggingface

# Load .env file
include .env

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

# Rebuild index commands (requires API running)
rebuild-index-mistral:
	@echo "🔄 Rebuilding index with Mistral embeddings..."
	@API_KEY_CLEAN=$$(echo $(API_KEY) | tr -d "'"); \
	curl -X POST http://localhost:8000/rebuild?provider=mistral \
		-H "X-API-Key: $$API_KEY_CLEAN" \
		-H "Content-Type: application/json" \
		2>/dev/null | python -m json.tool
	@echo "✅ Index rebuild complete"

rebuild-index-huggingface:
	@echo "🔄 Rebuilding index with HuggingFace embeddings..."
	@API_KEY_CLEAN=$$(echo $(API_KEY) | tr -d "'"); \
	curl -X POST http://localhost:8000/rebuild?provider=huggingface \
		-H "X-API-Key: $$API_KEY_CLEAN" \
		-H "Content-Type: application/json" \
		2>/dev/null | python -m json.tool
	@echo "✅ Index rebuild complete"

rebuild-index: rebuild-index-mistral
	@echo "Index rebuild done (Mistral)"
