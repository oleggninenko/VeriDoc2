.PHONY: help install install-dev test lint format clean build docker-build docker-run docker-stop deploy-staging deploy-production

# Default target
help:
	@echo "Available commands:"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  test           - Run tests"
	@echo "  test-cov       - Run tests with coverage"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black and isort"
	@echo "  clean          - Clean build artifacts"
	@echo "  build          - Build Python package"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-stop    - Stop Docker container"
	@echo "  deploy-staging - Deploy to staging environment"
	@echo "  deploy-prod    - Deploy to production environment"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev,test,docs]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 . --max-line-length=120 --extend-ignore=E203,W503
	black --check --diff --line-length 120 .
	isort --check-only --diff .
	bandit -r . -f json -o bandit-report.json || true

format:
	black . --line-length 120
	isort . --profile black --line-length 120

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Building
build:
	python -m build

# Docker
docker-build:
	docker build -t veridoc-ai:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Deployment (placeholder commands - customize for your environment)
deploy-staging:
	@echo "Deploying to staging environment..."
	# Add your staging deployment commands here
	# Example: kubectl apply -f k8s/staging/

deploy-prod:
	@echo "Deploying to production environment..."
	# Add your production deployment commands here
	# Example: kubectl apply -f k8s/production/

# Development helpers
run:
	uvicorn simple_web_interface_v2:app --reload --host 0.0.0.0 --port 8000

run-docker:
	docker run -p 8000:8000 -v $(PWD)/uploads:/app/uploads veridoc-ai:latest

# Security
security-scan:
	bandit -r . -f json -o bandit-report.json
	safety check

# Documentation
docs-serve:
	mkdocs serve

docs-build:
	mkdocs build

# Git helpers
git-hooks:
	pre-commit install

# Environment setup
setup-env:
	@echo "Setting up development environment..."
	python -m venv .venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source .venv/bin/activate  # On Unix/macOS"
	@echo "  .venv\\Scripts\\activate     # On Windows"

# Quick start
quick-start: setup-env install-dev git-hooks
	@echo "Development environment setup complete!"
	@echo "Run 'make run' to start the application"
