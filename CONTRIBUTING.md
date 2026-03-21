# Contributing to Personal LLM RAG System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Questions?](#questions)

## Code of Conduct

This project adheres to a standard code of conduct:
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/personal-llm-rag.git
   cd personal-llm-rag
   ```
3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/GauravPathania18/personal-llm-rag.git
   ```
4. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\Activate.ps1  # Windows
   ```
5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
6. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report:
- Check if the issue already exists
- Collect information about the bug (logs, error messages, steps to reproduce)

Include in your report:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Screenshots/logs if applicable

### Suggesting Features

- Use a clear, descriptive title
- Provide detailed description of the feature
- Explain why this feature would be useful
- Consider potential implementation approaches

### Contributing Code

Areas where contributions are especially welcome:
- **Documentation improvements**
- **Bug fixes**
- **Performance optimizations**
- **New features** (discuss in an issue first)
- **Tests**
- **UI/UX improvements**

## Development Workflow

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following our code style guidelines

4. **Test your changes**:
   ```bash
   pytest
   ```

5. **Commit your changes** with clear messages

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters max
- **Import order**: stdlib, third-party, local
- **Type hints**: encouraged for function signatures
- **Docstrings**: Google style preferred

### Formatting

Use these tools before committing:

```bash
# Format code
black . --line-length 100

# Sort imports
isort . --profile black

# Lint code
ruff check .
```

### Pre-commit Hooks (Optional)

Install pre-commit hooks to run checks automatically:

```bash
pip install pre-commit
pre-commit install
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_specific.py
```

## Commit Messages

Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, missing semi colons, etc)
- `refactor:` Code refactoring
- `test:` Adding or refactoring tests
- `chore:` Maintenance tasks

Examples:
```
feat: add WebSocket support for real-time chat
fix: resolve ChromaDB connection timeout issue
docs: update API endpoint documentation
refactor: optimize RAPTOR clustering algorithm
```

## Pull Request Process

1. **Update your branch** with latest upstream changes
2. **Ensure CI passes** (tests, linting)
3. **Fill out the PR template** completely
4. **Link related issues** using keywords (Fixes #123, Closes #456)
5. **Request review** from maintainers
6. **Address review feedback**
7. **Squash commits** if requested

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] PR description explains changes clearly
- [ ] No merge conflicts

## Project Structure

Understand the codebase:

```
personal-llm-rag/
├── Backend/                  # FastAPI orchestrator
│   ├── app/api/             # Route handlers
│   ├── app/services/        # Business logic
│   └── app/core/            # Configuration
├── personal_LLM_embedder/  # Embedding service
├── VECTOR_STORAGE_SERVICE/  # Vector DB with RAPTOR
│   └── app/services/        # RAPTOR, reranker, etc.
└── tests/                   # Test suite
```

## Questions?

- **General questions**: Open a [Discussion](https://github.com/GauravPathania18/personal-llm-rag/discussions)
- **Bug reports**: Open an [Issue](https://github.com/GauravPathania18/personal-llm-rag/issues)
- **Feature requests**: Open an [Issue](https://github.com/GauravPathania18/personal-llm-rag/issues)

## Recognition

Contributors will be recognized in our README and release notes.

Thank you for contributing! 🚀
