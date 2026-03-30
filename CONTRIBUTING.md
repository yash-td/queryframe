# Contributing to QueryFrame

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/queryframe.git`
3. Install in dev mode: `pip install -e ".[dev,all]"`
4. Create a branch: `git checkout -b feature/your-feature`

## Development Workflow

1. Write tests first (TDD)
2. Run tests: `pytest`
3. Lint: `ruff check src/ tests/`
4. Format: `ruff format src/ tests/`
5. Type check: `mypy src/queryframe/`

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Maintain 80%+ test coverage
- Follow existing code style
- Update documentation if needed

## Security

If you discover a security vulnerability, please email security@movargroup.com instead of opening an issue.
