# Contributing to Federated Learning Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.9+
- Docker & Docker Compose V2
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Nikhil172913832/Federated_Learning.git
cd Federated_Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
cd complete/fl
pip install -e ".[dev]"
```

## Code Quality Standards

### Code Style

We use the following tools to maintain code quality:

```bash
# Format code with black
black fl/ tests/

# Lint with flake8
flake8 fl/ tests/

# Type check with mypy
mypy fl/ --ignore-missing-imports

# Sort imports with isort
isort fl/ tests/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
pip install pre-commit
pre-commit install
```

## Testing Requirements

All contributions must include tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=fl --cov-report=term-missing

# Run specific test file
pytest tests/test_data_validation.py -v

# Run property-based tests
pytest tests/test_data_validation.py -v --hypothesis-show-statistics
```

### Test Coverage

- Minimum coverage: 80%
- All new functions must have tests
- Include edge cases and error conditions

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add docstrings to all public functions
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run tests
pytest tests/ -v --cov=fl

# Check code quality
black --check fl/ tests/
flake8 fl/ tests/
mypy fl/
```

### 4. Commit Changes

Follow conventional commits format:

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in data loading"
git commit -m "docs: update README"
git commit -m "test: add tests for evaluation module"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable

## PR Review Checklist

Before submitting, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Coverage >= 80%
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts
- [ ] PR description is clear and complete

## Code Review Guidelines

### For Reviewers

- Be constructive and respectful
- Focus on code quality, not personal preferences
- Suggest improvements with examples
- Approve when standards are met

### For Contributors

- Respond to feedback promptly
- Ask questions if unclear
- Make requested changes
- Mark conversations as resolved

## Documentation Standards

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
    """
    pass
```

### Type Hints

Always include type hints:

```python
from typing import List, Dict, Optional

def process_data(
    data: List[int],
    config: Dict[str, Any],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    pass
```

## Testing Guidelines

### Unit Tests

Test individual functions in isolation:

```python
def test_load_data_validates_inputs():
    """Test that load_data validates partition bounds."""
    with pytest.raises(ValueError):
        load_data(partition_id=10, num_partitions=5)
```

### Integration Tests

Test component interactions:

```python
def test_federated_round():
    """Test complete FL round with real components."""
    # Setup
    server = FederatedServer(...)
    clients = [FederatedClient(...) for _ in range(3)]
    
    # Execute
    results = run_fl_round(server, clients)
    
    # Verify
    assert results["accuracy"] > 0.5
```

### Property-Based Tests

Use Hypothesis for edge cases:

```python
from hypothesis import given, strategies as st

@given(partition_id=st.integers(), num_partitions=st.integers())
def test_partition_bounds(partition_id, num_partitions):
    """Property: partition_id must be in [0, num_partitions)."""
    if not (0 <= partition_id < num_partitions and num_partitions > 0):
        with pytest.raises(ValueError):
            load_data(partition_id, num_partitions)
```

## Issue Reporting

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Alternatives considered

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Give credit where due
- Follow the Code of Conduct

## Questions?

- Open a [GitHub Discussion](https://github.com/Nikhil172913832/Federated_Learning/discussions)
- Check existing issues and PRs
- Read the documentation

Thank you for contributing! 🎉
