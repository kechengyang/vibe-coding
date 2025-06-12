# Contributing to Employee Health Monitoring System

Thank you for your interest in contributing to the Employee Health Monitoring System! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others when contributing to the project.

## How Can I Contribute?

There are many ways to contribute to the Employee Health Monitoring System:

1. **Reporting Bugs**: If you find a bug, please report it by creating an issue in the GitHub repository.
2. **Suggesting Features**: If you have an idea for a new feature, please suggest it by creating an issue in the GitHub repository.
3. **Writing Code**: If you want to contribute code, please follow the guidelines below.
4. **Improving Documentation**: If you find errors or omissions in the documentation, please help improve it.
5. **Reviewing Pull Requests**: Help review pull requests from other contributors.

## Reporting Bugs

When reporting bugs, please include:

1. **Steps to Reproduce**: Detailed steps to reproduce the bug.
2. **Expected Behavior**: What you expected to happen.
3. **Actual Behavior**: What actually happened.
4. **Environment**: Information about your environment (OS, Python version, etc.).
5. **Screenshots**: If applicable, add screenshots to help explain the problem.

## Suggesting Features

When suggesting features, please include:

1. **Feature Description**: A clear and concise description of the feature.
2. **Use Case**: Why this feature would be useful.
3. **Possible Implementation**: If you have ideas on how to implement the feature, please share them.

## Development Workflow

### Setting Up Development Environment

1. Fork the repository on GitHub.
2. Clone your fork locally:
```bash
git clone https://github.com/your-username/employee-health-monitoring.git
cd employee-health-monitoring
```

3. Set up the development environment:
```bash
python install.py
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
```

### Making Changes

1. Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes.
3. Run tests to ensure your changes don't break existing functionality:
```bash
pytest
```

4. Commit your changes:
```bash
git commit -m "Add your commit message here"
```

5. Push your changes to your fork:
```bash
git push origin feature/your-feature-name
```

6. Create a pull request from your fork to the main repository.

### Pull Request Process

1. Ensure your code follows the project's coding standards.
2. Update the documentation if necessary.
3. Include tests for your changes.
4. Ensure all tests pass.
5. Fill out the pull request template.

## Coding Standards

The project follows these coding standards:

1. **PEP 8**: Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
2. **Type Hints**: Use type hints for function and method signatures.
3. **Docstrings**: Include docstrings for all modules, classes, and functions.
4. **Tests**: Write tests for all new functionality.

The project uses pre-commit hooks to enforce these standards. You can run the pre-commit hooks manually:

```bash
pre-commit run --all-files
```

## Testing

The project uses pytest for testing. You can run the tests with:

```bash
pytest
```

When adding new functionality, please include tests for that functionality.

## Documentation

The project uses Markdown for documentation. When adding or modifying functionality, please update the documentation accordingly.

## License

By contributing to the Employee Health Monitoring System, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

If you have any questions about contributing, please create an issue in the GitHub repository.

Thank you for your contributions!
