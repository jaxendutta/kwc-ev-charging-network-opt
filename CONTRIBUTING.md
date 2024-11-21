# Contributing to KW EV Charging Station Optimization

We welcome contributions to the KW EV Charging Station Optimization project! This document provides guidelines and information for contributors.

## Table of Contents
- [Contributing to KW EV Charging Station Optimization](#contributing-to-kw-ev-charging-station-optimization)
  - [Table of Contents](#table-of-contents)
  - [Development Setup](#development-setup)
    - [Environment Setup](#environment-setup)
    - [Development Installation](#development-installation)
  - [Development Workflow](#development-workflow)
    - [1. Branching Strategy](#1-branching-strategy)
    - [2. Code Style](#2-code-style)
    - [3. Testing](#3-testing)
    - [4. Documentation](#4-documentation)
  - [Making Changes](#making-changes)
    - [1. Creating a New Feature](#1-creating-a-new-feature)
    - [2. Fixing Bugs](#2-fixing-bugs)
    - [3. Improving Documentation](#3-improving-documentation)
  - [Pull Request Process](#pull-request-process)
  - [Data Management](#data-management)
  - [Common Development Tasks](#common-development-tasks)
  - [Project Structure Guidelines](#project-structure-guidelines)
  - [Contact](#contact)

## Development Setup

### Environment Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/kw-ev-charging-optimization.git
   cd kw-ev-charging-optimization
   ```

3. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

### Development Installation
1. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Branching Strategy
- `main` - stable production code
- `develop` - development branch
- Feature branches: `feature/your-feature-name`
- Bug fix branches: `fix/bug-description`
- Documentation branches: `docs/description`

### 2. Code Style
We follow PEP 8 with additional conventions:

1. Use Black for formatting:
   ```bash
   black src/ tests/
   ```

2. Line length: 88 characters (Black default)

3. Import order:
   ```python
   # Standard library
   import os
   import sys
   
   # Third-party packages
   import numpy as np
   import pandas as pd
   
   # Local modules
   from src.data import utils
   ```

4. Documentation strings:
   ```python
   def function_name(param1: type, param2: type) -> return_type:
       """Short description.
       
       Detailed description of function behavior.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ExceptionType: Description of when this exception occurs
       """
   ```

### 3. Testing
1. Write tests for new features:
   ```python
   # tests/test_module.py
   def test_new_feature():
       """Test description."""
       expected = ...
       result = ...
       assert result == expected
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

3. Check coverage:
   ```bash
   pytest --cov=src tests/
   ```

### 4. Documentation
- Update docstrings for all new functions/classes
- Update README.md if adding new features
- Document any changes to data structures or APIs

## Making Changes

### 1. Creating a New Feature
1. Create feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement feature with tests

3. Update documentation

4. Run test suite:
   ```bash
   pytest tests/
   black src/ tests/
   flake8 src/ tests/
   ```

### 2. Fixing Bugs
1. Create bug fix branch:
   ```bash
   git checkout -b fix/bug-description
   ```

2. Add test case that reproduces bug
3. Fix bug
4. Verify all tests pass

### 3. Improving Documentation
1. Create documentation branch:
   ```bash
   git checkout -b docs/description
   ```

2. Make documentation changes
3. Build and verify documentation

## Pull Request Process
1. Update relevant documentation
2. Run full test suite
3. Push changes to your fork
4. Create PR against `develop` branch
5. Await code review

## Data Management
- Store raw data in `data/raw/`
- Process data in `data/processed/`
- Use provided utility functions in `src.data.utils`
- Document any new data sources or formats

## Common Development Tasks
1. Adding a new data source:
   - Add API client in `src/data/api_client.py`
   - Add validation in `src/data/utils.py`
   - Create tests in `tests/test_api_client.py`

2. Modifying optimization model:
   - Update model formulation in notebooks
   - Add new constraints/objectives
   - Document changes in notebook markdown

3. Adding visualizations:
   - Add new visualization functions to `src/visualization/map_viz.py`
   - Follow existing style for consistency
   - Include examples in notebooks

## Project Structure Guidelines
```plaintext
src/
├── data/           # Data processing modules
├── models/         # Optimization models
└── visualization/  # Visualization utilities

notebooks/          # Analysis notebooks
tests/              # Test modules
```

## Contact
- Create an issue for bugs or feature requests
- Contact maintainers for other questions