# Contributions

We welcome contributions to the KWC EV Charging Station Network Optimization project! This document outlines the process for contributing to the project.

## Table of Contents
- [Contributions](#contributions)
  - [Table of Contents](#table-of-contents)
  - [Development Setup](#development-setup)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)
  - [Development Workflow](#development-workflow)
    - [Branch Strategy](#branch-strategy)
    - [Testing](#testing)
  - [Code Standards](#code-standards)
    - [Style Guidelines](#style-guidelines)
    - [Documentation Requirements](#documentation-requirements)
    - [Import Organization](#import-organization)
  - [Making Changes](#making-changes)
    - [Adding Features](#adding-features)
    - [Fixing Bugs](#fixing-bugs)
    - [Modifying the Model](#modifying-the-model)
  - [Pull Request Process](#pull-request-process)
  - [Data Management](#data-management)
    - [Data Structure](#data-structure)
    - [Data Guidelines](#data-guidelines)
  - [Project Organization](#project-organization)
    - [Directory Structure](#directory-structure)
    - [Key Components](#key-components)
  - [Contact](#contact)

## Development Setup

### Prerequisites
- Python 3.12.7 or higher
- Gurobi Optimizer License
- OpenChargeMap API key
- Git

### Environment Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/kw-ev-charging-optimization.git
   cd kw-ev-charging-optimization
   ```

3. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e .
   ```

5. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch
- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`

### Testing
- Run tests: `pytest tests/`
- Check coverage: `pytest --cov=src tests/`
- Verify setup: `python src/verify_setup.py`

## Code Standards

### Style Guidelines
- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 88 characters
- Comprehensive docstrings

### Documentation Requirements
- Function docstrings with:
  - Purpose
  - Parameters
  - Returns
  - Examples
- Module documentation
- Updated README for new features

### Import Organization
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

## Making Changes

### Adding Features
1. Create feature branch
2. Implement with tests
3. Update documentation
4. Run full test suite
5. Submit pull request

### Fixing Bugs
1. Create bug fix branch
2. Add reproducing test
3. Fix bug
4. Verify tests
5. Submit pull request

### Modifying the Model
1. Update notebooks
2. Document changes
3. Test solver behavior
4. Update parameters
5. Validate results

## Pull Request Process
1. Update documentation
2. Run all tests
3. Format code
4. Create detailed PR description
5. Await review

## Data Management

### Data Structure
```plaintext
data/
├── raw/               # Original data files
│   ├── boundaries/    # Geographic boundaries
│   ├── charging_stations/
│   ├── ev_fsa/        # EV ownership data
│   ├── population/    # Census data
│   └── potential_locations/
└── processed/         # Processed datasets
    ├── demand_points/
    ├── ev_fsa_analyzed/
    ├── integrated_analyzed_data/
    └── optimization_inputs/
```

### Data Guidelines
1. Raw data is immutable
2. Document all processing steps
3. Use provided utility functions
4. Include data validation
5. Maintain data versions

## Project Organization

### Directory Structure
```plaintext
kw-ev-charging-optimization/
├── data/           # Data storage
├── notebooks/      # Analysis notebooks
├── src/            # Source code
│   ├── data/       # Data processing
│   ├── model/      # Optimization model
│   └── visualization/
├── tests/          # Test modules
└── configs/        # Configuration files
```

### Key Components
1. Data Manager (`src/data/data_manager.py`)
2. Network Optimizer (`src/model/network_optimizer.py`)
3. Analysis Notebooks (`notebooks/`)
4. Visualization Tools (`src/visualization/`)

## Contact
- Submit issues for bugs
- Start discussions for features
- Tag maintainers for review