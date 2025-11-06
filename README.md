# Free Fermion Library

[![PyPI version](https://img.shields.io/pypi/v/free-fermion-lib.svg)](https://pypi.org/project/free-fermion-lib/)
[![Python versions](https://img.shields.io/pypi/pyversions/free-fermion-lib.svg)](https://pypi.org/project/free-fermion-lib/)
<!--[![Build Status](https://github.com/jdwhitfield/free-fermion-lib/workflows/CI/badge.svg)](https://github.com/jdwhitfield/free-fermion-lib/actions)-->
[![Downloads](https://img.shields.io/pypi/dm/free-fermion-lib.svg)](https://pypi.org/project/free-fermion-lib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/free-fermion-lib/badge/?version=latest)](https://free-fermion-lib.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jdwhitfield/free-fermion-lib/branch/main/graph/badge.svg)](https://codecov.io/gh/jdwhitfield/free-fermion-lib)

A comprehensive Python library for working with free fermion quantum systems, providing tools for combinatorial functions, graph theory algorithms, quantum physics utilities, and advanced distance measures for quantum state analysis. The library includes statistical and quantum distance measures for characterizing quantum states and their proximity to classical, stabilizer, or free-fermion subspaces. This work was supported by the U.S. Department of Energy, Office of Basic Energy Sciences, under Award DE-SC0019374 and by Army Research Office grant W911NF2410043.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Documentation](#documentation)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Development](#development)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

### From PyPI (Recommended)

```bash
pip install free-fermion-lib
```

### From Source

```bash
git clone https://github.com/jdwhitfield/free-fermion-lib.git
cd free-fermion-lib
pip install -e .
```

### Development Installation

For development with all optional dependencies:

```bash
git clone https://github.com/jdwhitfield/free-fermion-lib.git
cd free-fermion-lib
pip install -e ".[dev,docs]"
```

### Requirements

- **Python**: >= 3.8
- **Core Dependencies**:
  - NumPy >= 1.20.0
  - SciPy >= 1.7.0
  - NetworkX >= 2.6.0
  - Matplotlib >= 3.3.0

## Quick Start

```python
import numpy as np
import ff

# Generate Jordan-Wigner operators for 3 sites
n_sites = 3
alphas = ff.jordan_wigner_alphas(n_sites)

# Generate a Gaussian state
rho = ff.random_FF_state(n_sites)

# Compute correlation matrix
gamma = ff.compute_2corr_matrix(rho, n_sites, alphas)

# Compute pfaffian of a skew-symmetric matrix
skew_matrix = np.array([[0, 1, -2], [-1, 0, 3], [2, -3, 0]])
pfaffian_value = ff.pf(skew_matrix)
```

## Features

### Core Modules

- **[`ff_lib`](https://free-fermion-lib.readthedocs.io/en/latest/api.html#ff-lib-module)**: Core free-fermion functions
  - Jordan-Wigner transformations (Dirac and Majorana fermions)
  - Symplectic free-fermion diagonalization
  - Gaussian state generation and manipulation
  - Fermionic correlation matrix computations
  - Wick's theorem implementation

- **[`ff_distance_measures`](https://free-fermion-lib.readthedocs.io/en/latest/api.html#ff-distance-measures-module)**: Quantum state distance measures and entropy functions
  - Stabilizer Rényi Entropy (SRE) for quantifying magic resources
  - Fermionic Anti-Flatness (FAF) for measuring deviation from free-fermion behavior
  - Statistical distance measures: Total variation, Jensen-Shannon divergence, Bhattacharyya coefficient
  - Information-theoretic measures: Kullback-Leibler divergence, Rényi entropy, linear entropy
  - Quantum distance measures: Trace distance for density matrices
  - Fermionic covariance distribution analysis

- **[`ff_combinatorics`](https://free-fermion-lib.readthedocs.io/en/latest/api.html#ff-combinatorics-module)**: Combinatorial matrix functions
  - Pfaffian computation via combinatorial formula
  - Hafnian computation
  - Permanent and determinant calculations
  - Sign of permutation functions

- **[`ff_graph_theory`](https://free-fermion-lib.readthedocs.io/en/latest/api.html#ff-graph-theory-module)**: Graph algorithms and visualization
  - Pfaffian ordering algorithm (FKT algorithm) for planar graphs
  - Perfect matching algorithms
  - Planar graph generation and visualization
  - Dual graph construction

- **[`ff_utils`](https://free-fermion-lib.readthedocs.io/en/latest/api.html#ff-utils-module)**: Common utility functions
  - Matrix cleaning and formatting
  - Random bitstring generation
  - Direct sum operations
  - Pretty printing with numerical precision control

### Statistical Distance Measures

The [`ff_distance_measures`](src/ff/ff_distance_measures.py) module provides a comprehensive suite of distance measures for quantum state analysis:

**Quantum-Specific Measures:**
- **Stabilizer Rényi Entropy (SRE)**: Quantifies "magic" or non-stabilizer resources in quantum states
- **Fermionic Anti-Flatness (FAF)**: Measures deviation from free-fermion (Gaussian) behavior
- **Trace Distance**: Standard quantum distance measure between density matrices

**Statistical Distance Measures:**
- **Jensen-Shannon Divergence**: Symmetric, bounded version of KL divergence
- **Total Variation Distance**: L1-based distance between probability distributions
- **Bhattacharyya Coefficient**: Similarity measure between distributions
- **Kullback-Leibler Divergence**: Information-theoretic divergence measure

**Entropy Measures:**
- **Rényi Entropy**: Generalized entropy with tunable parameter α
- **Linear Entropy**: Efficient measure of mixedness for quantum states

These measures are fundamental tools for characterizing quantum states and their proximity to classical, stabilizer, or free-fermion subspaces in quantum many-body systems.

## Documentation

📚 **[Full Documentation](https://free-fermion-lib.readthedocs.io/)**

- [Installation Guide](https://free-fermion-lib.readthedocs.io/en/latest/installation.html)
- [Quick Start Tutorial](https://free-fermion-lib.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://free-fermion-lib.readthedocs.io/en/latest/api.html)
- [Examples and Tutorials](https://free-fermion-lib.readthedocs.io/en/latest/examples.html)
- [Contributing Guide](https://free-fermion-lib.readthedocs.io/en/latest/contributing.html)

## Examples

### Basic Pfaffian Computation

```python
import numpy as np
from ff.ff_combinatorics import pf

# Create a skew-symmetric matrix
matrix = np.array([[0, 1, -2, 3],
                   [-1, 0, 4, -5],
                   [2, -4, 0, 6],
                   [-3, 5, -6, 0]])

# Compute pfaffian
pfaffian_value = pf(matrix)
print(f"Pfaffian: {pfaffian_value}")
```

### Jordan-Wigner Transformation

```python
from ff.ff_lib import jordan_wigner_alphas, build_H, build_op
import numpy as np

# Generate operators for a 4-site system
n_sites = 4
alphas = jordan_wigner_alphas(n_sites)

# Create a hopping Hamiltonian
hopping_matrix = np.diag(np.ones(n_sites-1), 1) + np.diag(np.ones(n_sites-1), -1)
H = build_H(n_sites, hopping_matrix)

H_op = build_op(H,alphas)
```

### Graph Theory Applications

```python
from ff.ff_graph_theory import pfaffian_orientation
import networkx as nx

# Create a planar graph
G = nx.grid_2d_graph(3, 3)
# Find pfaffian orientation
oriented_graph = pfaffian_orientation(G)
```

### Distance Measures and Quantum State Analysis

```python
from ff.ff_distance_measures import SRE, FAF, jensen_shannon_divergence, trace_distance
import numpy as np

# Compute Stabilizer Rényi Entropy for a quantum state
rho = np.eye(4) / 4  # Maximally mixed 2-qubit state
sre_value = SRE(rho, a=2)
print(f"Stabilizer Rényi Entropy: {sre_value}")

# Compute Fermionic Anti-Flatness
faf_value = FAF(rho, k=2)
print(f"Fermionic Anti-Flatness: {faf_value}")

# Compare two probability distributions
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])
js_div = jensen_shannon_divergence(p, q)
print(f"Jensen-Shannon Divergence: {js_div}")

# Compute trace distance between density matrices
rho1 = np.array([[0.7, 0], [0, 0.3]])
rho2 = np.array([[0.6, 0], [0, 0.4]])
td = trace_distance(rho1, rho2)
print(f"Trace Distance: {td}")
```

## API Reference

The complete API documentation is available at [free-fermion-lib.readthedocs.io](https://free-fermion-lib.readthedocs.io/en/latest/api.html).

### Key Functions

| Module | Function | Description |
|--------|----------|-------------|
| `ff_lib` | `jordan_wigner_alphas()` | Generate Jordan-Wigner operators |
| `ff_lib` | `build_H()` | Construct Hamiltonian matrices |
| `ff_lib` | `random_FF_state()` | Generate random Gaussian states |
| `ff_distance_measures` | `SRE()` | Compute Stabilizer Rényi Entropy |
| `ff_distance_measures` | `FAF()` | Compute Fermionic Anti-Flatness |
| `ff_distance_measures` | `stabilizer_distribution()` | Compute stabilizer probability distribution |
| `ff_distance_measures` | `jensen_shannon_divergence()` | Compute Jensen-Shannon divergence |
| `ff_distance_measures` | `total_variation_distance()` | Compute total variation distance |
| `ff_distance_measures` | `trace_distance()` | Compute trace distance between density matrices |
| `ff_distance_measures` | `relative_entropy()` | Compute Kullback-Leibler divergence |
| `ff_combinatorics` | `pf()` | Compute matrix pfaffian |
| `ff_combinatorics` | `hafnian()` | Compute matrix hafnian |
| `ff_graph_theory` | `pfaffian_orientation()` | Find pfaffian orientation of graphs |
| `ff_utils` | `clean_matrix()` | Clean numerical matrices |

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://free-fermion-lib.readthedocs.io/en/latest/contributing.html) for details.

<!--
### Quick Contributing Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/free-fermion-lib.git
   cd free-fermion-lib
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
5. **Make your changes** and add tests
6. **Run the test suite**:
   ```bash
   pytest
   ```
7. **Check code formatting**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```
8. **Commit and push** your changes
9. **Submit a pull request**
-->

### Development Guidelines

- Follow [PEP 8](https://pep8.org/) style guidelines
- Add tests for new functionality
- Update documentation for API changes
- Ensure all tests pass before submitting PR
- Use meaningful commit messages

### Reporting Issues

Please report bugs and feature requests on our [GitHub Issues](https://github.com/jdwhitfield/free-fermion-lib/issues) page.

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/jdwhitfield/free-fermion-lib.git
cd free-fermion-lib

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=ff --cov-report=html

# Run specific test file
pytest tests/test_ff_lib.py

# Run tests in parallel
pytest -n auto
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation locally
cd docs/
make html

# View documentation
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```

<!--
### Performance Testing

```bash
# Run performance benchmarks
pytest tests/test_performance.py -v

# Profile specific functions
python -m cProfile -s cumulative your_script.py
```
-->
## Citation

If you use this library in your research, please cite:

```bibtex
@software{free_fermion_lib,
  author = {James D. Whitfield},
  title = {Free Fermion Library: A Python package for quantum free fermion systems},
  version = {1.0.0},
  year = {2025},
  url = {https://github.com/jdwhitfield/free-fermion-lib},
}
```

<!--
### Related Publications

If this library contributes to your research, please also consider citing the foundational work:

- Whitfield, J.D. et al. "Free fermion systems and quantum computation." *Journal of Quantum Physics* (2025).
-->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project depends on several open-source packages:
- NumPy (BSD License)
- SciPy (BSD License)
- NetworkX (BSD License)
- Matplotlib (PSF License)

## Acknowledgments

This work was supported by:
- U.S. Department of Energy, Office of Basic Energy Sciences, under Award DE-SC0019374
- Army Research Office grant W911NF2410043

Special thanks to the contributors and the open-source community.

## Contact

**James D. Whitfield**
📧 Email: [James.D.Whitfield@dartmouth.edu](mailto:James.D.Whitfield@dartmouth.edu)
🏛️ Institution: Dartmouth College
🌐 Website: [https://github.com/jdwhitfield](https://github.com/jdwhitfield)

### Support

- 📖 **Documentation**: [free-fermion-lib.readthedocs.io](https://free-fermion-lib.readthedocs.io/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/jdwhitfield/free-fermion-lib/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/jdwhitfield/free-fermion-lib/discussions)
- 📧 **Email Support**: For academic collaborations and research inquiries

---

<div align="center">

**Free Fermion Library** - Advancing quantum physics research through open-source software

[![GitHub stars](https://img.shields.io/github/stars/jdwhitfield/free-fermion-lib.svg?style=social&label=Star)](https://github.com/jdwhitfield/free-fermion-lib)
[![GitHub forks](https://img.shields.io/github/forks/jdwhitfield/free-fermion-lib.svg?style=social&label=Fork)](https://github.com/jdwhitfield/free-fermion-lib/fork)

</div>
