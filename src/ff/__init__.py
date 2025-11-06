"""
Free Fermion Library - A Python package for quantum free fermion systems

This package provides comprehensive tools for working with free fermion
quantum systems, including combinatorial functions, graph theory algorithms,
and quantum physics utilities.

Modules:
    ff_lib: Core quantum physics and linear algebra functions
    ff_combinatorics: Combinatorial matrix functions (pfaffian, hafnian, etc.)
    ff_graph_theory: Graph algorithms and visualization for planar graphs
    ff_distance_measures: Distance measures for quantum states (SRE, FAF, etc.)
    ff_random_states: Random state generators and path construction functions
    ff_utils: Common utility functions

Copyright 2025 James.D.Whitfield@dartmouth.edu
Licensed under MIT License.
"""

__version__ = "1.0.0"
__author__ = "James D. Whitfield"
__email__ = "James.D.Whitfield@dartmouth.edu"

# Combinatorial functions from ff_combinatorics
from .ff_combinatorics import sgn, pf, hf, pt, dt, dt_eigen

# Graph theory functions from ff_graph_theory
from .ff_graph_theory import (
    plot_graph_with_edge_weights,
    generate_random_planar_graph,
    plot_planar_embedding,
    dual_graph_H,
    faces,
    complete_face,
    pfo_algorithm,
    compute_tree_depth,
    count_perfect_matchings,
    find_perfect_matchings_brute,
    count_perfect_matchings_planar,
)

# Core quantum physics functions from ff_lib
from .ff_lib import (
    permutation_to_matrix,
    pauli_matrices,
    generate_pauli_group,
    jordan_wigner_lowering,
    jordan_wigner_alphas,
    jordan_wigner_majoranas,
    rotate_operators,
    build_V,
    build_H,
    build_Omega,
    build_reordering_xx_to_xp,
    build_K,
    random_FF_rotation,
    random_FF_state,
    kitaev_chain,
    is_symp,
    check_canonical_form,
    generate_gaussian_state,
    build_op,
    random_H_generator,
    correlation_matrix,
    compute_cov_matrix,
    compute_2corr_matrix,
    compute_algebra_S,
    is_matchgate,
    eigh_sp,
    eigv_sp,
    eigm_sp_can,
    eigm_sp,
)

# Utility functions from ff_utils
from .ff_utils import (
    print_custom,
    clean,
    formatted_output,
    generate_random_bitstring,
    kron_plus,
)

# Distance measure functions from ff_distance_measures
from .ff_distance_measures import (
    stabilizer_distribution,
    SRE,
    FAF,
    renyi_entropy,
    linear_entropy,
    cov_distribution,
    total_variation_distance,
    trace_distance,
    relative_entropy,
    jensen_shannon_divergence,
    bhattacharyya_coeff,
)

# Random state generation functions from ff_random_states
from .ff_random_states import (
    random_qubit_pure_state,
    random_CHP_state,
    random_FF_state_randH,
    random_FF_state_rotPDF,
    random_FF_pure_state_W0,
    random_FF_pure_state_WN,
    random_FF_pure_state_CN,
    get_orthogonal_vectors,
    build_unitary_path,
    build_linear_path,
)

# Additional utility functions from ff_utils
from .ff_utils import (
    cast_to_density_matrix,
    cast_to_pdf,
    analyze_pdf,
)

# Define what gets imported with "from ff import *"
__all__ = [
    # Core quantum physics functions from ff_lib
    "permutation_to_matrix",
    "pauli_matrices",
    "generate_pauli_group",
    "jordan_wigner_lowering",
    "jordan_wigner_alphas",
    "jordan_wigner_majoranas",
    "rotate_operators",
    "build_V",
    "build_H",
    "build_Omega",
    "build_reordering_xx_to_xp",
    "build_K",
    "kitaev_chain",
    "random_FF_rotation",
    "random_FF_state",
    "random_H_generator",
    "correlation_matrix",
    "is_symp",
    "check_canonical_form",
    "generate_gaussian_state",
    "build_op",
    "compute_cov_matrix",
    "compute_2corr_matrix",
    "compute_algebra_S",
    "is_matchgate",
    "eigh_sp",
    "eigv_sp",
    "eigm_sp_can",
    "eigm_sp",
    # Combinatorial functions from ff_combinatorics
    "sgn",
    "pf",
    "hf",
    "pt",
    "dt",
    "dt_eigen",
    # Graph theory functions from ff_graph_theory
    "plot_graph_with_edge_weights",
    "generate_random_planar_graph",
    "plot_planar_embedding",
    "dual_graph_H",
    "faces",
    "complete_face",
    "count_perfect_matchings",
    "find_perfect_matchings_brute",
    "count_perfect_matchings_planar",
    "pfo_algorithm",
    "compute_tree_depth",
    # Utility functions from ff_utils
    "print_custom",
    "clean",
    "formatted_output",
    "generate_random_bitstring",
    "kron_plus",
    # Distance measure functions from ff_distance_measures
    "stabilizer_distribution",
    "SRE",
    "FAF",
    "renyi_entropy",
    "linear_entropy",
    "cov_distribution",
    "total_variation_distance",
    "trace_distance",
    "relative_entropy",
    "jensen_shannon_divergence",
    "bhattacharyya_coeff",
    # Random state generation functions from ff_random_states
    "random_qubit_pure_state",
    "random_CHP_state",
    "random_FF_state_randH",
    "random_FF_state_rotPDF",
    "random_FF_pure_state_W0",
    "random_FF_pure_state_WN",
    "random_FF_pure_state_CN",
    "get_orthogonal_vectors",
    "build_unitary_path",
    "build_linear_path",
    # Additional utility functions from ff_utils
    "cast_to_density_matrix",
    "cast_to_pdf",
    "analyze_pdf",
]
