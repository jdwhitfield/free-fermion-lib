API Reference
=============

This page contains the complete API reference for the Free Fermion Library.

Core Module (ff_lib)
--------------------

.. automodule:: ff.ff_lib
   :members:
   :undoc-members:
   :show-inheritance:

Combinatorics Module (ff_combinatorics)
---------------------------------------

.. automodule:: ff.ff_combinatorics
   :members:
   :undoc-members:
   :show-inheritance:

Graph Theory Module (ff_graph_theory)
-------------------------------------

.. automodule:: ff.ff_graph_theory
   :members:
   :undoc-members:
   :show-inheritance:

Distance Measures Module (ff_distance_measures)
-----------------------------------------------

.. automodule:: ff.ff_distance_measures
   :members:
   :undoc-members:
   :show-inheritance:

The distance measures module provides fundamental tools for quantifying quantum states
and their proximity to classical, stabilizer, or free-fermion subspaces. Key measures include:

- **Stabilizer Rényi Entropy (SRE)**: Quantifies deviation from stabilizer states
- **Fermionic Anti-Flatness (FAF)**: Measures distance from free-fermion behavior
- **General entropy measures**: Rényi entropy, linear entropy for probability distributions
- **Stabilizer and covariance distributions**: Core probability distributions for analysis
- **Distance measures**: Total variation distance, trace distance, relative entropy
- **Divergence measures**: Jensen-Shannon divergence, Bhattacharyya coefficient

Distance Measures Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_distance_measures.stabilizer_distribution
   ff.ff_distance_measures.SRE
   ff.ff_distance_measures.FAF
   ff.ff_distance_measures.renyi_entropy
   ff.ff_distance_measures.linear_entropy
   ff.ff_distance_measures.cov_distribution
   ff.ff_distance_measures.total_variation_distance
   ff.ff_distance_measures.trace_distance
   ff.ff_distance_measures.relative_entropy
   ff.ff_distance_measures.jensen_shannon_divergence
   ff.ff_distance_measures.bhattacharyya_coeff

Random States Module (ff_random_states)
---------------------------------------

.. automodule:: ff.ff_random_states
   :members:
   :undoc-members:
   :show-inheritance:

The random states module provides comprehensive random state generation capabilities
for quantum systems, including:

- **Haar random states**: Uniformly distributed pure states over qubit Hilbert spaces
- **Clifford states**: Random stabilizer states using the Stim package (optional dependency)
- **Free fermion states**: Various methods for generating random FF states with different constraints
- **Path construction**: Unitary and linear interpolation between quantum states

Note: Some functions require the optional ``stim`` package for Clifford state generation.

Utilities Module (ff_utils)
---------------------------

.. automodule:: ff.ff_utils
   :members:
   :undoc-members:
   :show-inheritance:

Function Index
--------------

Core Quantum Physics Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jordan-Wigner Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_lib.jordan_wigner_lowering
   ff.ff_lib.jordan_wigner_alphas
   ff.ff_lib.jordan_wigner_majoranas
   ff.ff_lib.rotate_operators
   ff.ff_lib._perform_rotation

Matrix Construction
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_lib.build_V
   ff.ff_lib.build_H
   ff.ff_lib.build_Omega
   ff.ff_lib.build_K
   ff.ff_lib.build_reordering_xx_to_xp
   ff.ff_lib.permutation_to_matrix
   ff.ff_lib.pauli_matrices
   ff.ff_lib.build_op

Gaussian States and Correlation Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_lib.generate_gaussian_state
   ff.ff_lib.compute_cov_matrix
   ff.ff_lib.compute_2corr_matrix
   ff.ff_lib.compute_algebra_S
   ff.ff_lib.correlation_matrix
   ff.ff_lib.random_FF_rotation
   ff.ff_lib.random_FF_state
   ff.ff_lib.random_H_generator
   ff.ff_lib.kitaev_chain

Symplectic Operations
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_lib.eigh_sp
   ff.ff_lib.eigv_sp
   ff.ff_lib.eigm_sp_can
   ff.ff_lib.eigm_sp
   ff.ff_lib.is_symp
   ff.ff_lib.check_canonical_form

Quantum Physics Analysis Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_lib.is_matchgate
   ff.ff_lib.wick_contraction

Combinatorial Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ff.ff_combinatorics.sgn
   ff.ff_combinatorics.pf
   ff.ff_combinatorics.hf
   ff.ff_combinatorics.pt
   ff.ff_combinatorics.dt
   ff.ff_combinatorics.dt_eigen

Graph Theory Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ff.ff_graph_theory.pfo_algorithm
   ff.ff_graph_theory.plot_graph_with_edge_weights
   ff.ff_graph_theory.generate_random_planar_graph
   ff.ff_graph_theory.plot_planar_embedding
   ff.ff_graph_theory.dual_graph_H
   ff.ff_graph_theory.faces
   ff.ff_graph_theory.complete_face

Graph Visualization Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_graph_theory._draw_labeled_multigraph
   ff.ff_graph_theory._draw_labeled_graph

Perfect Matching Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_graph_theory.count_perfect_matchings
   ff.ff_graph_theory.count_perfect_matchings_planar
   ff.ff_graph_theory.find_perfect_matchings_brute

Tree Analysis Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ff.ff_graph_theory.compute_tree_depth

Utility Functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ff.ff_utils._print
   ff.ff_utils.clean
   ff.ff_utils.formatted_output
   ff.ff_utils.generate_random_bitstring
   ff.ff_utils.kron_plus