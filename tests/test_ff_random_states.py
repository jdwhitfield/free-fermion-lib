"""
Random States Tests for Free Fermion Library

This module contains comprehensive tests for the random state generation
functionality in ff_random_states.py, including random qubit states,
Clifford states, free fermion states, and path construction functions.

Test categories:
- Random Qubit States (Haar random pure states)
- Random Clifford States (CHP states using Stim)
- Random Free Fermion States (mixed and pure, various methods)
- Path Construction Functions (unitary and linear interpolation)
- Utility Functions (orthogonal vector generation)
"""

import numpy as np
import pytest

# Import the library
import ff
import ff.ff_random_states as ff_rs


class TestRandomQubitStates:
    """Test random qubit state generation"""

    def test_random_qubit_pure_state_basic(self):
        """Test basic properties of random qubit pure states"""
        n = 2
        psi = ff_rs.random_qubit_pure_state(n, seed=42)

        # Check dimensions
        expected_dim = 2**n
        assert psi.shape == (expected_dim, 1), f"State should be {expected_dim}x1"

        # Check normalization
        assert np.allclose(np.linalg.norm(psi), 1.0), "State should be normalized"

        # Check that it's a complex vector
        assert psi.dtype == complex, "State should be complex"

    def test_random_qubit_pure_state_reproducibility(self):
        """Test that random qubit states are reproducible with seed"""
        n = 3
        psi1 = ff_rs.random_qubit_pure_state(n, seed=123)
        psi2 = ff_rs.random_qubit_pure_state(n, seed=123)

        assert np.allclose(psi1, psi2), "Should be reproducible with same seed"

    def test_random_qubit_pure_state_different_seeds(self):
        """Test that different seeds produce different states"""
        n = 2
        psi1 = ff_rs.random_qubit_pure_state(n, seed=123)
        psi2 = ff_rs.random_qubit_pure_state(n, seed=456)

        # States should be different (with high probability)
        assert not np.allclose(psi1, psi2, atol=1e-10), "Different seeds should produce different states"

    def test_random_qubit_pure_state_input_validation(self):
        """Test input validation for random qubit pure states"""
        # Test invalid types
        with pytest.raises(TypeError):
            ff_rs.random_qubit_pure_state("2")

        with pytest.raises(TypeError):
            ff_rs.random_qubit_pure_state(2.5)

        # Test invalid values
        with pytest.raises(ValueError):
            ff_rs.random_qubit_pure_state(0)

        with pytest.raises(ValueError):
            ff_rs.random_qubit_pure_state(-1)

    def test_random_qubit_pure_state_various_sizes(self):
        """Test random qubit states for various system sizes"""
        for n in [1, 2, 3, 4]:
            psi = ff_rs.random_qubit_pure_state(n, seed=42)
            expected_dim = 2**n
            assert psi.shape == (expected_dim, 1), f"State should be {expected_dim}x1 for n={n}"
            assert np.allclose(np.linalg.norm(psi), 1.0), f"State should be normalized for n={n}"


class TestRandomCliffordStates:
    """Test random Clifford state generation (requires stim)"""

    def test_random_CHP_state_availability_check(self):
        """Test that CHP state function handles missing stim gracefully"""
        # This test checks the import handling
        # If stim is not available, should raise ImportError with helpful message
        try:
            psi = ff_rs.random_CHP_state(2)
            # If we get here, stim is available
            assert psi.shape == (4, 1), "CHP state should have correct dimensions"
            assert np.allclose(np.linalg.norm(psi), 1.0), "CHP state should be normalized"
        except ImportError as e:
            # Check that error message is helpful
            assert "stim" in str(e).lower(), "Error message should mention stim"
            assert "pip install stim" in str(e), "Error message should provide installation instructions"

    @pytest.mark.skipif(not ff_rs.STIM_AVAILABLE, reason="stim package not available")
    def test_random_CHP_state_basic(self):
        """Test basic properties of random CHP states (only if stim available)"""
        n_qubits = 2
        psi = ff_rs.random_CHP_state(n_qubits)

        # Check dimensions
        expected_dim = 2**n_qubits
        assert psi.shape == (expected_dim, 1), f"State should be {expected_dim}x1"

        # Check normalization
        assert np.allclose(np.linalg.norm(psi), 1.0), "State should be normalized"

    @pytest.mark.skipif(not ff_rs.STIM_AVAILABLE, reason="stim package not available")
    def test_random_CHP_state_different_calls(self):
        """Test that different CHP state calls produce valid states"""
        n_qubits = 2
        
        # Generate multiple states and check they are all valid
        for i in range(3):
            psi = ff_rs.random_CHP_state(n_qubits)
            assert np.allclose(np.linalg.norm(psi), 1.0), f"State {i} should be normalized"
            assert psi.shape == (2**n_qubits, 1), f"State {i} should have correct shape"

    def test_random_CHP_state_input_validation(self):
        """Test input validation for CHP states"""
        if not ff_rs.STIM_AVAILABLE:
            pytest.skip("stim package not available")

        # Test invalid types
        with pytest.raises(TypeError):
            ff_rs.random_CHP_state("2")

        # Test invalid values
        with pytest.raises(ValueError):
            ff_rs.random_CHP_state(0)


class TestRandomFFStates:
    """Test random free fermion state generation"""

    def test_random_FF_state_randH_basic(self):
        """Test basic properties of random FF states from random Hamiltonian"""
        n_sites = 2
        rho = ff_rs.random_FF_state_randH(n_sites, seed=42)

        # Check dimensions
        expected_dim = 2**n_sites
        assert rho.shape == (expected_dim, expected_dim), f"State should be {expected_dim}x{expected_dim}"

        # Check normalization
        assert np.allclose(np.trace(rho), 1.0), "State should be normalized"

        # Check that it's positive semidefinite
        eigenvals = np.linalg.eigvals(rho)
        assert np.all(eigenvals >= -1e-10), "State should be positive semidefinite"

        # Check that it's Hermitian
        assert np.allclose(rho, rho.conj().T), "State should be Hermitian"

    def test_random_FF_state_rotPDF_basic(self):
        """Test basic properties of random FF states from rotated PDF"""
        n_sites = 2
        rho = ff_rs.random_FF_state_rotPDF(n_sites, seed=42)

        # Check dimensions
        expected_dim = 2**n_sites
        assert rho.shape == (expected_dim, expected_dim), f"State should be {expected_dim}x{expected_dim}"

        # Check normalization
        assert np.allclose(np.trace(rho), 1.0), "State should be normalized"

        # Check that it's positive semidefinite
        eigenvals = np.linalg.eigvals(rho)
        assert np.all(eigenvals >= -1e-10), "State should be positive semidefinite"

        # Check that it's Hermitian
        assert np.allclose(rho, rho.conj().T), "State should be Hermitian"

    def test_random_FF_state_rotPDF_returnS(self):
        """Test random FF state with returnS=True"""
        n_sites = 2
        rho, s = ff_rs.random_FF_state_rotPDF(n_sites, returnS=True, seed=42)

        # Check that both are returned
        expected_dim = 2**n_sites
        assert rho.shape == (expected_dim, expected_dim), "State should be correct size"
        assert len(s) == expected_dim, "Probability distribution should have correct length"

        # Check that s is a valid probability distribution
        assert np.allclose(np.sum(s), 1.0), "Probability distribution should sum to 1"
        assert np.all(s >= 0), "Probabilities should be non-negative"

    def test_random_FF_pure_state_W0_basic(self):
        """Test basic properties of random FF pure states from vacuum rotation"""
        n_sites = 2
        psi = ff_rs.random_FF_pure_state_W0(n_sites, seed=42)

        # Check dimensions
        expected_dim = 2**n_sites
        assert psi.shape == (expected_dim, 1), f"State should be {expected_dim}x1"

        # Check normalization
        assert np.allclose(np.linalg.norm(psi), 1.0), "State should be normalized"

    def test_random_FF_pure_state_WN_basic(self):
        """Test basic properties of random FF pure states with fixed particle number"""
        n_sites = 3
        N = 2
        psi = ff_rs.random_FF_pure_state_WN(n_sites, N=N, seed=42)

        # Check dimensions
        expected_dim = 2**n_sites
        assert psi.shape == (expected_dim, 1), f"State should be {expected_dim}x1"

        # Check normalization
        assert np.allclose(np.linalg.norm(psi), 1.0), "State should be normalized"

    def test_random_FF_pure_state_WN_random_N(self):
        """Test random FF pure states with random particle number"""
        n_sites = 3
        psi = ff_rs.random_FF_pure_state_WN(n_sites, seed=42)

        # Check dimensions
        expected_dim = 2**n_sites
        assert psi.shape == (expected_dim, 1), f"State should be {expected_dim}x1"

        # Check normalization
        assert np.allclose(np.linalg.norm(psi), 1.0), "State should be normalized"

    def test_random_FF_pure_state_WN_input_validation(self):
        """Test input validation for WN states"""
        n_sites = 3

        # Test invalid N values
        with pytest.raises(ValueError):
            ff_rs.random_FF_pure_state_WN(n_sites, N=-1)

        with pytest.raises(ValueError):
            ff_rs.random_FF_pure_state_WN(n_sites, N=n_sites + 1)

        with pytest.raises(ValueError):
            ff_rs.random_FF_pure_state_WN(n_sites, N="invalid")

    def test_random_FF_pure_state_CN_basic(self):
        """Test basic properties of random FF pure states with orbital rotations"""
        n_sites = 2
        psi = ff_rs.random_FF_pure_state_CN(n_sites, seed=42)

        # Check dimensions
        expected_dim = 2**n_sites
        assert psi.shape == (expected_dim, 1), f"State should be {expected_dim}x1"

        # Check normalization
        assert np.allclose(np.linalg.norm(psi), 1.0), "State should be normalized"

    def test_random_FF_states_reproducibility(self):
        """Test that all random FF state functions are reproducible with seed"""
        n_sites = 2
        seed = 123

        # Test all random state functions
        rho1 = ff_rs.random_FF_state_randH(n_sites, seed=seed)
        rho2 = ff_rs.random_FF_state_randH(n_sites, seed=seed)
        assert np.allclose(rho1, rho2), "randH should be reproducible"

        rho3 = ff_rs.random_FF_state_rotPDF(n_sites, seed=seed)
        rho4 = ff_rs.random_FF_state_rotPDF(n_sites, seed=seed)
        assert np.allclose(rho3, rho4), "rotPDF should be reproducible"

        psi1 = ff_rs.random_FF_pure_state_W0(n_sites, seed=seed)
        psi2 = ff_rs.random_FF_pure_state_W0(n_sites, seed=seed)
        assert np.allclose(psi1, psi2), "W0 should be reproducible"

        psi3 = ff_rs.random_FF_pure_state_WN(n_sites, N=1, seed=seed)
        psi4 = ff_rs.random_FF_pure_state_WN(n_sites, N=1, seed=seed)
        assert np.allclose(psi3, psi4), "WN should be reproducible"

        psi5 = ff_rs.random_FF_pure_state_CN(n_sites, seed=seed)
        psi6 = ff_rs.random_FF_pure_state_CN(n_sites, seed=seed)
        assert np.allclose(psi5, psi6), "CN should be reproducible"

    def test_random_FF_states_input_validation(self):
        """Test input validation for all random FF state functions"""
        # Test invalid types
        with pytest.raises(TypeError):
            ff_rs.random_FF_state_randH("2")

        with pytest.raises(TypeError):
            ff_rs.random_FF_state_rotPDF(2.5)

        with pytest.raises(TypeError):
            ff_rs.random_FF_pure_state_W0([2])

        with pytest.raises(TypeError):
            ff_rs.random_FF_pure_state_WN(None)

        with pytest.raises(TypeError):
            ff_rs.random_FF_pure_state_CN({"n": 2})

        # Test invalid values
        with pytest.raises(ValueError):
            ff_rs.random_FF_state_randH(0)

        with pytest.raises(ValueError):
            ff_rs.random_FF_state_rotPDF(-1)

        with pytest.raises(ValueError):
            ff_rs.random_FF_pure_state_W0(0)

        with pytest.raises(ValueError):
            ff_rs.random_FF_pure_state_WN(-1)

        with pytest.raises(ValueError):
            ff_rs.random_FF_pure_state_CN(0)


class TestPathConstructionFunctions:
    """Test path construction and interpolation functions"""

    def test_get_orthogonal_vectors_basic(self):
        """Test basic properties of orthogonal vector generation"""
        v = np.array([[1], [2], [3]])
        orth_vecs = ff_rs.get_orthogonal_vectors(v)

        # Check dimensions
        n = v.shape[0]
        assert orth_vecs.shape == (n, n), f"Should return {n}x{n} matrix"

        # Check that first column is normalized input vector
        v_norm = v / np.linalg.norm(v)
        assert np.allclose(orth_vecs[:, 0:1], v_norm), "First column should be normalized input"

        # Check orthonormality
        assert np.allclose(orth_vecs @ orth_vecs.conj().T, np.eye(n)), "Should be orthonormal"

    def test_get_orthogonal_vectors_zero_vector(self):
        """Test orthogonal vector generation for zero vector"""
        v = np.array([[0], [0], [0]])
        orth_vecs = ff_rs.get_orthogonal_vectors(v)

        # Should return identity matrix for zero vector
        assert np.allclose(orth_vecs, np.eye(3)), "Should return identity for zero vector"

    def test_get_orthogonal_vectors_various_dimensions(self):
        """Test orthogonal vectors for various dimensions"""
        for n in [2, 3, 4, 5]:
            v = np.random.randn(n, 1)
            orth_vecs = ff_rs.get_orthogonal_vectors(v)

            assert orth_vecs.shape == (n, n), f"Should be {n}x{n} for dimension {n}"
            assert np.allclose(orth_vecs @ orth_vecs.conj().T, np.eye(n)), f"Should be orthonormal for dimension {n}"

    def test_build_linear_path_basic(self):
        """Test basic properties of linear path construction"""
        w = np.array([[1], [0], [0]])
        v = np.array([[0], [1], [0]])

        path = ff_rs.build_linear_path(w, v)

        # Test endpoints
        start = path(0.0)
        end = path(1.0)

        w_norm = w / np.linalg.norm(w)
        v_norm = v / np.linalg.norm(v)

        assert np.allclose(start, w_norm), "Path should start at normalized w"
        assert np.allclose(end, v_norm), "Path should end at normalized v"

        # Test that intermediate points are normalized
        mid = path(0.5)
        assert np.allclose(np.linalg.norm(mid), 1.0), "Intermediate points should be normalized"

    def test_build_unitary_path_basic(self):
        """Test basic properties of unitary path construction"""
        w = np.array([[1], [0], [0]])
        v = np.array([[0], [1], [0]])

        path = ff_rs.build_unitary_path(w, v)

        # Test endpoints
        start = path(0.0)
        end = path(1.0)

        w_norm = w / np.linalg.norm(w)
        v_norm = v / np.linalg.norm(v)

        assert np.allclose(start, w_norm, atol=1e-10), "Path should start at normalized w"
        assert np.allclose(end, v_norm, atol=1e-10), "Path should end at normalized v"

        # Test that intermediate points are normalized
        mid = path(0.5)
        assert np.allclose(np.linalg.norm(mid), 1.0), "Intermediate points should be normalized"

    def test_build_paths_various_vectors(self):
        """Test path construction for various vector pairs"""
        test_vectors = [
            (np.array([[1], [0]]), np.array([[0], [1]])),
            (np.array([[1], [1], [0]]), np.array([[0], [0], [1]])),
            (np.array([[1], [2], [3]]), np.array([[3], [2], [1]])),
        ]

        for w, v in test_vectors:
            # Test linear path
            linear_path = ff_rs.build_linear_path(w, v)
            start_linear = linear_path(0.0)
            end_linear = linear_path(1.0)

            w_norm = w / np.linalg.norm(w)
            v_norm = v / np.linalg.norm(v)

            assert np.allclose(start_linear, w_norm), f"Linear path should start correctly for {w.flatten()}"
            assert np.allclose(end_linear, v_norm), f"Linear path should end correctly for {v.flatten()}"

            # Test unitary path
            unitary_path = ff_rs.build_unitary_path(w, v)
            start_unitary = unitary_path(0.0)
            end_unitary = unitary_path(1.0)

            assert np.allclose(start_unitary, w_norm, atol=1e-10), f"Unitary path should start correctly for {w.flatten()}"
            assert np.allclose(end_unitary, v_norm, atol=1e-10), f"Unitary path should end correctly for {v.flatten()}"

    def test_path_normalization_preservation(self):
        """Test that paths preserve normalization throughout"""
        w = np.array([[1], [2], [3]])
        v = np.array([[3], [1], [2]])

        linear_path = ff_rs.build_linear_path(w, v)
        unitary_path = ff_rs.build_unitary_path(w, v)

        # Test normalization at various points
        test_points = [0.0, 0.25, 0.5, 0.75, 1.0]

        for t in test_points:
            linear_point = linear_path(t)
            unitary_point = unitary_path(t)

            assert np.allclose(np.linalg.norm(linear_point), 1.0), f"Linear path should be normalized at t={t}"
            assert np.allclose(np.linalg.norm(unitary_point), 1.0), f"Unitary path should be normalized at t={t}"


class TestIntegrationAndEdgeCases:
    """Test integration between functions and edge cases"""

    def test_small_system_sizes(self):
        """Test all functions work for small system sizes"""
        n_sites = 1

        # Test all random state functions for single site
        rho1 = ff_rs.random_FF_state_randH(n_sites, seed=42)
        assert rho1.shape == (2, 2), "Single site mixed state should be 2x2"

        rho2 = ff_rs.random_FF_state_rotPDF(n_sites, seed=42)
        assert rho2.shape == (2, 2), "Single site mixed state should be 2x2"

        psi1 = ff_rs.random_FF_pure_state_W0(n_sites, seed=42)
        assert psi1.shape == (2, 1), "Single site pure state should be 2x1"

        psi2 = ff_rs.random_FF_pure_state_WN(n_sites, N=0, seed=42)
        assert psi2.shape == (2, 1), "Single site pure state should be 2x1"

        psi3 = ff_rs.random_FF_pure_state_CN(n_sites, seed=42)
        assert psi3.shape == (2, 1), "Single site pure state should be 2x1"

    def test_consistency_between_methods(self):
        """Test that different methods produce valid quantum states"""
        n_sites = 2
        seed = 42

        # Generate states with different methods
        states = [
            ff_rs.random_FF_state_randH(n_sites, seed=seed),
            ff_rs.random_FF_state_rotPDF(n_sites, seed=seed),
        ]

        pure_states = [
            ff_rs.random_FF_pure_state_W0(n_sites, seed=seed),
            ff_rs.random_FF_pure_state_WN(n_sites, N=1, seed=seed),
            ff_rs.random_FF_pure_state_CN(n_sites, seed=seed),
        ]

        # Check that all mixed states are valid density matrices
        for i, rho in enumerate(states):
            assert np.allclose(np.trace(rho), 1.0), f"Mixed state {i} should be normalized"
            assert np.allclose(rho, rho.conj().T), f"Mixed state {i} should be Hermitian"
            eigenvals = np.linalg.eigvals(rho)
            assert np.all(eigenvals >= -1e-10), f"Mixed state {i} should be positive semidefinite"

        # Check that all pure states are normalized
        for i, psi in enumerate(pure_states):
            assert np.allclose(np.linalg.norm(psi), 1.0), f"Pure state {i} should be normalized"

    def test_path_construction_edge_cases(self):
        """Test path construction for edge cases"""
        # Test identical vectors
        v = np.array([[1], [0], [0]])
        linear_path = ff_rs.build_linear_path(v, v)
        unitary_path = ff_rs.build_unitary_path(v, v)

        v_norm = v / np.linalg.norm(v)

        # Path should be constant
        for t in [0.0, 0.5, 1.0]:
            assert np.allclose(linear_path(t), v_norm), "Linear path should be constant for identical vectors"
            assert np.allclose(unitary_path(t), v_norm, atol=1e-10), "Unitary path should be constant for identical vectors"

        # Test orthogonal vectors
        w = np.array([[1], [0]])
        v = np.array([[0], [1]])

        linear_path = ff_rs.build_linear_path(w, v)
        unitary_path = ff_rs.build_unitary_path(w, v)

        # Paths should interpolate correctly
        start_linear = linear_path(0.0)
        end_linear = linear_path(1.0)
        start_unitary = unitary_path(0.0)
        end_unitary = unitary_path(1.0)

        assert np.allclose(start_linear, w), "Linear path should start at w"
        assert np.allclose(end_linear, v), "Linear path should end at v"
        assert np.allclose(start_unitary, w, atol=1e-10), "Unitary path should start at w"
        assert np.allclose(end_unitary, v, atol=1e-10), "Unitary path should end at v"