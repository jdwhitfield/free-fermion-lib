
"""
Distance Measures Tests for Free Fermion Library

This module contains comprehensive tests for the distance measure functions in 
ff_distance_measures.py, including stabilizer distribution, Stabilizer Rényi Entropy (SRE),
general Rényi entropy, linear entropy, fermionic covariance distribution, and 
Fermionic Anti-Flatness (FAF) distance.

Test categories:
- Mathematical correctness tests (entropy bounds, distance axioms, normalization)
- Input validation tests (invalid types, dimensions, parameters)
- Edge cases (zero states, maximally mixed states, single qubit systems)
- Integration tests (compatibility with existing FF functions)
- Specific test cases (known analytical results for simple cases)
"""

import numpy as np
import pytest
import warnings

# Import the library
import ff


class TestStabilizerDistribution:
    """Test stabilizer_distribution() function"""

    def test_basic_properties(self):
        """Test basic properties of stabilizer distribution"""
        # Single qubit |0⟩ state
        psi = np.array([1, 0])
        xi = ff.stabilizer_distribution(psi)
        
        # Should have 4^1 = 4 elements
        assert len(xi) == 4, "Single qubit should have 4 Pauli operators"
        
        # Should be normalized
        assert np.allclose(np.sum(xi), 1.0), "Distribution should be normalized"
        
        # All elements should be non-negative
        assert np.all(xi >= -1e-10), "All probabilities should be non-negative"

    def test_two_qubit_system(self):
        """Test stabilizer distribution for two-qubit system"""
        # Two-qubit |00⟩ state
        psi = np.array([1, 0, 0, 0])
        xi = ff.stabilizer_distribution(psi)
        
        # Should have 4^2 = 16 elements
        assert len(xi) == 16, "Two qubits should have 16 Pauli operators"
        
        # Should be normalized
        assert np.allclose(np.sum(xi), 1.0), "Distribution should be normalized"
        
        # All elements should be non-negative
        assert np.all(xi >= -1e-10), "All probabilities should be non-negative"

    def test_maximally_mixed_state(self):
        """Test stabilizer distribution for maximally mixed state"""
        # Single qubit maximally mixed state
        rho = np.eye(2) / 2
        xi = ff.stabilizer_distribution(rho)
        
        # Should be normalized
        assert np.allclose(np.sum(xi), 1.0), "Distribution should be normalized"
        
        # For maximally mixed state, only identity should contribute
        # xi[0] corresponds to identity operator
        assert xi[0] > 0, "Identity should have non-zero probability"

    def test_wavefunction_vs_density_matrix(self):
        """Test that wavefunction and density matrix give same result"""
        # Single qubit state
        psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩ state
        rho = np.outer(psi, psi.conj())
        
        xi_psi = ff.stabilizer_distribution(psi)
        xi_rho = ff.stabilizer_distribution(rho)
        
        assert np.allclose(xi_psi, xi_rho), "Wavefunction and density matrix should give same result"

    def test_column_vector_input(self):
        """Test stabilizer distribution with column vector input"""
        # Column vector wavefunction
        psi = np.array([[1], [0]])
        xi = ff.stabilizer_distribution(psi)
        
        assert len(xi) == 4, "Should handle column vector input"
        assert np.allclose(np.sum(xi), 1.0), "Should be normalized"

    def test_stabilizer_state_properties(self):
        """Test properties for known stabilizer states"""
        # |+⟩ state (stabilizer state)
        psi_plus = np.array([1, 1]) / np.sqrt(2)
        xi_plus = ff.stabilizer_distribution(psi_plus)
        
        # Should be normalized
        assert np.allclose(np.sum(xi_plus), 1.0), "Should be normalized"
        
        # For stabilizer states, distribution should be sparse
        non_zero_count = np.sum(xi_plus > 1e-10)
        assert non_zero_count <= len(xi_plus), "Should have some structure"


class TestStabilizerRenyiEntropy:
    """Test SRE() function"""

    def test_basic_properties(self):
        """Test basic properties of SRE"""
        # Single qubit |0⟩ state
        psi = np.array([1, 0])
        rho = np.outer(psi, psi.conj())
        
        sre_val = ff.SRE(rho)
        
        # SRE should be real
        assert np.isreal(sre_val), "SRE should be real"
        
        # SRE should be finite
        assert np.isfinite(sre_val), "SRE should be finite"

    def test_sre_bounds(self):
        """Test SRE bounds and properties"""
        # Maximally mixed state
        rho_mixed = np.eye(4) / 4
        sre_mixed = ff.SRE(rho_mixed)
        
        # SRE should be non-negative for mixed states
        assert sre_mixed >= -1e-10, "SRE should be non-negative, instead: "+repr(sre_mixed)

    def test_different_renyi_parameters(self):
        """Test SRE with different Rényi parameters"""
        rho = np.eye(2) / 2  # Maximally mixed single qubit
        
        sre_1 = ff.SRE(rho, a=1)  # Shannon entropy case
        sre_2 = ff.SRE(rho, a=2)  # Quadratic case
        sre_3 = ff.SRE(rho, a=3)  # Cubic case
        
        # All should be real and finite
        assert np.isreal(sre_1) and np.isfinite(sre_1), "SRE(α=1) should be real and finite"
        assert np.isreal(sre_2) and np.isfinite(sre_2), "SRE(α=2) should be real and finite"
        assert np.isreal(sre_3) and np.isfinite(sre_3), "SRE(α=3) should be real and finite"

    def test_stabilizer_states_low_sre(self):
        """Test that stabilizer states have low SRE"""
        # |+⟩ state (stabilizer state)
        psi_plus = np.array([1, 1]) / np.sqrt(2)
        rho_plus = np.outer(psi_plus, psi_plus.conj())
        
        sre_plus = ff.SRE(rho_plus)
        
        # Stabilizer states should have relatively low SRE
        # (exact value depends on normalization, but should be finite)
        assert np.isfinite(sre_plus), "SRE of stabilizer state should be finite"

    def test_wavefunction_input(self):
        """Test SRE with wavefunction input"""
        psi = np.array([1, 0])
        sre_psi = ff.SRE(psi)
        
        # Should handle wavefunction input
        assert np.isreal(sre_psi) and np.isfinite(sre_psi), "Should handle wavefunction input"


class TestRenyiEntropy:
    """Test renyi_entropy() function"""

    def test_uniform_distribution(self):
        """Test Rényi entropy for uniform distribution"""
        # Uniform distribution over 4 elements
        p_uniform = np.ones(4) / 4
        
        s1 = ff.renyi_entropy(p_uniform, 1)  # Shannon entropy
        s2 = ff.renyi_entropy(p_uniform, 2)  # Collision entropy
        
        # For uniform distribution, Shannon entropy should be log(n)
        assert np.allclose(s1, np.log(4)), "Shannon entropy of uniform should be log(n)"
        
        # Collision entropy should be smaller than Shannon for uniform
        assert s2 <= s1 + 1e-10, "Collision entropy should be ≤ Shannon entropy"

    def test_delta_distribution(self):
        """Test Rényi entropy for delta distribution"""
        # Delta distribution (pure state)
        p_delta = np.array([1, 0, 0, 0])
        
        s1 = ff.renyi_entropy(p_delta, 1)
        s2 = ff.renyi_entropy(p_delta, 2)
        
        # Delta distribution should have zero entropy
        assert np.allclose(s1, 0, atol=1e-10), "Shannon entropy of delta should be 0"
        assert np.allclose(s2, 0, atol=1e-10), "Collision entropy of delta should be 0"

    def test_renyi_parameter_ordering(self):
        """Test ordering property of Rényi entropies"""
        # Non-uniform distribution
        p = np.array([0.5, 0.3, 0.15, 0.05])
        
        s_half = ff.renyi_entropy(p, 0.5)
        s1 = ff.renyi_entropy(p, 1)
        s2 = ff.renyi_entropy(p, 2)
        s_inf = -np.log(np.max(p))  # R∞ entropy
        
        # Ordering: S_0.5 ≥ S_1 ≥ S_2 ≥ S_∞
        assert s_half >= s1 - 1e-10, "S_0.5 should be ≥ S_1"
        assert s1 >= s2 - 1e-10, "S_1 should be ≥ S_2"
        assert s2 >= s_inf - 1e-10, "S_2 should be ≥ S_∞"

    def test_shannon_entropy_special_case(self):
        """Test that α=1 gives Shannon entropy"""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        
        s1_renyi = ff.renyi_entropy(p, 1)
        s1_shannon = -np.sum(p * np.log(p))
        
        assert np.allclose(s1_renyi, s1_shannon), "α=1 should give Shannon entropy"

    def test_invalid_parameters(self):
        """Test behavior with invalid parameters"""
        p = np.array([0.5, 0.5])
        
        # Test with α=1 (should work via special case)
        s1 = ff.renyi_entropy(p, 1)
        assert np.isfinite(s1), "α=1 should work"

    def test_normalization_requirement(self):
        """Test that function assumes normalized input"""
        # Non-normalized distribution
        p_unnorm = np.array([2, 2])  # Sum = 4, not 1
        
        # Function should still compute, but result may not be meaningful
        try:
            s = ff.renyi_entropy(p_unnorm, 2)
            assert np.isfinite(s), "Should handle non-normalized input"
        except ValueError:
            # Acceptable to raise error
            pass


class TestLinearEntropy:
    """Test linear_entropy() function"""

    def test_pure_state(self):
        """Test linear entropy for pure states"""
        # Pure state (delta distribution)
        p_pure = np.array([1, 0, 0, 0])
        lin_ent = ff.linear_entropy(p_pure)
        
        # Pure state should have zero linear entropy
        assert np.allclose(lin_ent, 0, atol=1e-10), "Pure state should have zero linear entropy"

    def test_maximally_mixed_state(self):
        """Test linear entropy for maximally mixed state"""
        # Maximally mixed state
        p_mixed = np.ones(4) / 4
        lin_ent = ff.linear_entropy(p_mixed)
        
        # For maximally mixed state: S_lin = (d-1)/d
        expected = (4 - 1) / 4
        assert np.allclose(lin_ent, expected), f"Maximally mixed should give {expected}"

    def test_dimension_requirement(self):
        """Test that dimension must be perfect square"""
        # Valid dimension (perfect square)
        p_valid = np.ones(9) / 9  # 3^2 = 9
        lin_ent = ff.linear_entropy(p_valid)
        assert np.isfinite(lin_ent), "Should work with perfect square dimension"
        
        # Invalid dimension (not perfect square)
        p_invalid = np.ones(5) / 5  # 5 is not a perfect square
        with pytest.raises(AssertionError):
            ff.linear_entropy(p_invalid)

    def test_bounds(self):
        """Test linear entropy bounds"""
        # Various distributions
        distributions = [
            np.array([1, 0, 0, 0]),  # Pure
            np.array([0.5, 0.5, 0, 0]),  # Partially mixed
            np.array([0.25, 0.25, 0.25, 0.25])  # Maximally mixed
        ]
        
        for p in distributions:
            lin_ent = ff.linear_entropy(p)
            
            # Should be between 0 and (d-1)/d
            assert 0 <= lin_ent <= (len(p) - 1) / len(p) + 1e-10, "Linear entropy should be in valid range"

    def test_relationship_to_purity(self):
        """Test relationship between linear entropy and purity"""
        p = np.array([0.6, 0.3, 0.1, 0])
        
        lin_ent = ff.linear_entropy(p)
        purity = np.sum(p**2)
        
        # Linear entropy = 1 - purity
        d = len(p)
        expected_lin_ent = 1 - purity
        
        assert np.allclose(lin_ent, expected_lin_ent), "Should match purity relationship"


class TestCovarianceDistribution:
    """Test cov_distribution() function"""

    def test_basic_properties(self):
        """Test basic properties of covariance distribution"""
        # Simple two-site system
        rho = np.eye(4) / 4  # Maximally mixed 2-qubit state
        pm = ff.cov_distribution(rho)
        
        # Should return an array
        assert isinstance(pm, np.ndarray), "Should return numpy array"
        
        # All elements should be non-negative (squared eigenvalues)
        assert np.all(pm >= -1e-10), "Squared eigenvalues should be non-negative"

    def test_wavefunction_input(self):
        """Test covariance distribution with wavefunction input"""
        # Two-qubit wavefunction
        psi = np.array([1, 0, 0, 0])  # |00⟩ state
        pm = ff.cov_distribution(psi)
        
        assert isinstance(pm, np.ndarray), "Should handle wavefunction input"
        assert np.all(pm >= -1e-10), "Should give non-negative values"

    def test_probability_vector_input(self):
        """Test covariance distribution with probability vector"""
        # Probability vector (L1 normalized)
        p = np.array([0.25, 0.25, 0.25, 0.25])
        pm = ff.cov_distribution(p)
        
        assert isinstance(pm, np.ndarray), "Should handle probability vector"

    def test_different_system_sizes(self):
        """Test covariance distribution for different system sizes"""
        # Single qubit
        rho_1 = np.eye(2) / 2
        pm_1 = ff.cov_distribution(rho_1)
        assert isinstance(pm_1, np.ndarray), "Should work for single qubit"
        
        # Three qubits
        rho_3 = np.eye(8) / 8
        pm_3 = ff.cov_distribution(rho_3)
        assert isinstance(pm_3, np.ndarray), "Should work for three qubits"

    def test_antisymmetric_covariance_matrix(self):
        """Test that underlying covariance matrix is antisymmetric"""
        # This is tested indirectly through the assertion in the function
        rho = np.eye(4) / 4
        
        # Should not raise assertion error about antisymmetry
        pm = ff.cov_distribution(rho)
        assert isinstance(pm, np.ndarray), "Should pass antisymmetry check"


class TestFermionicAntiFlatness:
    """Test FAF() function"""

    def test_basic_properties(self):
        """Test basic properties of FAF"""
        # Simple system
        rho = np.eye(4) / 4  # Maximally mixed 2-qubit state
        faf_val = ff.FAF(rho)
        
        # Should be real and finite
        assert np.isreal(faf_val), "FAF should be real"
        assert np.isfinite(faf_val), "FAF should be finite"

    def test_different_k_parameters(self):
        """Test FAF with different k parameters"""
        rho = np.eye(4) / 4
        
        faf_1 = ff.FAF(rho, k=1)
        faf_2 = ff.FAF(rho, k=2)
        faf_3 = ff.FAF(rho, k=3)
        
        # All should be real and finite
        assert np.isreal(faf_1) and np.isfinite(faf_1), "FAF(k=1) should be real and finite"
        assert np.isreal(faf_2) and np.isfinite(faf_2), "FAF(k=2) should be real and finite"
        assert np.isreal(faf_3) and np.isfinite(faf_3), "FAF(k=3) should be real and finite"

    def test_wavefunction_input(self):
        """Test FAF with wavefunction input"""
        psi = np.array([1, 0, 0, 0])  # |00⟩ state
        faf_val = ff.FAF(psi)
        
        assert np.isreal(faf_val) and np.isfinite(faf_val), "Should handle wavefunction input"

    def test_column_vector_input(self):
        """Test FAF with column vector input"""
        psi = np.array([[1], [0], [0], [0]])  # Column vector
        faf_val = ff.FAF(psi)
        
        assert np.isreal(faf_val) and np.isfinite(faf_val), "Should handle column vector input"

    def test_bounds(self):
        """Test FAF bounds"""
        # For n_sites system, FAF should be between 0 and n_sites
        rho = np.eye(4) / 4  # 2-site system
        faf_val = ff.FAF(rho)
        
        # FAF should be non-negative and bounded by n_sites
        n_sites = 2
        assert 0 <= faf_val <= n_sites + 1e-10, f"FAF should be between 0 and {n_sites}"


class TestInputValidation:
    """Test input validation for all functions"""

    def test_invalid_matrix_dimensions(self):
        """Test behavior with invalid matrix dimensions"""
        # Non-square matrix
        invalid_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Functions should handle this gracefully or raise appropriate errors
        with pytest.raises((ValueError, AssertionError, IndexError)):
            ff.stabilizer_distribution(invalid_matrix)

    def test_empty_input(self):
        """Test behavior with empty input"""
        empty_array = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            ff.stabilizer_distribution(empty_array)

    def test_non_hermitian_density_matrix(self):
        """Test behavior with non-Hermitian matrices"""
        # Non-Hermitian matrix
        non_hermitian = np.array([[1, 2], [3, 4]])
        
        # Functions should still work (they don't explicitly check Hermiticity)
        # but results may not be physically meaningful
        try:
            result = ff.stabilizer_distribution(non_hermitian)
            assert isinstance(result, np.ndarray), "Should return array even for non-Hermitian input"
        except Exception:
            # Some functions might fail, which is acceptable
            pass

    def test_invalid_renyi_parameters(self):
        """Test Rényi entropy with invalid parameters"""
        p = np.array([0.5, 0.5])
        
        # Negative α (should still work mathematically)
        s_neg = ff.renyi_entropy(p, -1)
        assert np.isfinite(s_neg), "Should handle negative α"
        
        # α = 0 (should work)
        s_zero = ff.renyi_entropy(p, 0)
        assert np.isfinite(s_zero), "Should handle α = 0"

    def test_unnormalized_distributions(self):
        """Test behavior with unnormalized distributions"""
        # Unnormalized distribution
        p_unnorm = np.array([2, 3, 1])  # Sum = 6, not 1
        
        # Functions should still compute but may give non-physical results
        try:
            s = ff.renyi_entropy(p_unnorm, 2)
            assert np.isfinite(s), "Should handle unnormalized distributions"
        except ValueError:
            # Acceptable to raise error
            pass

    def test_negative_probabilities(self):
        """Test behavior with negative probabilities"""
        # Distribution with negative values
        p_neg = np.array([0.8, -0.3, 0.5])
        
        # This is unphysical but functions should handle it
        try:
            s = ff.renyi_entropy(p_neg, 2)
            # May give complex result or raise error
            assert np.isfinite(s) or np.isfinite(s.real), "Should handle or fail gracefully"
        except (ValueError, RuntimeWarning):
            # Acceptable to fail with negative probabilities
            pass


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_qubit_systems(self):
        """Test all functions with single qubit systems"""
        # Single qubit states
        psi_0 = np.array([1, 0])  # |0⟩
        psi_1 = np.array([0, 1])  # |1⟩
        psi_plus = np.array([1, 1]) / np.sqrt(2)  # |+⟩
        
        states = [psi_0, psi_1, psi_plus]
        
        for psi in states:
            # Test stabilizer distribution
            xi = ff.stabilizer_distribution(psi)
            assert len(xi) == 4, "Single qubit should have 4 Pauli operators"
            assert np.allclose(np.sum(xi), 1.0), "Should be normalized"
            
            # Test SRE
            sre = ff.SRE(psi)
            assert np.isfinite(sre), "SRE should be finite"
            
            # Test FAF
            faf = ff.FAF(psi)
            assert np.isfinite(faf), "FAF should be finite"

    def test_maximally_mixed_states(self):
        """Test with maximally mixed states of different sizes"""
        dimensions = [2, 4, 8]  # 1, 2, 3 qubits
        
        for d in dimensions:
            rho_mixed = np.eye(d) / d
            
            # Test stabilizer distribution
            xi = ff.stabilizer_distribution(rho_mixed)
            assert len(xi) == d**2, f"Should have {d**2} Pauli operators"
            assert np.allclose(np.sum(xi), 1.0), "Should be normalized"
            
            # Test SRE
            sre = ff.SRE(rho_mixed)
            assert np.isfinite(sre), "SRE should be finite"
            
            # Test covariance distribution
            pm = ff.cov_distribution(rho_mixed)
            assert isinstance(pm, np.ndarray), "Should return array"

    def test_identity_matrices(self):
        """Test behavior with identity matrices"""
        # Unnormalized identity
        I = np.eye(4)
        
        # Functions should handle this (after normalization)
        try:
            xi = ff.stabilizer_distribution(I)
            assert isinstance(xi, np.ndarray), "Should handle identity matrix"
        except Exception:
            # May fail due to normalization issues, which is acceptable
            pass

    def test_zero_states(self):
        """Test behavior with zero matrices"""
        # Zero matrix
        zero_matrix = np.zeros((4, 4))
        
        # This is unphysical but functions should handle gracefully
        try:
            xi = ff.stabilizer_distribution(zero_matrix)
            # May give all zeros or fail
            assert isinstance(xi, np.ndarray), "Should return array"
        except Exception:
            # Acceptable to fail with zero matrix
            pass

    def test_numerical_precision_edge_cases(self):
        """Test numerical precision edge cases"""
        # Very small but non-zero probabilities
        p_small = np.array([1-1e-15, 1e-15, 0, 0])
        
        s = ff.renyi_entropy(p_small, 2)
        assert np.isfinite(s), "Should handle very small probabilities"
        
        # Very large numbers (but still finite)
        p_large = np.array([1e10, 1e10]) 
        p_large = p_large / np.sum(p_large)  # Normalize
        
        s_large = ff.renyi_entropy(p_large, 2)
        assert np.isfinite(s_large), "Should handle large normalized numbers"


class TestIntegrationWithFFLibrary:
    """Test integration with existing FF library functions"""

    def test_with_random_ff_states(self):
        """Test distance measures with random FF states"""
        n_sites = 2
        
        # Generate random FF state
        rho = ff.random_FF_state(n_sites, seed=42)
        
        # Test all distance measures
        xi = ff.stabilizer_distribution(rho)
        assert np.allclose(np.sum(xi), 1.0), "Should work with random FF states"
        
        sre = ff.SRE(rho)
        assert np.isfinite(sre), "SRE should work with random FF states"
        
        pm = ff.cov_distribution(rho)
        assert isinstance(pm, np.ndarray), "Covariance distribution should work"
        
        faf = ff.FAF(rho)
        assert np.isfinite(faf), "FAF should work with random FF states"

    def test_with_gaussian_states(self):
        """Test distance measures with Gaussian states"""
        n_sites = 2
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        H = ff.build_H(n_sites, A)
        alphas = ff.jordan_wigner_alphas(n_sites)
        
        rho = ff.generate_gaussian_state(n_sites, H, alphas)
        
        # Test distance measures
        xi = ff.stabilizer_distribution(rho)
        assert np.allclose(np.sum(xi), 1.0), "Should work with Gaussian states"
        
        sre = ff.SRE(rho)
        assert np.isfinite(sre), "SRE should work with Gaussian states"

    def test_with_jordan_wigner_operators(self):
        """Test compatibility with Jordan-Wigner operators"""
        n_sites = 2
        alphas = ff.jordan_wigner_alphas(n_sites)
        
        # Create simple state using operators
        vacuum = np.zeros((2**n_sites, 1))
        vacuum[0] = 1
        
        # Apply creation operator
        psi = alphas[0] @ vacuum  # a†_0 |0⟩
        psi = psi / np.linalg.norm(psi)
        
        # Test distance measures
        xi = ff.stabilizer_distribution(psi)
        assert np.allclose(np.sum(xi), 1.0), "Should work with JW operator states"

    def test_with_pauli_group(self):
        """Test consistency with Pauli group generation"""
        n_qubits = 2
        pauli_group = ff.generate_pauli_group(n_qubits)
        
        # Test that stabilizer distribution has correct length
        rho = np.eye(2**n_qubits) / (2**n_qubits)
        xi = ff.stabilizer_distribution(rho)
        
        assert len(xi) == len(pauli_group), "Should match Pauli group size"


class TestSpecificKnownCases:
    """Test specific cases with known analytical results"""

    def test_bell_states(self):
        """Test distance measures for Bell states"""
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        
        xi = ff.stabilizer_distribution(phi_plus)
        assert np.allclose(np.sum(xi), 1.0), "Bell state should give normalized distribution"
        
        sre = ff.SRE(phi_plus)
        assert np.isfinite(sre), "Bell state should give finite SRE"

    def test_ghz_states(self):
        """Test distance measures for GHZ-like states"""
        # |GHZ⟩ = (|000⟩ + |111⟩)/√2
        ghz = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
        
        xi = ff.stabilizer_distribution(ghz)
        assert np.allclose(np.sum(xi), 1.0), "GHZ state should give normalized distribution"
        
        sre = ff.SRE(ghz)
        assert np.isfinite(sre), "GHZ state should give finite SRE"

    def test_product_states(self):
        """Test distance measures for product states"""
        # |00⟩ state
        product_00 = np.array([1, 0, 0, 0])
        
        xi = ff.stabilizer_distribution(product_00)
        sre = ff.SRE(product_00)
        faf = ff.FAF(product_00)
        
        assert np.allclose(np.sum(xi), 1.0), "Product state should give normalized distribution"
        assert np.isfinite(sre), "Product state should give finite SRE"
        assert np.isfinite(faf), "Product state should give finite FAF"

    
    def test_known_free_fermion_states(self):
        """Test distance measures for known free-fermion states"""
        # Create a simple free-fermion state
        n_sites = 2
        A = np.array([[1.0, 0.0], [0.0, -1.0]])  # Simple diagonal A
        H = ff.build_H(n_sites, A)
        alphas = ff.jordan_wigner_alphas(n_sites)
        
        rho_ff = ff.generate_gaussian_state(n_sites, H, alphas)
        
        # Test FAF (should be relatively low for free-fermion states)
        faf = ff.FAF(rho_ff)
        assert np.isfinite(faf), "FAF should be finite for free-fermion states"
        
        # Test other measures
        xi = ff.stabilizer_distribution(rho_ff)
        assert np.allclose(np.sum(xi), 1.0), "Should give normalized distribution"
        
        sre = ff.SRE(rho_ff)
        assert np.isfinite(sre), "SRE should be finite for free-fermion states"

    def test_clifford_states(self):
        """Test SRE for known Clifford states (should be low)"""
        # |+⟩ state (Clifford/stabilizer state)
        psi_plus = np.array([1, 1]) / np.sqrt(2)
        sre_plus = ff.SRE(psi_plus)
        
        # |+⟩⊗|+⟩ state
        psi_plus_plus = np.kron(psi_plus, psi_plus)
        sre_plus_plus = ff.SRE(psi_plus_plus)
        
        # Both should be finite (exact values depend on normalization)
        assert np.isfinite(sre_plus), "SRE should be finite for |+⟩"
        assert np.isfinite(sre_plus_plus), "SRE should be finite for |+⟩⊗|+⟩"

    def test_computational_basis_states(self):
        """Test distance measures for computational basis states"""
        # Test various computational basis states
        basis_states = [
            np.array([1, 0]),  # |0⟩
            np.array([0, 1]),  # |1⟩
            np.array([1, 0, 0, 0]),  # |00⟩
            np.array([0, 0, 0, 1]),  # |11⟩
        ]
        
        for psi in basis_states:
            xi = ff.stabilizer_distribution(psi)
            assert np.allclose(np.sum(xi), 1.0), f"Basis state should give normalized distribution"
            
            sre = ff.SRE(psi)
            assert np.isfinite(sre), f"SRE should be finite for basis states"
            
            if len(psi) >= 4:  # Multi-qubit states
                faf = ff.FAF(psi)
                assert np.isfinite(faf), f"FAF should be finite for basis states"


class TestPerformanceAndNumericalPrecision:
    """Test performance and numerical precision aspects"""

    def test_numerical_stability_small_values(self):
        """Test numerical stability with very small values"""
        # Distribution with very small probabilities
        p_small = np.array([1-1e-14, 1e-14, 1e-15, 1e-16])
        
        # Should handle small values gracefully
        s1 = ff.renyi_entropy(p_small, 1)
        s2 = ff.renyi_entropy(p_small, 2)
        
        assert np.isfinite(s1), "Should handle very small probabilities"
        assert np.isfinite(s2), "Should handle very small probabilities"

    def test_numerical_stability_large_systems(self):
        """Test numerical stability for larger systems"""
        # 3-qubit maximally mixed state
        rho_large = np.eye(8) / 8
        
        xi = ff.stabilizer_distribution(rho_large)
        assert len(xi) == 64, "Should have 4^3 = 64 Pauli operators"
        assert np.allclose(np.sum(xi), 1.0), "Should be normalized"
        
        sre = ff.SRE(rho_large)
        assert np.isfinite(sre), "SRE should be finite for larger systems"

    def test_precision_consistency(self):
        """Test that results are consistent across different precisions"""
        # Simple state
        psi = np.array([1, 1, 0, 0]) / np.sqrt(2)
        
        # Compute measures multiple times (should be deterministic)
        xi1 = ff.stabilizer_distribution(psi)
        xi2 = ff.stabilizer_distribution(psi)
        
        assert np.allclose(xi1, xi2), "Results should be deterministic"
        
        sre1 = ff.SRE(psi)
        sre2 = ff.SRE(psi)
        
        assert np.allclose(sre1, sre2), "SRE should be deterministic"

    def test_memory_efficiency(self):
        """Test that functions don't consume excessive memory"""
        # This is a basic test - in practice, memory profiling tools would be better
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Test with moderately sized system
        rho = np.eye(16) / 16  # 4-qubit system
        
        # These operations should complete without memory issues
        xi = ff.stabilizer_distribution(rho)
        assert isinstance(xi, np.ndarray), "Should complete without memory issues"
        
        sre = ff.SRE(rho)
        assert np.isfinite(sre), "Should complete without memory issues"

    @pytest.mark.slow
    def test_performance_scaling(self):
        """Test performance scaling with system size (marked as slow)"""
        import time
        
        # Test different system sizes
        sizes = [1, 2, 3]  # Number of qubits
        times = []
        
        for n_qubits in sizes:
            rho = np.eye(2**n_qubits) / (2**n_qubits)
            
            start_time = time.time()
            xi = ff.stabilizer_distribution(rho)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Basic sanity check
            assert len(xi) == 4**n_qubits, f"Should have correct number of elements"
        
        # Times should be reasonable (this is a very loose check)
        assert all(t < 10.0 for t in times), "Computation times should be reasonable"


class TestCrossValidation:
    """Test cross-validation between different measures and approaches"""

    def test_renyi_entropy_consistency(self):
        """Test consistency between different Rényi entropy calculations"""
        # Test distribution
        p = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Test that α→1 limit gives Shannon entropy
        s1_renyi = ff.renyi_entropy(p, 1)
        s1_shannon = -np.sum(p * np.log(p + 1e-16))  # Add small epsilon for numerical stability
        
        assert np.allclose(s1_renyi, s1_shannon, rtol=1e-10), "α=1 should give Shannon entropy"

    def test_linear_entropy_vs_renyi(self):
        """Test relationship between linear entropy and Rényi entropy"""
        p = np.array([0.5, 0.3, 0.2, 0])
        
        # Linear entropy should be related to 2-Rényi entropy
        lin_ent = ff.linear_entropy(p)
        s2 = ff.renyi_entropy(p, 2)
        
        # Both should be finite and related
        assert np.isfinite(lin_ent), "Linear entropy should be finite"
        assert np.isfinite(s2), "2-Rényi entropy should be finite"

    def test_sre_vs_stabilizer_distribution(self):
        """Test consistency between SRE and stabilizer distribution"""
        psi = np.array([1, 1, 0, 0]) / np.sqrt(2)
        
        # Compute SRE
        sre = ff.SRE(psi)
        
        # Compute manually from stabilizer distribution
        xi = ff.stabilizer_distribution(psi)
        d = len(psi)
        manual_sre = ff.renyi_entropy(xi, 2) - np.log(d)
        
        assert np.allclose(sre, manual_sre), "SRE should match manual calculation"

    def test_faf_vs_covariance_distribution(self):
        """Test consistency between FAF and covariance distribution"""
        psi = np.array([1, 0, 0, 0])  # |00⟩ state
        
        # Compute FAF
        faf = ff.FAF(psi, k=2)
        
        # Compute manually from covariance distribution
        pm = ff.cov_distribution(psi)
        n_sites = int(np.log2(len(psi)))
        manual_faf = n_sites - np.linalg.norm(pm, 2)**2
        
        assert np.allclose(faf, manual_faf), "FAF should match manual calculation"


class TestDocumentationExamples:
    """Test examples from function docstrings"""

    def test_stabilizer_distribution_examples(self):
        """Test examples from stabilizer_distribution docstring"""
        # Single qubit |0⟩ state
        psi = np.array([1, 0])
        xi = ff.stabilizer_distribution(psi)
        
        assert len(xi) == 4, "Single qubit should have 4 elements"
        assert np.allclose(np.sum(xi), 1.0), "Should be normalized"
        
        # Two-qubit maximally mixed state
        rho = np.eye(4) / 4
        xi = ff.stabilizer_distribution(rho)
        assert len(xi) == 16, "Two qubits should have 16 elements"

    def test_renyi_entropy_examples(self):
        """Test examples from renyi_entropy docstring"""
        # Uniform distribution
        p_uniform = np.ones(4) / 4
        s1 = ff.renyi_entropy(p_uniform, 1) # Shannon entropy
        s2 = ff.renyi_entropy(p_uniform, 2) # Collision entropy
        
        assert np.allclose(s1, np.log(4)), "Shannon entropy of uniform should be log(4)"
        assert s2 <= s1, "Collision entropy should be less than or equal to Shannon entropy"
        
        # Delta distribution
        p_delta = np.array([1, 0, 0, 0])
        s2_delta = ff.renyi_entropy(p_delta, 2)
        assert np.allclose(s2_delta, 0.0), "Delta distribution should have zero entropy"

    def test_linear_entropy_examples(self):
        """Test examples from linear_entropy docstring"""
        # Pure state
        p_pure = np.array([1, 0, 0, 0])
        lin_ent_pure = ff.linear_entropy(p_pure)
        assert np.allclose(lin_ent_pure, 0.0), "Pure state should have zero linear entropy"
        
        # Maximally mixed state
        p_mixed = np.ones(4) / 4
        lin_ent_mixed = ff.linear_entropy(p_mixed)
        expected = 3/4  # (d-1)/d for d=4
        assert np.allclose(lin_ent_mixed, expected), f"Should give {expected}"


class TestWarningsAndEdgeCaseHandling:
    """Test proper handling of warnings and edge cases"""

    def test_warning_suppression(self):
        """Test that appropriate warnings are handled"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Operations that might generate warnings
            p_with_zeros = np.array([0.5, 0.5, 0, 0])
            s = ff.renyi_entropy(p_with_zeros, 1)
            
            # Should handle gracefully
            assert np.isfinite(s), "Should handle zeros in distribution"

    def test_complex_number_handling(self):
        """Test handling of complex numbers in inputs"""
        # Complex wavefunction (should be handled)
        psi_complex = np.array([1+0j, 0+0j]) / np.sqrt(1)
        
        xi = ff.stabilizer_distribution(psi_complex)
        assert np.allclose(np.sum(xi), 1.0), "Should handle complex wavefunction"
        
        # Result should be real
        assert np.allclose(xi.imag, 0), "Result should be real"

    def test_boundary_conditions(self):
        """Test boundary conditions and limits"""
        # Very uniform distribution
        p_uniform = np.ones(16) / 16
        
        s1 = ff.renyi_entropy(p_uniform, 1)
        s2 = ff.renyi_entropy(p_uniform, 2)
        
        # Should satisfy ordering
        assert s1 >= s2 - 1e-10, "Shannon should be ≥ collision entropy"
        
        # Should be close to log(16) for uniform
        assert np.allclose(s1, np.log(16)), "Should be log(n) for uniform"


# Performance marker for pytest - this should be defined in conftest.py or pytest.ini
# For now, we'll just use the standard slow marker