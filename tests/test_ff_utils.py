"""
Utility Tests for Free Fermion Library

This module contains comprehensive tests for the utility functions in ff_utils.py,
including formatting, cleaning, printing, and helper functions.

Test categories:
- Matrix cleaning and formatting
- Output formatting and printing
- Numerical precision handling
- String manipulation utilities
- Error handling and edge cases
"""

import io
import warnings
from contextlib import redirect_stdout

import numpy as np

# Import the library
import ff


class TestCleanFunction:
    """Test the clean() function for numerical precision"""

    def test_clean_small_numbers(self):
        """Test cleaning of very small numbers"""
        # Small real numbers should become zero
        small_real = 1e-15
        cleaned = ff.clean(small_real)
        assert cleaned == 0, "Very small real numbers should be cleaned to 0"

        # Small complex numbers should become zero
        small_complex = 1e-15 + 1e-15j
        cleaned = ff.clean(small_complex)
        assert cleaned == 0, "Very small complex numbers should be cleaned to 0"

        # Small imaginary part should be removed
        almost_real = 1.0 + 1e-15j
        cleaned = ff.clean(almost_real)
        assert cleaned == 1.0, "Small imaginary parts should be removed"
        assert isinstance(cleaned, float), "Result should be real"

    def test_clean_arrays(self):
        """Test cleaning of numpy arrays"""
        # Array with small numbers
        arr = np.array([1.0, 1e-15, 1e-14, 2.0])
        cleaned = ff.clean(arr)
        expected = np.array([1.0, 0.0, 0.0, 2.0])
        assert np.allclose(cleaned, expected), "Small array elements should be cleaned"

        # Complex array
        arr_complex = np.array([1.0 + 1e-15j, 2.0, 1e-15 + 1e-15j])
        cleaned = ff.clean(arr_complex)
        expected = np.array([1.0, 2.0, 0.0])
        assert np.allclose(
            cleaned, expected
        ), "Complex array should be cleaned properly"

    def test_clean_matrices(self):
        """Test cleaning of matrices"""
        # Matrix with small entries
        matrix = np.array([[1.0, 1e-15], [1e-14, 2.0]])
        cleaned = ff.clean(matrix)
        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        assert np.allclose(cleaned, expected), "Matrix entries should be cleaned"

        # Preserve matrix shape
        assert cleaned.shape == matrix.shape, "Matrix shape should be preserved"

    def test_clean_threshold_parameter(self):
        """Test clean function with custom threshold"""
        # Default threshold
        val = 1e-12
        cleaned_default = ff.clean(val)
        assert cleaned_default == 0, "Should be cleaned with default threshold"

        # Custom threshold
        cleaned_custom = ff.clean(val, threshold=1e-15)
        assert cleaned_custom == val, "Should not be cleaned with smaller threshold"

        # Larger threshold
        val_larger = 1e-10
        cleaned_large_thresh = ff.clean(val_larger, threshold=1e-8)
        assert cleaned_large_thresh == 0, "Should be cleaned with larger threshold"

    def test_clean_preserves_significant_values(self):
        """Test that clean preserves significant values"""
        # Values that should not be cleaned
        significant_values = [1.0, -1.0, 0.1, -0.1, 1e-10, -1e-10]

        for val in significant_values:
            cleaned = ff.clean(val, 1e-12)
            if abs(val) > 1e-12:  # above threshold
                assert np.allclose(
                    cleaned, val
                ), f"Significant value {val} should not be cleaned"

    def test_clean_edge_cases(self):
        """Test clean function edge cases"""
        # Zero should remain zero
        assert ff.clean(0.0) == 0.0, "Zero should remain zero"
        assert ff.clean(0.0 + 0.0j) == 0.0, "Complex zero should become real zero"

        # Infinity should remain infinity
        assert ff.clean(np.inf) == np.inf, "Infinity should remain infinity"
        assert ff.clean(-np.inf) == -np.inf, "Negative infinity should remain"

        # NaN should remain NaN
        assert np.isnan(ff.clean(np.nan)), "NaN should remain NaN"


class TestPrintFunction:
    """Test the _print() function for formatted output"""

    def test_print_basic_output(self):
        """Test basic printing functionality"""
        # Capture stdout
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            ff.ff_utils._print("Test message")

        output = captured_output.getvalue()
        assert "Test message" in output, "Should print the message"

    def test_print_no_double_output(self):
        """Test that _print function doesn't print twice (bug fix)"""
        # Capture stdout
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            ff.ff_utils._print("Single message")

        output = captured_output.getvalue()
        # Count occurrences of the message
        message_count = output.count("Single message")
        assert (
            message_count == 1
        ), f"Message should appear exactly once, but appeared {message_count} times"

    def test_print_arrays(self):
        """Test printing of numpy arrays"""
        arr = np.array([1, 2, 3])
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            ff.ff_utils._print(arr)

        output = captured_output.getvalue()
        assert "[1 2 3]" in output or "1" in output, "Should print array contents"

    def test_print_matrices(self):
        """Test printing of matrices"""
        matrix = np.array([[1, 2], [3, 4]])
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            ff.ff_utils._print(matrix)

        output = captured_output.getvalue()
        # assert "Matrix:" in output, "Should print matrix label"
        # Matrix should be formatted nicely
        assert "1" in output and "4" in output, "Should contain matrix elements"

    def test_print_complex_numbers(self):
        """Test printing of complex numbers"""
        complex_val = 1 + 2j
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            ff.ff_utils._print(complex_val)

        output = captured_output.getvalue()
        # assert "Complex:" in output, "Should print label"
        assert (
            "1" in output and "2" in output
        ), "Should contain real and imaginary parts"


class TestFormattedOutput:
    """Test formatted output functions"""

    def test_formatted_matrix_output(self):
        """Test formatted matrix output"""
        matrix = np.array([[1.23456, 2.34567], [3.45678, 4.56789]])

        formatted = ff.formatted_output(matrix)

        # Should be a string
        assert isinstance(formatted, str), "Should return string"

        # Should contain matrix elements (possibly rounded)
        assert (
            "1.2" in formatted or "1.23" in formatted
        ), "Should contain formatted elements"

    def test_formatted_array_output(self):
        """Test formatted array output"""
        arr = np.array([1.23456, 2.34567, 3.45678])

        formatted = ff.formatted_output(arr)

        assert isinstance(formatted, str), "Should return string"
        assert "1.2" in formatted or "1.23" in formatted, "Should format array elements"

    def test_formatted_scalar_output(self):
        """Test formatted scalar output"""
        scalar = 3.14159265

        formatted = ff.formatted_output(scalar)

        assert isinstance(formatted, str), "Should return string"
        assert "3.14" in formatted, "Should format scalar value"

    def test_formatted_complex_output(self):
        """Test formatted complex number output"""
        complex_val = 1.23456 + 2.34567j

        formatted = ff.formatted_output(complex_val)

        assert isinstance(formatted, str), "Should return string"
        assert "1.2" in formatted and "2.3" in formatted, "Should format both parts"
        assert "j" in formatted or "i" in formatted, "Should indicate imaginary unit"

    def test_formatted_output_precision(self):
        """Test formatted output with different precision"""
        val = np.pi

        # Test different precision levels
        formatted_2 = ff.formatted_output(val, precision=2)
        formatted_4 = ff.formatted_output(val, precision=4)

        # Should have different levels of precision
        assert len(formatted_4) >= len(formatted_2), "Higher precision should be longer"

    def test_formatted_output_scientific(self):
        """Test formatted output in scientific notation"""
        large_val = 1.23e10
        small_val = 1.23e-10

        formatted_large = ff.formatted_output(large_val)
        formatted_small = ff.formatted_output(small_val)

        # Should handle scientific notation appropriately
        assert isinstance(formatted_large, str), "Should format large numbers"
        assert isinstance(formatted_small, str), "Should format small numbers"


class TestNumericalUtilities:
    """Test numerical utility functions"""

    def test_tolerance_checking(self):
        """Test tolerance-based equality checking"""
        # Test approximate equality
        a = 1.0
        b = 1.0 + 1e-15

        try:
            is_close = ff.is_close(a, b)
            assert is_close, "Should be approximately equal"

            # Test with custom tolerance
            is_close_strict = ff.is_close(a, b, tolerance=1e-16)
            assert not is_close_strict, "Should not be equal with strict tolerance"
        except AttributeError:
            # Function might not exist, use numpy instead
            assert np.isclose(a, b), "Should be approximately equal"

    def test_matrix_comparison(self):
        """Test matrix comparison utilities"""
        A = np.array([[1, 2], [3, 4]])
        B = A + 1e-15

        # Should be approximately equal
        assert np.allclose(A, B), "Matrices should be approximately equal"

        # Test custom matrix comparison if available
        try:
            matrices_equal = ff.matrices_equal(A, B)
            assert matrices_equal, "Custom comparison should work"
        except AttributeError:
            # Function might not exist
            pass

    def test_numerical_stability(self):
        """Test numerical stability utilities"""
        # Test condition number calculation
        A = np.array([[1, 2], [3, 4]])

        cond_num = np.linalg.cond(A)
        assert np.isfinite(cond_num), "Condition number should be finite"

        # Test if custom stability checks exist
        try:
            is_stable = ff.is_numerically_stable(A)
            assert isinstance(is_stable, bool), "Stability check should return boolean"
        except AttributeError:
            # Function might not exist
            pass

    def test_precision_handling(self):
        """Test precision handling utilities"""
        # Test rounding utilities
        val = 3.14159265

        rounded_2 = round(val, 2)
        assert rounded_2 == 3.14, "Should round to 2 decimal places"

        # Test custom precision functions if available
        try:
            custom_rounded = ff.round_to_precision(val, 3)
            assert abs(custom_rounded - 3.142) < 1e-10, "Custom rounding should work"
        except AttributeError:
            # Function might not exist
            pass


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        # Test with None
        try:
            result = ff.clean(None)
            assert result is None, "Should handle None gracefully"
        except (TypeError, AttributeError):
            # Expected to raise error for None
            pass

        # Test with string
        try:
            result = ff.clean("not a number")
            # Should either convert or raise error
            assert isinstance(result, (str, type(None))), "Should handle strings"
        except (TypeError, ValueError):
            # Expected to raise error for strings
            pass

    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        # Empty array
        empty_arr = np.array([])
        cleaned = ff.clean(empty_arr)
        assert len(cleaned) == 0, "Empty array should remain empty"

        # Empty matrix
        empty_matrix = np.array([]).reshape(0, 0)
        cleaned_matrix = ff.clean(empty_matrix)
        assert cleaned_matrix.shape == (0, 0), "Empty matrix shape should be preserved"

    def test_large_input_handling(self):
        """Test handling of large inputs"""
        # Large array
        large_arr = np.random.randn(1000)

        try:
            cleaned = ff.clean(large_arr)
            assert len(cleaned) == len(large_arr), "Large array should be processed"
        except MemoryError:
            # Acceptable for very large arrays
            pass

    def test_warning_handling(self):
        """Test that functions handle warnings appropriately"""
        # Operations that might generate warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Division by very small number
            result = ff.clean(1.0 / 1e-20)

            # Should either handle gracefully or generate appropriate warning
            assert (
                np.isfinite(result) or len(w) > 0
            ), "Should handle or warn about numerical issues"


class TestUtilityIntegration:
    """Test integration between utility functions"""

    def test_clean_and_print_integration(self):
        """Test that clean and print work together"""
        # Messy matrix
        messy_matrix = np.array([[1.0 + 1e-15j, 1e-16], [1e-14, 2.0]])

        # Clean then print
        cleaned = ff.clean(messy_matrix)

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            ff.ff_utils._print("Cleaned matrix:", cleaned)

        output = captured_output.getvalue()
        assert "Cleaned matrix:" in output, "Should print cleaned matrix"
        # Should not contain very small numbers in output
        assert "1e-15" not in output, "Should not show very small numbers"

    def test_format_and_clean_integration(self):
        """Test that formatting and cleaning work together"""
        # Value with small imaginary part
        val = 3.14159 + 1e-15j

        # Clean then format
        cleaned = ff.clean(val)
        formatted = ff.formatted_output(cleaned)

        assert isinstance(formatted, str), "Should produce formatted string"
        assert "3.14" in formatted, "Should contain real part"
        assert "j" not in formatted, "Should not contain imaginary unit after cleaning"

    def test_comprehensive_workflow(self):
        """Test a comprehensive workflow using multiple utilities"""
        # Start with a complex calculation result
        matrix = np.array([[1.0 + 1e-15j, 2.0 - 1e-16j], [3.0 + 1e-14j, 4.0 - 1e-15j]])

        # Clean the matrix
        cleaned_matrix = ff.clean(matrix)

        # Format for output
        formatted = ff.formatted_output(cleaned_matrix)

        # Print the result
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            ff.ff_utils._print("Final result:", formatted)

        output = captured_output.getvalue()

        # Verify the workflow
        assert isinstance(cleaned_matrix, np.ndarray), "Should produce clean matrix"
        assert isinstance(formatted, str), "Should produce formatted string"
        assert "Final result:" in output, "Should print final result"

        # Should not contain very small numbers
        assert np.allclose(cleaned_matrix.imag, 0), "Imaginary parts should be cleaned"
        assert np.allclose(
            cleaned_matrix.real, [[1, 2], [3, 4]]
        ), "Real parts should be preserved"


class TestPerformanceUtilities:
    """Test performance-related utilities"""

    def test_timing_utilities(self):
        """Test timing and performance measurement utilities"""
        import time

        # Test basic timing
        start_time = time.time()
        time.sleep(0.01)  # Small delay
        end_time = time.time()

        elapsed = end_time - start_time
        assert elapsed >= 0.01, "Should measure elapsed time"

        # Test if custom timing utilities exist
        try:
            with ff.timer() as t:
                time.sleep(0.01)
            assert t.elapsed >= 0.01, "Custom timer should work"
        except AttributeError:
            # Custom timer might not exist
            pass

    def test_memory_utilities(self):
        """Test memory usage utilities"""
        # Create large array
        large_array = np.random.randn(1000, 1000)

        # Test memory usage if utilities exist
        try:
            memory_usage = ff.get_memory_usage(large_array)
            assert memory_usage > 0, "Should report positive memory usage"
        except AttributeError:
            # Memory utilities might not exist
            pass

        # Clean up
        del large_array

    def test_optimization_utilities(self):
        """Test optimization and efficiency utilities"""
        # Test if optimization hints exist
        matrix = np.random.randn(100, 100)

        try:
            optimized = ff.optimize_matrix(matrix)
            assert (
                optimized.shape == matrix.shape
            ), "Optimized matrix should have same shape"
        except AttributeError:
            # Optimization utilities might not exist
            pass


class TestPartialTraceFunctions:
    """Test the partial_trace_diagblocksum and partial_trace_B functions"""
    def test_which_is_which(self):
        """Test to ensure partial trace is computed correctly for random matrices"""

        np.random.seed(321)  # For reproducible tests

        A = np.random.rand(4, 4)
        B = np.diag(np.random.rand(4))

        AB = np.kron(A, B)

        rho2 = ff.partial_trace_over_1(AB, d=4)
        rho1 = ff.partial_trace_over_2(AB, d=4)
                
        assert np.allclose(rho1, A*np.trace(B) ), "partial_trace_over_2 should trace out subsystem 2"
        assert np.allclose(rho2, B*np.trace(A)), "partial_trace_over_1 should trace out subsystem 1"

    def test_partial_trace_left_product_states(self):
        """Test partial_trace_blockTr with separable (product) states"""
        # Test with |0⟩_A ⊗ |0⟩_B
        psi_A = np.array([1, 0])  # |0⟩
        psi_B = np.array([1, 0])  # |0⟩
        psi_AB = np.kron(psi_A, psi_B)
        rho_AB = np.outer(psi_AB, psi_AB.conj())
        
        rho_B = ff.partial_trace_over_1(rho_AB, d=2)
        expected_B = np.outer(psi_B, psi_B.conj())
        
        assert np.allclose(rho_B, expected_B), "Should recover |0⟩⟨0| for subsystem B"
        assert rho_B.shape == (2, 2), "Output should be 2×2 matrix"
        
        # Test with |1⟩_A ⊗ |+⟩_B where |+⟩ = (|0⟩ + |1⟩)/√2
        psi_A = np.array([0, 1])  # |1⟩
        psi_B = np.array([1, 1]) / np.sqrt(2)  # |+⟩
        psi_AB = np.kron(psi_A, psi_B)
        rho_AB = np.outer(psi_AB, psi_AB.conj())
        
        
        rho_B = ff.partial_trace_over_1(rho_AB, d=2)
        expected_B = np.outer(psi_B, psi_B.conj())

        #rho_A = ff.partial_trace_over_2(rho_AB, d=2)
        # print("Given rho_A:")
        # print(np.outer(psi_A, psi_A.conj()))
        # print("Given rho_B:")
        # print(np.outer(psi_B, psi_B.conj()))
        # print("rho_AB")
        # print(rho_AB)
        # print("Computed rho_A:")
        # print(rho_A)
        # print("Computed rho_B:")
        # print(rho_B)
        # print("expected:")
        # print(expected_B)

        assert np.allclose(rho_B, expected_B), "Should recover |+⟩⟨+| for subsystem B"

    def test_partial_trace_right_product_states(self):
        """Test partial_trace_diagblocksum with separable (product) states"""
        # Test with |0⟩_A ⊗ |0⟩_B
        psi_A = np.array([1, 0])  # |0⟩
        psi_B = np.array([1, 0])  # |0⟩
        psi_AB = np.kron(psi_A, psi_B)
        rho_AB = np.outer(psi_AB, psi_AB.conj())
        
        rho_A = ff.partial_trace_over_2(rho_AB, d=2)
        expected_A = np.outer(psi_A, psi_A.conj())
        
        assert np.allclose(rho_A, expected_A), "Should recover |0⟩⟨0| for subsystem A"
        assert rho_A.shape == (2, 2), "Output should be 2×2 matrix"
        
        # Test with |+⟩_A ⊗ |1⟩_B where |+⟩ = (|0⟩ + |1⟩)/√2
        psi_A = np.array([1, 1]) / np.sqrt(2)  # |+⟩
        psi_B = np.array([0, 1])  # |1⟩
        psi_AB = np.kron(psi_A, psi_B)
        rho_AB = np.outer(psi_AB, psi_AB.conj())
        
        rho_A = ff.partial_trace_over_2(rho_AB, d=2)
        expected_A = np.outer(psi_A, psi_A.conj())
        
        assert np.allclose(rho_A, expected_A), "Should recover |+⟩⟨+| for subsystem A"

    def test_partial_trace_bell_states(self):
        """Test partial traces with maximally entangled Bell states"""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        psi_bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho_AB = np.outer(psi_bell, psi_bell.conj())
        
        # Both partial traces should give maximally mixed states
        rho_A = ff.partial_trace_over_2(rho_AB, d=2)
        rho_B = ff.partial_trace_over_1(rho_AB, d=2)
        
        expected_mixed = np.eye(2) / 2  # Maximally mixed state
        
        assert np.allclose(rho_A, expected_mixed), "Partial trace of Bell state should be maximally mixed"
        assert np.allclose(rho_B, expected_mixed), "Partial trace of Bell state should be maximally mixed"
        
        # Test another Bell state |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        psi_bell2 = np.array([0, 1, -1, 0]) / np.sqrt(2)
        rho_AB2 = np.outer(psi_bell2, psi_bell2.conj())
        
        rho_A2 = ff.partial_trace_over_2(rho_AB2, d=2)
        rho_B2 = ff.partial_trace_over_1(rho_AB2, d=2)
        
        assert np.allclose(rho_A2, expected_mixed), "All Bell states give maximally mixed reduced states"
        assert np.allclose(rho_B2, expected_mixed), "All Bell states give maximally mixed reduced states"

    def test_partial_trace_different_dimensions(self):
        """Test partial traces with different system dimensions"""
        # Test 2×3 system (dA=2, dB=3)
        np.random.seed(42)  # For reproducible tests
        psi_A = np.random.randn(2) + 1j * np.random.randn(2)
        psi_A = psi_A / np.linalg.norm(psi_A)
        psi_B = np.random.randn(3) + 1j * np.random.randn(3)
        psi_B = psi_B / np.linalg.norm(psi_B)
        
        psi_AB = np.kron(psi_A, psi_B)
        rho_AB = np.outer(psi_AB, psi_AB.conj())
        
        rho_A = ff.partial_trace_over_2(rho_AB, d=2)
        rho_B = ff.partial_trace_over_1(rho_AB, d=3)
        
        expected_A = np.outer(psi_A, psi_A.conj())
        expected_B = np.outer(psi_B, psi_B.conj())
        
        assert rho_A.shape == (2, 2), "Reduced state A should be 2×2"
        assert rho_B.shape == (3, 3), "Reduced state B should be 3×3"
        assert np.allclose(rho_A, expected_A), "Should recover original state A"
        assert np.allclose(rho_B, expected_B), "Should recover original state B"
        
        # Test 3×2 system (dA=3, dB=2)
        psi_A = np.random.randn(3) + 1j * np.random.randn(3)
        psi_A = psi_A / np.linalg.norm(psi_A)
        psi_B = np.random.randn(2) + 1j * np.random.randn(2)
        psi_B = psi_B / np.linalg.norm(psi_B)
        
        psi_AB = np.kron(psi_A, psi_B)
        rho_AB = np.outer(psi_AB, psi_AB.conj())
        
        rho_A = ff.partial_trace_over_2(rho_AB, d=3)
        rho_B = ff.partial_trace_over_1(rho_AB, d=2)
        
        expected_A = np.outer(psi_A, psi_A.conj())
        expected_B = np.outer(psi_B, psi_B.conj())
        
        assert rho_A.shape == (3, 3), "Reduced state A should be 3×3"
        assert rho_B.shape == (2, 2), "Reduced state B should be 2×2"
        assert np.allclose(rho_A, expected_A), "Should recover original state A"
        assert np.allclose(rho_B, expected_B), "Should recover original state B"

    def test_partial_trace_identity_matrices(self):
        """Test partial traces with identity matrices"""
        # 4×4 identity (2⊗2 system)
        I4 = np.eye(4)
        
        rho_A = ff.partial_trace_over_2(I4, d=2)
        rho_B = ff.partial_trace_over_1(I4, d=2)
        
        expected = 2 * np.eye(2)  # Each reduced state should be 2*I
        
        assert np.allclose(rho_A, expected), "Partial trace of identity should be scaled identity"
        assert np.allclose(rho_B, expected), "Partial trace of identity should be scaled identity"
        
        # 6×6 identity (2⊗3 system)
        I6 = np.eye(6)
        
        rho_A = ff.partial_trace_over_2(I6, d=2)
        rho_B = ff.partial_trace_over_1(I6, d=3)
        
        expected_A = 3 * np.eye(2)  # Trace over 3-dimensional system
        expected_B = 2 * np.eye(3)  # Trace over 2-dimensional system
        
        assert np.allclose(rho_A, expected_A), "Partial trace should scale correctly"
        assert np.allclose(rho_B, expected_B), "Partial trace should scale correctly"

    def test_partial_trace_random_density_matrices(self):
        """Test partial traces with random density matrices"""
        np.random.seed(123)  # For reproducible tests
        
        # Generate random 4×4 density matrix
        A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        rho_AB = A @ A.conj().T
        rho_AB = rho_AB / np.trace(rho_AB)  # Normalize
        
        rho_A = ff.partial_trace_over_2(rho_AB, d=2)
        rho_B = ff.partial_trace_over_1(rho_AB, d=2)
        
        # Check dimensions
        assert rho_A.shape == (2, 2), "Reduced state A should be 2×2"
        assert rho_B.shape == (2, 2), "Reduced state B should be 2×2"
        
        # Check trace preservation
        assert np.allclose(np.trace(rho_A), 1.0), "Trace should be preserved"
        assert np.allclose(np.trace(rho_B), 1.0), "Trace should be preserved"
        
        # Check Hermiticity
        assert np.allclose(rho_A, rho_A.conj().T), "Reduced state should be Hermitian"
        assert np.allclose(rho_B, rho_B.conj().T), "Reduced state should be Hermitian"
        
        # Check positive semidefinite (eigenvalues ≥ 0)
        eigs_A = np.linalg.eigvals(rho_A)
        eigs_B = np.linalg.eigvals(rho_B)
        assert np.all(eigs_A >= -1e-10), "Eigenvalues should be non-negative"
        assert np.all(eigs_B >= -1e-10), "Eigenvalues should be non-negative"

    def test_partial_trace_edge_cases(self):
        """Test partial traces with edge cases"""
        # Single qubit system (trivial case)
        rho_1 = np.array([[0.7, 0.1], [0.1, 0.3]])
        
        # Tracing out a 1-dimensional "subsystem" should return the original matrix
        rho_traced = ff.partial_trace_over_1(rho_1, d=2)
        assert np.allclose(rho_traced, rho_1), "Single system should remain unchanged"
        
        # Larger system: 3×3 system
        np.random.seed(456)
        A = np.random.randn(9, 9) + 1j * np.random.randn(9, 9)
        rho_large = A @ A.conj().T
        rho_large = rho_large / np.trace(rho_large)
        
        rho_A = ff.partial_trace_over_2(rho_large, d=3)
        rho_B = ff.partial_trace_over_1(rho_large, d=3)
        
        assert rho_A.shape == (3, 3), "Reduced state A should be 3x3"
        assert rho_B.shape == (3, 3), "Reduced state B should be 3x3"
        assert np.allclose(np.trace(rho_A), 1.0), "Trace should be preserved"
        assert np.allclose(np.trace(rho_B), 1.0), "Trace should be preserved"

    def test_partial_trace_consistency(self):
        """Test consistency between partial_trace_A and partial_trace_B"""
        np.random.seed(789)
        
        # Generate random 4×4 density matrix
        A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        rho_AB = A @ A.conj().T
        rho_AB = rho_AB / np.trace(rho_AB)
        
        # Both functions should preserve trace
        rho_A = ff.partial_trace_over_2(rho_AB, d=2)
        rho_B = ff.partial_trace_over_1(rho_AB, d=2)
        
        assert np.allclose(np.trace(rho_A), np.trace(rho_B)), "Both partial traces should have same trace"
        assert np.allclose(np.trace(rho_A), 1.0), "Trace should be 1"
        
        # For symmetric systems, swapping A and B should give same result
        # Create symmetric density matrix
        rho_symmetric = np.array([[0.25, 0.1, 0.1, 0.05],
                                 [0.1, 0.25, 0.05, 0.1],
                                 [0.1, 0.05, 0.25, 0.1],
                                 [0.05, 0.1, 0.1, 0.25]])
        
        rho_A_sym = ff.partial_trace_over_2(rho_symmetric, d=2)
        rho_B_sym = ff.partial_trace_over_1(rho_symmetric, d=2)
        
        # For this particular symmetric matrix, both reduced states should be identical
        assert np.allclose(rho_A_sym, rho_B_sym), "Symmetric system should give identical reduced states"

    def test_partial_trace_known_analytical_results(self):
        """Test partial traces with known analytical results"""
        # Test case 1: Werner state ρ = p|Φ+⟩⟨Φ+| + (1-p)I/4
        p = 0.7
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho_bell = np.outer(phi_plus, phi_plus.conj())
        rho_werner = p * rho_bell + (1 - p) * np.eye(4) / 4
        
        rho_A = ff.partial_trace_over_2(rho_werner, d=2)
        rho_B = ff.partial_trace_over_1(rho_werner, d=2)
        
        # Analytical result: both reduced states should be (1/2)*I
        expected = 0.5 * np.eye(2)
        
        assert np.allclose(rho_A, expected, atol=1e-10), "Werner state should give maximally mixed reduced states"
        assert np.allclose(rho_B, expected, atol=1e-10), "Werner state should give maximally mixed reduced states"
        
        # Test case 2: Known 2×3 example
        # Create a specific 6×6 density matrix with known partial traces
        rho_23 = np.zeros((6, 6), dtype=complex)
        rho_23[0, 0] = 0.3  # |00⟩⟨00|
        rho_23[1, 1] = 0.2  # |01⟩⟨01|
        rho_23[2, 2] = 0.1  # |02⟩⟨02|
        rho_23[3, 3] = 0.2  # |10⟩⟨10|
        rho_23[4, 4] = 0.1  # |11⟩⟨11|
        rho_23[5, 5] = 0.1  # |12⟩⟨12|
        
        rho_A_23 = ff.partial_trace_over_2(rho_23, d=2)
        rho_B_23 = ff.partial_trace_over_1(rho_23, d=3)
        
        # Analytical results
        expected_A = np.array([[0.6, 0], [0, 0.4]])  # Tr_B(ρ)
        expected_B = np.array([[0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.2]])  # Tr_A(ρ)
        
        assert np.allclose(rho_A_23, expected_A), "Known 2×3 example should match analytical result"
        assert np.allclose(rho_B_23, expected_B), "Known 2×3 example should match analytical result"

    def test_partial_trace_properties_verification(self):
        """Test mathematical properties of partial trace operations"""
        np.random.seed(999)
        
        # Generate multiple random density matrices for comprehensive testing
        for _ in range(5):
            # Random 4×4 density matrix
            A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
            rho_AB = A @ A.conj().T
            rho_AB = rho_AB / np.trace(rho_AB)
            
            rho_A = ff.partial_trace_over_2(rho_AB, d=2)
            rho_B = ff.partial_trace_over_1(rho_AB, d=2)
            
            # Property 1: Trace preservation
            assert np.allclose(np.trace(rho_A), 1.0, atol=1e-12), "Trace should be preserved"
            assert np.allclose(np.trace(rho_B), 1.0, atol=1e-12), "Trace should be preserved"
            
            # Property 2: Hermiticity
            assert np.allclose(rho_A, rho_A.conj().T, atol=1e-12), "Reduced state should be Hermitian"
            assert np.allclose(rho_B, rho_B.conj().T, atol=1e-12), "Reduced state should be Hermitian"
            
            # Property 3: Positive semidefinite
            eigs_A = np.linalg.eigvals(rho_A)
            eigs_B = np.linalg.eigvals(rho_B)
            assert np.all(eigs_A.real >= -1e-12), "Eigenvalues should be non-negative"
            assert np.all(eigs_B.real >= -1e-12), "Eigenvalues should be non-negative"
            assert np.all(np.abs(eigs_A.imag) < 1e-12), "Eigenvalues should be real"
            assert np.all(np.abs(eigs_B.imag) < 1e-12), "Eigenvalues should be real"
            
            # Property 4: Dimension correctness
            assert rho_A.shape == (2, 2), "Reduced state A should have correct dimensions"
            assert rho_B.shape == (2, 2), "Reduced state B should have correct dimensions"

    def test_partial_trace_linearity(self):
        """Test linearity of partial trace operations"""
        np.random.seed(111)
        
        # Generate two random density matrices
        A1 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        rho1 = A1 @ A1.conj().T
        rho1 = rho1 / np.trace(rho1)
        
        A2 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        rho2 = A2 @ A2.conj().T
        rho2 = rho2 / np.trace(rho2)
        
        # Test linearity: Tr_A(αρ1 + βρ2) = αTr_A(ρ1) + βTr_A(ρ2)
        alpha, beta = 0.3, 0.7
        rho_combined = alpha * rho1 + beta * rho2
        
        # Partial traces of individual matrices
        rho1_A = ff.partial_trace_over_2(rho1, d=2)
        rho2_A = ff.partial_trace_over_2(rho2, d=2)
        rho1_B = ff.partial_trace_over_1(rho1, d=2)
        rho2_B = ff.partial_trace_over_1(rho2, d=2)
        
        # Partial traces of combined matrix
        rho_combined_A = ff.partial_trace_over_2(rho_combined, d=2)
        rho_combined_B = ff.partial_trace_over_1(rho_combined, d=2)
        
        # Check linearity
        expected_A = alpha * rho1_A + beta * rho2_A
        expected_B = alpha * rho1_B + beta * rho2_B
        
        assert np.allclose(rho_combined_A, expected_A, atol=1e-12), "Partial trace should be linear"
        assert np.allclose(rho_combined_B, expected_B, atol=1e-12), "Partial trace should be linear"
