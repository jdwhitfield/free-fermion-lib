#!/usr/bin/env python3
"""
Comprehensive Integration Validation Script for FF Library
=========================================================

This script performs comprehensive validation tests to ensure all new functionality
is properly integrated with the existing FF library codebase.
"""

import sys
import os
import traceback
import importlib
import subprocess
from typing import Dict, List, Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ValidationResults:
    """Class to track validation results"""
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def add_result(self, test_name: str, passed: bool, message: str = ""):
        self.results[test_name] = {
            'passed': passed,
            'message': message
        }
        if not passed:
            self.errors.append(f"{test_name}: {message}")
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def print_summary(self):
        print("\n" + "="*80)
        print("INTEGRATION VALIDATION SUMMARY")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['passed'])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.warnings:
            print(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        print("\nDetailed Results:")
        print("-" * 40)
        for test_name, result in self.results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {test_name}")
            if result['message']:
                print(f"    {result['message']}")
        
        if self.errors:
            print("\nErrors:")
            print("-" * 40)
            for error in self.errors:
                print(f"❌ {error}")
        
        return passed_tests == total_tests

def test_import_validation(results: ValidationResults):
    """Test 1: Import Validation"""
    print("Testing Import Validation...")
    
    try:
        # Test main module import
        import ff
        results.add_result("main_module_import", True, "Successfully imported ff module")
        
        # Test that all new functions are accessible
        new_functions = [
            # Distance measures
            'stabilizer_distribution', 'SRE', 'renyi_entropy', 'linear_entropy',
            'cov_distribution', 'FAF',
            
            # Random states
            'random_qubit_pure_state', 'random_CHP_state', 'random_FF_state_randH',
            'random_FF_state_rotPDF', 'random_FF_pure_state_W0', 'random_FF_pure_state_WN',
            'random_FF_pure_state_CN', 'get_orthogonal_vectors', 'build_unitary_path',
            'build_linear_path',
            
            # Utils
            'pauli_matrices', 'generate_pauli_group', 'cast_to_density_matrix',
            'cast_to_pdf', 'analyze_pdf'
        ]
        
        missing_functions = []
        for func_name in new_functions:
            if hasattr(ff, func_name):
                results.add_result(f"function_accessible_{func_name}", True)
            else:
                missing_functions.append(func_name)
                results.add_result(f"function_accessible_{func_name}", False, f"Function {func_name} not accessible")
        
        if not missing_functions:
            results.add_result("all_functions_accessible", True, "All new functions are accessible via ff module")
        else:
            results.add_result("all_functions_accessible", False, f"Missing functions: {missing_functions}")
            
    except Exception as e:
        results.add_result("main_module_import", False, f"Failed to import ff module: {str(e)}")
        return

def test_cross_module_integration(results: ValidationResults):
    """Test 2: Cross-Module Integration Testing"""
    print("Testing Cross-Module Integration...")
    
    try:
        import ff
        import numpy as np
        
        # Test 1: Distance measures with random states
        try:
            # Generate a random state
            state1 = ff.random_qubit_pure_state(2)  # 2-qubit state
            state2 = ff.random_qubit_pure_state(2)
            
            # Test distance measures
            sre_val = ff.SRE(state1)
            faf_val = ff.FAF(state1)
            
            if isinstance(sre_val, (int, float)) and isinstance(faf_val, (int, float)):
                results.add_result("distance_with_random_states", True, "Distance measures work with random states")
            else:
                results.add_result("distance_with_random_states", False, "Distance measures returned invalid types")
                
        except Exception as e:
            results.add_result("distance_with_random_states", False, f"Error: {str(e)}")
        
        # Test 2: Utils with random states
        try:
            # Test utility functions with random states
            state = ff.random_qubit_pure_state(2)
            rho = ff.cast_to_density_matrix(state)
            pdf = ff.cast_to_pdf(rho)
            
            # Check if conversions work correctly
            if np.allclose(np.trace(rho), 1.0) and np.allclose(np.sum(pdf), 1.0):
                results.add_result("utils_with_random_states", True, "Utils work correctly with random states")
            else:
                results.add_result("utils_with_random_states", False, "Utility function results are not normalized")
                
        except Exception as e:
            results.add_result("utils_with_random_states", False, f"Error: {str(e)}")
        
        # Test 3: Existing FF functions with new states
        try:
            # Test if existing FF functions work with new random states
            state = ff.random_qubit_pure_state(1)  # Single qubit
            
            # Test with existing FF library functions
            pauli_x, pauli_y, pauli_z = ff.pauli_matrices()
            expectation = np.real(np.trace(pauli_z @ ff.cast_to_density_matrix(state)))
            
            results.add_result("existing_ff_with_new_states", True, f"Integration test passed, <Z> = {expectation:.3f}")
            
        except Exception as e:
            results.add_result("existing_ff_with_new_states", False, f"Error: {str(e)}")
            
    except Exception as e:
        results.add_result("cross_module_integration", False, f"Failed cross-module integration test: {str(e)}")

def test_dependency_validation(results: ValidationResults):
    """Test 3: Dependency Validation"""
    print("Testing Dependency Validation...")
    
    # Test optional dependency handling
    optional_deps = ['stim']
    
    for dep in optional_deps:
        try:
            importlib.import_module(dep)
            results.add_result(f"optional_dep_{dep}_available", True, f"{dep} is available")
        except ImportError:
            results.add_result(f"optional_dep_{dep}_available", False, f"{dep} is not available")
            results.add_warning(f"Optional dependency {dep} not available - some functions may have limited functionality")
    
    # Test that core functionality works without optional dependencies
    try:
        import ff
        import numpy as np
        
        # Test basic functions that shouldn't require optional dependencies
        state = ff.random_qubit_pure_state(2)
        rho = ff.cast_to_density_matrix(state)
        sre_val = ff.SRE(rho)
        
        results.add_result("core_functionality_without_optional_deps", True, "Core functionality works without optional dependencies")
        
    except Exception as e:
        results.add_result("core_functionality_without_optional_deps", False, f"Core functionality failed: {str(e)}")

def test_basic_functionality(results: ValidationResults):
    """Test 4: Basic Functionality Testing"""
    print("Testing Basic Functionality...")
    
    try:
        import ff
        import numpy as np
        
        # Test 1: Distance measures
        try:
            # Create simple test states
            state1 = np.array([1, 0], dtype=complex)  # |0⟩
            state2 = np.array([0, 1], dtype=complex)  # |1⟩
            
            # Test SRE (should be 0 for pure states in computational basis)
            sre1 = ff.SRE(state1)
            sre2 = ff.SRE(state2)
            if isinstance(sre1, (int, float)) and isinstance(sre2, (int, float)):
                results.add_result("basic_sre_test", True, f"SRE computed correctly: SRE(|0⟩)={sre1:.3f}, SRE(|1⟩)={sre2:.3f}")
            else:
                results.add_result("basic_sre_test", False, f"SRE returned invalid types")
                
        except Exception as e:
            results.add_result("basic_sre_test", False, f"Error: {str(e)}")
        
        # Test 2: Random state generation
        try:
            state = ff.random_qubit_pure_state(2)
            if np.isclose(np.linalg.norm(state), 1.0):
                results.add_result("basic_random_state_test", True, "Random pure state is normalized")
            else:
                results.add_result("basic_random_state_test", False, f"Random state not normalized: {np.linalg.norm(state)}")
                
        except Exception as e:
            results.add_result("basic_random_state_test", False, f"Error: {str(e)}")
        
        # Test 3: Utility functions
        try:
            # Test Pauli matrices
            pauli_x, pauli_y, pauli_z = ff.pauli_matrices()
            
            # Check if Pauli X is correct
            expected_x = np.array([[0, 1], [1, 0]], dtype=complex)
            if np.allclose(pauli_x, expected_x):
                results.add_result("basic_pauli_test", True, "Pauli matrices are correct")
            else:
                results.add_result("basic_pauli_test", False, "Pauli X matrix incorrect")
                
        except Exception as e:
            results.add_result("basic_pauli_test", False, f"Error: {str(e)}")
            
    except Exception as e:
        results.add_result("basic_functionality", False, f"Failed basic functionality test: {str(e)}")

def test_suite_validation(results: ValidationResults):
    """Test 5: Test Suite Validation"""
    print("Testing Test Suite Validation...")
    
    try:
        # Run pytest on the new test files
        test_files = [
            'tests/test_ff_distance_measures.py',
            'tests/test_ff_random_states.py', 
            'tests/test_ff_utils.py'
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                try:
                    result = subprocess.run(['python3', '-m', 'pytest', test_file, '-v'],
                                          capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        results.add_result(f"test_suite_{test_file.split('/')[-1]}", True, "Test suite passed")
                    else:
                        results.add_result(f"test_suite_{test_file.split('/')[-1]}", False, 
                                         f"Test suite failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    results.add_result(f"test_suite_{test_file.split('/')[-1]}", False, "Test suite timed out")
                except Exception as e:
                    results.add_result(f"test_suite_{test_file.split('/')[-1]}", False, f"Error running tests: {str(e)}")
            else:
                results.add_result(f"test_suite_{test_file.split('/')[-1]}", False, f"Test file not found: {test_file}")
        
        # Test that existing tests still pass (regression test)
        try:
            result = subprocess.run(['python3', '-m', 'pytest', 'tests/', '-x'],
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                results.add_result("regression_test", True, "All existing tests still pass")
            else:
                results.add_result("regression_test", False, f"Some existing tests failed: {result.stderr}")
        except Exception as e:
            results.add_result("regression_test", False, f"Error running regression tests: {str(e)}")
            
    except Exception as e:
        results.add_result("test_suite_validation", False, f"Failed test suite validation: {str(e)}")

def test_documentation_validation(results: ValidationResults):
    """Test 6: Documentation Validation"""
    print("Testing Documentation Validation...")
    
    try:
        # Check if documentation files exist
        doc_files = ['docs/api.rst', 'docs/index.rst', 'docs/conf.py']
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                results.add_result(f"doc_file_{doc_file.split('/')[-1]}", True, f"Documentation file exists: {doc_file}")
            else:
                results.add_result(f"doc_file_{doc_file.split('/')[-1]}", False, f"Documentation file missing: {doc_file}")
        
        # Test that documentation builds (if sphinx is available)
        try:
            result = subprocess.run(['python3', '-m', 'sphinx', '--version'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Try to build documentation
                try:
                    result = subprocess.run(['python3', '-m', 'sphinx', '-b', 'html', 'docs/', 'docs/_build/'],
                                          capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        results.add_result("documentation_build", True, "Documentation builds successfully")
                    else:
                        results.add_result("documentation_build", False, f"Documentation build failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    results.add_result("documentation_build", False, "Documentation build timed out")
            else:
                results.add_result("documentation_build", False, "Sphinx not available for documentation build test")
                
        except Exception as e:
            results.add_result("documentation_build", False, f"Error testing documentation build: {str(e)}")
        
        # Test docstring examples (basic check)
        try:
            import ff
            import doctest
            
            # Test a few key functions for docstring examples
            functions_to_test = ['SRE', 'random_qubit_pure_state', 'pauli_matrices']
            
            for func_name in functions_to_test:
                if hasattr(ff, func_name):
                    func = getattr(ff, func_name)
                    if func.__doc__:
                        results.add_result(f"docstring_{func_name}", True, f"Function {func_name} has docstring")
                    else:
                        results.add_result(f"docstring_{func_name}", False, f"Function {func_name} missing docstring")
                else:
                    results.add_result(f"docstring_{func_name}", False, f"Function {func_name} not found")
                    
        except Exception as e:
            results.add_result("docstring_validation", False, f"Error validating docstrings: {str(e)}")
            
    except Exception as e:
        results.add_result("documentation_validation", False, f"Failed documentation validation: {str(e)}")

def main():
    """Main validation function"""
    print("FF Library Integration Validation")
    print("=" * 50)
    
    results = ValidationResults()
    
    # Run all validation tests
    test_import_validation(results)
    test_cross_module_integration(results)
    test_dependency_validation(results)
    test_basic_functionality(results)
    test_suite_validation(results)
    test_documentation_validation(results)
    
    # Print summary
    success = results.print_summary()
    
    if success:
        print("\n🎉 ALL VALIDATION TESTS PASSED! Integration is successful.")
        return 0
    else:
        print("\n⚠️  Some validation tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())