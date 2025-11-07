"""
Free Fermion Utilities Module

Common utility functions for the free fermion codebase.

Copyright 2025 James.D.Whitfield@dartmouth.edu
Licensed under MIT License.
"""

import numpy as np
import matplotlib.pyplot as plt


def print_custom(obj, k=9):
    """Custom print function with small number suppression

    Args:
        obj: Any object to be printed
        k: The number of decimal places to print

    Returns:
        None
    """
    if isinstance(obj, (int, float, complex, list, np.ndarray, np.matrix)):
        _print(obj, k)
    else:
        print(obj)


def _print(obj, k=9):
    """Printing with small number suppression (using numpy printoptions)

    Args:
        obj: Any object to be printed
        k: The number of decimal places to print

    Returns:
        None
    """
    # get current precision
    p = np.get_printoptions()["precision"]

    try:
        val = np.array(obj)

        # check if input is completely real
        # If it is, don't print complex part
        if np.allclose(val.imag, np.zeros_like(val)):
            val = val.real

        # change to request precision
        np.set_printoptions(precision=k)

        # clean and print
        print(clean(val, k))

    except (ValueError, TypeError):
        # If numpy array conversion fails, just print the object
        print(obj)

    finally:
        # reset precision
        np.set_printoptions(precision=p)


def clean(obj, threshold=1e-6):
    """
    Clean small numerical values from arrays or matrices.

    Args:
        obj: array, scalar, NumPy array or matrix
        threshold: Values below this threshold are set to zero

    Note: if threshold is an integer, it will be converted to 10^-threshold

    Returns:
        Cleaned obj with rounded values and small values set to zero.
    """

    if isinstance(threshold, int):
        # If threshold is an integer, convert to 10^-threshold
        ndigits = threshold
        threshold = 10 ** (-threshold)
    else:
        ndigits = -round(np.log10(threshold))

    if isinstance(obj, list):
        # If it's a list, convert to numpy array
        obj_array = np.array(obj)
        obj_array = np.round(obj_array, ndigits)
        # Set small values to zero
        obj_array[np.abs(obj_array) < threshold] = 0
        return obj_array.tolist()

    elif isinstance(obj, (np.matrix, np.ndarray)):
        # If it's a numpy matrix or array, ensure it's a numpy array
        if hasattr(obj, "imag"):  # if complex, check for small imaginary part
            # If it's complex, check the imaginary part
            if np.all(np.abs(obj.imag) < threshold):
                # If the imaginary part is small, return only the real part
                return np.round(obj.real, ndigits)
        # Round the array and set small values to zero
        obj = np.round(obj, ndigits)
        obj[np.abs(obj) < threshold] = 0
        return obj

    if isinstance(obj, str):
        if obj.replace(".", "", 1).isnumeric():
            # If it's a numeric string, convert to float
            obj = float(obj)
            obj = np.round(obj, ndigits)
            return str(obj)
        else:
            # If it's a non-numeric string, return as is
            return obj

    elif isinstance(obj, (int, float)):
        return np.round(obj, ndigits)

    elif isinstance(obj, complex):
        if abs(obj.imag) < threshold:
            # If the imaginary part is small, return only the real part
            return np.round(obj.real, ndigits)
        else:
            # If the imaginary part is significant, return the complex number rounded
            return np.round(obj, ndigits)
    else:
        raise TypeError("Unsupported type for cleaning: {}".format(type(obj)))


def formatted_output(obj, precision=6):
    """
    Format numerical output with specified precision.

    Args:
        obj: Object to format
        precision: Number of decimal places

    Returns:
        Formatted string representation
    """
    if isinstance(obj, (int, float, complex)):
        if isinstance(obj, complex):
            if abs(obj.imag) < 1e-10:
                return f"{obj.real:.{precision}f}"
            else:
                return f"{obj.real:.{precision}f} + {obj.imag:.{precision}f}j"
        else:
            return f"{obj:.{precision}f}"
    else:
        return str(obj)


def generate_random_bitstring(n, k):
    """Generates a random bit string of length n with Hamming weight k.

    Based on `np.random.choice`

    Args:
        n: The length of the bit string.
        k: The Hamming weight (number of 1s).

    Returns:
        A NumPy array representing the bit string, or None if k is invalid.
    """
    if k < 0 or k > n:
        return None  # Invalid Hamming weight

    bitstring = np.zeros(n, dtype=int)

    indices = np.random.choice(n, size=k, replace=False)
    bitstring[indices] = 1
    return bitstring


def kron_plus(a, b):
    """Computes the direct sum of two matrices

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Direct sum matrix [[a, 0], [0, b]]
    """
    Z01 = np.zeros((a.shape[0], b.shape[1]))
    return np.block([[a, Z01], [Z01.T, b]])


def cast_to_pdf(rho):
    """Casts input density matrix or wave function to a probability distribution."""

    rho = cast_to_density_matrix(np.asarray(rho, dtype=complex))
    pdf = np.diag(rho)

    #cast to real if possible
    if np.allclose(pdf,pdf.real):
        pdf = pdf.real
    else:
        raise ValueError("Probability distribution has non-negligible imaginary part")

    return pdf


def cast_to_density_matrix(rho):
    """Casts input to a density matrix.

    Args:
        rho: Input array, can be a density matrix, wavefunction, or probability distribution.

    Returns:
        Density matrix as a NumPy array.
    """
    rho = np.asarray(rho, dtype=complex)

    # Handle vector inputs (1D arrays or column vectors)
    if len(rho.shape) == 1 or min(rho.shape) == 1:
        # Flatten to 1D for norm calculations
        rho_flat = rho.flatten()
        l1_norm = np.linalg.norm(rho_flat, ord=1)
        l2_norm = np.linalg.norm(rho_flat, ord=2)

        # Determine if it's a wavefunction or probability distribution
        if np.allclose(l2_norm, 1):
            # Treated as normalized wavefunction
            rho = rho.reshape(-1, 1)  # Ensure column vector
            rho = rho @ rho.conj().T
            #print("assuming wavefunction input")
        elif np.allclose(l1_norm, 1):
            # Treated as normalized probability distribution
            rho = np.diag(rho_flat)
            #print("assuming prob input")
        else:
            raise ValueError(
                f"Vector input must be normalized (L1 or L2 norm ≈ 1). "
                f"Got L1={l1_norm:.6f}, L2={l2_norm:.6f}"
            )

    # Validate density matrix properties
    if len(rho.shape) != 2 or rho.shape[0] != rho.shape[1] or not np.allclose(np.trace(rho),1):
        raise ValueError("Density matrix must be square and normalized (trace=1)")

    return rho


def analyze_pdf(rho, name=None, stem=True):
    """Population analysis and visualization for probability distributions.
    
    Analyzes and visualizes probability distributions from density matrices or 
    wavefunctions. Computes both diagonal elements and eigenvalues of the density
    matrix and displays them with optional logarithmic scaling.
    
    Args:
        rho (numpy.ndarray): Density matrix, wavefunction, or probability distribution.
            - If 2D square matrix: treated as density matrix
            - If 1D array or column vector: interpreted based on normalization
                - L2 norm ≈ 1: treated as normalized wavefunction
                - L1 norm ≈ 1: treated as normalized probability distribution
        name (str, optional): Name for the plot title. If provided, will appear 
            as "Population analysis of {name}". Default is None.
        stem (bool, optional): Whether to use stem plot (True) or line plot (False).
            Default is True.
    
    Returns:
        None: Function creates a matplotlib plot but does not return values.
        
    Raises:
        ValueError: If input array has invalid dimensions or normalization.
        TypeError: If input is not a numpy array or array-like object.
        
    Notes:
        - The function automatically converts wavefunctions to density matrices
        - Diagonal elements are plotted in red, eigenvalues in black dashed
        - Y-axis uses logarithmic scaling to highlight small probabilities
        - Small numerical values are cleaned using the FF library clean() function
        - Plot is created but not displayed; user must call plt.show() separately
        
    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> # Analyze a simple 2x2 density matrix
        >>> rho = np.array([[0.7, 0.1], [0.1, 0.3]])
        >>> analyze_pdf(rho, name="2-level system")
        >>> plt.show()
        
        >>> # Analyze a wavefunction
        >>> psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        >>> analyze_pdf(psi, name="Bell state", stem=False)
        >>> plt.show()
    """
    # Input validation
    if not isinstance(rho, (np.ndarray, list)):
        raise TypeError("Input must be a numpy array or array-like object")
    
    rho = np.asarray(rho, dtype=complex)
    
    if rho.size == 0:
        raise ValueError("Input array cannot be empty")
    
    rho = cast_to_density_matrix(rho)

    # Extract diagonal elements and eigenvalues
    pdf = np.diag(rho)
    eigenvals, _ = np.linalg.eigh(rho)
    
    # Clean small numerical artifacts
    pdf = clean(pdf)
    eigenvals = clean(eigenvals)
    
    # Sort eigenvalues in descending order for better visualization
    eigenvals = np.sort(eigenvals)[::-1]
    
    # Create the plot
    if stem:
        plt.stem(range(len(pdf)), pdf, linefmt='red', markerfmt='ro', 
                basefmt=' ', label='diag(ρ)')
        plt.stem(range(len(eigenvals)), eigenvals, linefmt='k--', markerfmt='k^',
                basefmt=' ', label='eigenvalues')
    else:
        plt.plot(pdf, color="red", marker='o', label='diag(ρ)')
        plt.plot(eigenvals, color='black', linestyle='dashed', marker='^', 
                label='eigenvalues')
    
    # Set logarithmic scale and labels
    plt.yscale('log')
    plt.ylabel("Probability (log scale)")
    plt.xlabel("Index")
    
    # Set title
    title = "Population analysis"
    if name is not None:
        title += f" of {name}"
    plt.title(title)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Note: Function does not call plt.show() to allow user control
    return
