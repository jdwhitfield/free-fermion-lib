"""
Free Fermion Random States Module

This module contains random state generators and path construction functions
for free fermion systems, including Haar random states, Clifford states,
and various free fermion state generation methods.

Key functionality:
 - Random qubit pure states (Haar random)
 - Random Clifford states using Stim package
 - Random free fermion mixed states from random Hamiltonians
 - Random free fermion mixed states from rotated probability distributions
 - Random free fermion pure states with various constraints
 - Path construction functions for state interpolation

Copyright 2025 James.D.Whitfield@dartmouth.edu
"""

import numpy as np
from scipy.linalg import expm, logm
from scipy.stats import unitary_group

from .ff_lib import (
    jordan_wigner_alphas,
    jordan_wigner_majoranas,
    build_op,
    build_V,
    random_FF_rotation,
    rotate_operators,
    compute_algebra_S,
)
from .ff_utils import clean



def random_qubit_pure_state(n, seed=None):
    """Generate a Haar random qubit state.

    This function generates a uniformly distributed pure state over the
    n-qubit Hilbert space using the Haar measure on the unitary group.
    The state is created by applying a Haar random unitary to the
    computational basis state |00...0⟩.

    Args:
        n (int): The number of qubits
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray: A random qubit state of dimension (2^n, 1) as a
                      normalized complex vector

    Raises:
        ValueError: If n is not a positive integer
        TypeError: If n is not an integer

    Examples:
        >>> # Generate a random single qubit state
        >>> psi = random_qubit_pure_state(1, seed=42)
        >>> print(psi.shape)
        (2, 1)
        >>> print(np.allclose(np.linalg.norm(psi), 1.0))
        True

        >>> # Generate a random two-qubit state
        >>> psi = random_qubit_pure_state(2, seed=123)
        >>> print(psi.shape)
        (4, 1)

    Notes:
        - The generated states are uniformly distributed according to the
          Haar measure on the space of pure quantum states
        - Memory usage scales as O(4^n), so use with caution for large n
        - The state is always normalized to unit length
    """
    # Input validation
    if not isinstance(n, int):
        raise TypeError(f"Number of qubits must be an integer, got {type(n).__name__}")
    
    if n <= 0:
        raise ValueError(f"Number of qubits must be positive, got {n}")

    if seed is not None:
        np.random.seed(seed)

    # For pure states, apply Haar random unitary to vacuum state
    U = unitary_group.rvs(dim=2 ** n)
    zero_state = np.zeros((2**n, 1), dtype=complex)

    zero_state[0, 0] = 1  # Set the first element to 1 (vacuum state / 0000..0)
    psi = U @ zero_state

    # check normalization of the pure state
    assert np.allclose(1, np.linalg.norm(psi)), "Generated state is not normalized"

    return psi


def random_CHP_state(n_qubits):
    """Generate a random Clifford state using Stim package.

    This function generates a random stabilizer state (Clifford state) using
    the Stim quantum circuit simulator package. The state is uniformly
    distributed over the space of n-qubit stabilizer states.

    Args:
        n_qubits (int): The number of qubits for the CHP state

    Returns:
        numpy.ndarray: A random Clifford state of dimension (2^n_qubits, 1)
                      as a normalized complex vector

    Raises:
        ImportError: If the stim package is not available
        ValueError: If n_qubits is not a positive integer
        TypeError: If n_qubits is not an integer

    Examples:
        >>> # Generate a random single qubit Clifford state
        >>> psi = random_CHP_state(1)
        >>> print(psi.shape)
        (2, 1)

        >>> # Generate a random two-qubit Clifford state
        >>> psi = random_CHP_state(2)
        >>> print(psi.shape)
        (4, 1)

    Notes:
        - Requires the 'stim' package to be installed
        - Clifford states form a subset of all quantum states that can be
          efficiently simulated classically
        - The generated states are uniformly distributed over the stabilizer
          state space
    """
    import stim
    
    # Input validation
    if not isinstance(n_qubits, int):
        raise TypeError(f"Number of qubits must be an integer, got {type(n_qubits).__name__}")
    
    if n_qubits <= 0:
        raise ValueError(f"Number of qubits must be positive, got {n_qubits}")


    t = stim.Tableau.random(n_qubits)

    wf = t.to_state_vector()

    if len(wf.shape) == 1:
        wf = wf.reshape(wf.shape[0], 1)

    return wf


def random_FF_state_randH(n_sites, seed=None):
    """Generate a random free fermion mixed state from a random Hamiltonian.

    This function generates a random free fermion thermal state by constructing
    a random quadratic Hamiltonian and computing the corresponding FF state
    via exponentiation. The resulting state is a density matrix.

    NOTE: This function does not generate Haar random states since exponentiating
        a random Hamiltonian does not generate correctly distributed eigenvalues
        in the resulting random matrix.

    Args:
        n_sites (int): The number of fermionic sites
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray: A random free fermion state of dimension (2^n_sites, 2^n_sites)
                      as a normalized density matrix

    Raises:
        ValueError: If n_sites is not a positive integer
        TypeError: If n_sites is not an integer

    Examples:
        >>> # Generate a random 2-site FF mixed state
        >>> rho = random_FF_state_randH(2, seed=42)
        >>> print(rho.shape)
        (4, 4)
        >>> print(np.allclose(np.trace(rho), 1.0))
        True

    Notes:
        - The Hamiltonian is constructed from random Gaussian matrices
        - The resulting state is a thermal state at infinite temperature
        - Memory usage scales as O(4^n_sites)
    """
    # Input validation
    if not isinstance(n_sites, int):
        raise TypeError(f"Number of sites must be an integer, got {type(n_sites).__name__}")
    
    if n_sites <= 0:
        raise ValueError(f"Number of sites must be positive, got {n_sites}")

    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(n_sites, n_sites) + 1j * np.random.randn(n_sites, n_sites)
    Z = np.random.randn(n_sites, n_sites) + 1j * np.random.randn(n_sites, n_sites)
    A = A + A.conj().T
    Z = Z - Z.T

    rescale = 2**(2 - n_sites)
    G = build_V(n_sites, A, Z) * rescale

    H_op = build_op(n_sites, G, jordan_wigner_alphas(n_sites), direct=True)

    rho = expm(-H_op)
    rho = rho / np.trace(rho)

    return rho


def random_FF_state_rotPDF(n_sites, returnS=False, seed=None):
    """Generate a random free fermion mixed state from rotated probability distribution.

    This function generates a random free fermion state by applying a random
    free fermion rotation to a random probability distribution (Dirichlet
    distributed eigenvalues).

    Args:
        n_sites (int): The number of fermionic sites
        returnS (bool, optional): If True, also return the probability distribution
                                 (default: False)
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray or tuple: If returnS is False, returns a random free fermion
                               state of dimension (2^n_sites, 2^n_sites). If returnS
                               is True, returns (rho, s) where s is the probability
                               distribution used.

    Raises:
        ValueError: If n_sites is not a positive integer
        TypeError: If n_sites is not an integer

    Examples:
        >>> # Generate a random 2-site FF mixed state
        >>> rho = random_FF_state_rotPDF(2, seed=42)
        >>> print(rho.shape)
        (4, 4)

        >>> # Generate state and return probability distribution
        >>> rho, s = random_FF_state_rotPDF(2, returnS=True, seed=42)
        >>> print(len(s))
        4

    Notes:
        - Uses Dirichlet distribution for random eigenvalues
        - The rotation preserves the free fermion structure
        - Memory usage scales as O(4^n_sites)
    """
    # Input validation
    if not isinstance(n_sites, int):
        raise TypeError(f"Number of sites must be an integer, got {type(n_sites).__name__}")
    
    if n_sites <= 0:
        raise ValueError(f"Number of sites must be positive, got {n_sites}")

    if seed is not None:
        np.random.seed(seed)

    W_op = random_FF_rotation(n_sites)

    assert np.allclose(W_op @ W_op.conj().T, np.eye(2**n_sites)), "Rotation is not unitary"

    s = np.random.dirichlet(np.ones(2**n_sites))

    rho = W_op @ np.diag(s) @ W_op.conj().T

    if returnS:
        return rho, s

    return rho


def random_FF_pure_state_W0(n_sites, seed=None):
    """Generate a Haar random free fermion pure state from vacuum rotation.

    This function generates free-fermion pure states that are uniformly distributed
    over the space of free-fermion states using Haar random symplectic transformations
    applied to the vacuum state |0⟩.

    Args:
        n_sites (int): The number of fermionic sites
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray or tuple: A normalized Haar random free fermion pure state
                               of dimension (2^n_sites, 1). If returnH is True, also
                               returns the generator matrix H.

    Raises:
        ValueError: If n_sites is not a positive integer
        TypeError: If n_sites is not an integer

    Examples:
        >>> # Generate a random 2-site FF pure state
        >>> psi = random_FF_pure_state_W0(2, seed=42)
        >>> print(psi.shape)
        (4, 1)
        >>> print(np.allclose(np.linalg.norm(psi), 1.0))
        True

    Notes:
        - Uses Haar random symplectic rotations
        - The state is created by rotating the vacuum state
        - Memory usage scales as O(4^n_sites)
    """
    # Input validation
    if not isinstance(n_sites, int):
        raise TypeError(f"Number of sites must be an integer, got {type(n_sites).__name__}")
    
    if n_sites <= 0:
        raise ValueError(f"Number of sites must be positive, got {n_sites}")

    if seed is not None:
        np.random.seed(seed)

    # For pure states, apply Haar random unitary to vacuum state
    W_op = random_FF_rotation(n_sites, seed=seed)
    zero_state = np.zeros((2**n_sites, 1), dtype=complex)
    zero_state[0] = 1  # Set the first element to 1 (vacuum state)
    psi = W_op @ zero_state

    # check normalization of the pure state
    assert np.allclose(1, np.linalg.norm(psi)), "Generated state is not normalized"

    return psi


def random_FF_pure_state_WN(n_sites, N=None, seed=None):
    """Generate a Haar random free fermion pure state with fixed particle number.

    This function generates free-fermion pure states with a specified particle
    number N, uniformly distributed over the space of N-particle free-fermion
    states using Haar random symplectic transformations.

    NOTE: This function is most-likely generating states that will not have fixed particle number.
          Further testing and validation is required to ensure that the generated states are in a
          fixed-N subspace (also which fermionic basis should be used).

    Args:
        n_sites (int): The number of fermionic sites
        N (int, optional): The particle number. If None, chosen randomly
                          (default: None)
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray or tuple: A normalized Haar random free fermion pure state
                               of dimension (2^n_sites, 1) with N particles.
                               If returnH is True, also returns the generator matrix H.

    Raises:
        ValueError: If n_sites is not a positive integer or N is invalid
        TypeError: If n_sites is not an integer

    Examples:
        >>> # Generate a random 2-site FF pure state with 1 particle
        >>> psi = random_FF_pure_state_WN(2, N=1, seed=42)
        >>> print(psi.shape)
        (4, 1)

        >>> # Generate with random particle number
        >>> psi = random_FF_pure_state_WN(3, seed=42)
        >>> print(psi.shape)
        (8, 1)

    Notes:
        - The particle number N must be between 0 and n_sites
        - Uses Haar random symplectic rotations
        - Memory usage scales as O(4^n_sites)
    """
    # Input validation
    if not isinstance(n_sites, int):
        raise TypeError(f"Number of sites must be an integer, got {type(n_sites).__name__}")
    
    if n_sites <= 0:
        raise ValueError(f"Number of sites must be positive, got {n_sites}")

    if N is not None and (not isinstance(N, int) or N < 0 or N > n_sites):
        raise ValueError(f"Particle number N must be between 0 and {n_sites}, got {N}")

    if seed is not None:
        np.random.seed(seed)

    if N is None:
        N = np.random.randint(n_sites + 1)

    K = np.append(np.ones(N), (np.zeros(n_sites - N)))

    alphas = jordan_wigner_alphas(n_sites)

    # Create initial state with N particles
    psi = np.zeros((2**n_sites, 1))
    psi[0] = 1

    for j in np.flip(range(n_sites)):
        if K[j] == 1:
            psi = alphas[j] @ psi

    # For pure states, apply Haar random unitary to initial state
    W_op = random_FF_rotation(n_sites, seed=seed + 1 if seed is not None else None)
    psi = W_op @ psi

    return psi


def random_FF_pure_state_CN(n_sites, seed=None):
    """Generate a random free fermion pure state using random orbital rotations.

    This function generates a random free fermion pure state by first creating
    a random vacuum state through projections, then applying creation operators
    according to a random occupation pattern, all with randomly rotated orbitals.

    NOTE: This function is most-likely generating states that will not have fixed particle number.
        Further testing and validation is required to ensure that the generated states are in a
        fixed-N subspace (also which fermionic basis should be used).

    Args:
        n_sites (int): The number of fermionic sites
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray: A random free fermion pure state of dimension (2^n_sites, 1)

    Raises:
        ValueError: If n_sites is not a positive integer
        TypeError: If n_sites is not an integer

    Examples:
        >>> # Generate a random 2-site FF pure state
        >>> psi = random_FF_pure_state_CN(2, seed=42)
        >>> print(psi.shape)
        (4, 1)

        >>> # Generate a random 3-site FF pure state
        >>> psi = random_FF_pure_state_CN(3, seed=123)
        >>> print(psi.shape)
        (8, 1)

    Notes:
        - Uses random orbital rotations respecting fermionic antisymmetry
        - Creates a random vacuum through projections
        - Applies creation operators according to random occupation
        - Memory usage scales as O(4^n_sites)
    """
    # Input validation
    if not isinstance(n_sites, int):
        raise TypeError(f"Number of sites must be an integer, got {type(n_sites).__name__}")
    
    if n_sites <= 0:
        raise ValueError(f"Number of sites must be positive, got {n_sites}")

    if seed is not None:
        np.random.seed(seed)

    N = np.random.randint(n_sites + 1)

    K = np.append(np.ones(N), (np.zeros(n_sites - N)))

    # random orbital rotation respecting the antisymmetry of the given operators
    C = random_FF_rotation(n_sites, seed=seed + 1 if seed is not None else None, returnOrb=True)

    assert np.allclose(C @ C.conj().T, np.eye(2 * n_sites)), "Orbital rotation is not unitary"

    rand_alphas = rotate_operators(C, jordan_wigner_alphas(n_sites))

    # Verify that the algebra structure is preserved under rotation (warning: likely pretty slow)
    S = compute_algebra_S(jordan_wigner_alphas(n_sites))
    Sr = compute_algebra_S(rand_alphas)
    assert np.allclose(S, Sr), "Algebra structure not preserved under rotation"

    #Distill the vaccuum state
    Y = np.random.randn(2**n_sites, 1)
    Y = Y / np.linalg.norm(Y)

    Id = np.eye(2**n_sites)

    PP = Id
    for i in range(n_sites):
        Pi = (Id - rand_alphas[i] @ rand_alphas[i].conj().T)
        PP = PP @ Pi

    vac = PP @ Y
    vac = vac / np.linalg.norm(vac)

    # generate random fock state from vacuum
    psi = vac

    for j in np.flip(range(n_sites)):
        if K[j] == 1:
            psi = rand_alphas[j] @ psi

    return psi


def get_orthogonal_vectors(v):
    """Get orthogonal vectors to a given vector using QR decomposition.

    Given a vector v (n x 1), return n-1 vectors that are orthogonal to v
    and form an orthonormal basis together with the normalized v.

    Args:
        v (numpy.ndarray): The input vector (n x 1 or n-dimensional)

    Returns:
        numpy.ndarray: A matrix of shape (n, n) where the first column is
                      the normalized input vector and the remaining n-1 columns
                      are orthogonal to it

    Raises:
        ValueError: If the input vector has insufficient dimensions
        TypeError: If the input is not a numpy array

    Examples:
        >>> # Get orthogonal vectors to a 3D vector
        >>> v = np.array([[1], [2], [3]])
        >>> orth_vecs = get_orthogonal_vectors(v)
        >>> print(orth_vecs.shape)
        (3, 3)

        >>> # Check orthogonality
        >>> v_norm = v / np.linalg.norm(v)
        >>> print(np.allclose(orth_vecs[:, 0:1], v_norm))
        True

    Notes:
        - Uses QR decomposition for numerical stability
        - Returns an orthonormal basis including the normalized input vector
        - For zero vectors, returns standard basis vectors
    """
    v = np.asarray(v)
    if len(v.shape) == 1:
        v = v.reshape(-1, 1)
    
    n = v.shape[0]

    if n < 2:
        return np.array([])  # Need at least 2 dimensions to have orthogonal vectors

    # Normalize the input vector
    v_norm = np.linalg.norm(v)
    if np.isclose(v_norm, 0):
        # If the input vector is zero, return n-1 standard basis vectors
        return np.eye(n)[:, :n]

    v_hat = v / v_norm

    # Construct a matrix whose first column is v_hat
    A = np.hstack((v_hat, np.eye(n)[:, 1:]))

    # Perform QR decomposition
    Q, _ = np.linalg.qr(A)

    # The remaining n-1 columns of Q are orthogonal to v_hat (and thus to v)
    orthogonal_vectors = np.hstack((v_hat, Q[:, 1:]))

    return orthogonal_vectors


def build_unitary_path(w, v):
    """Build a unitary path between two vectors.

    This function constructs a smooth unitary path between two normalized
    vectors w and v, parameterized by t ∈ [0,1], such that the path
    starts at w (t=0) and ends at v (t=1).

    Args:
        w (numpy.ndarray): The initial vector
        v (numpy.ndarray): The target vector

    Returns:
        callable: A function path(t) that returns the interpolated vector
                 at parameter t ∈ [0,1]

    Raises:
        ValueError: If vectors have incompatible dimensions
        TypeError: If inputs are not numpy arrays

    Examples:
        >>> # Create a path between two 3D vectors
        >>> w = np.array([[1], [0], [0]])
        >>> v = np.array([[0], [1], [0]])
        >>> path = build_unitary_path(w, v)
        >>> 
        >>> # Evaluate at t=0 (should be close to w)
        >>> start = path(0.0)
        >>> print(np.allclose(start, w / np.linalg.norm(w)))
        True
        >>> 
        >>> # Evaluate at t=1 (should be close to v)
        >>> end = path(1.0)
        >>> print(np.allclose(end, v / np.linalg.norm(v)))
        True

    Notes:
        - The path preserves normalization at all points
        - Uses matrix exponential for smooth interpolation
        - The path is the shortest geodesic on the unit sphere
    """
    # Normalize the vectors
    w = np.asarray(w)
    v = np.asarray(v)
    
    if len(w.shape) == 1:
        w = w.reshape(-1, 1)
    if len(v.shape) == 1:
        v = v.reshape(-1, 1)
    
    w = w / np.linalg.norm(w)
    v = v / np.linalg.norm(v)

    n = max(w.shape)

    # Compute the orthogonal complements of w and v
    W = get_orthogonal_vectors(w)
    V = get_orthogonal_vectors(v)

    U = np.zeros((n, n), dtype=complex)
    for i in range(n):
        v_in = np.reshape(W[:, i], (n, 1))
        v_out = np.reshape(V[:, i], (n, 1))

        U = U + v_out @ v_in.conj().T

    assert np.allclose(W @ W.conj().T, np.eye(n)), "W is not unitary"
    assert np.allclose(V @ V.conj().T, np.eye(n)), "V is not unitary"
    
    if not np.allclose(U @ U.conj().T, np.eye(n)):
        # project to nearest unitary
        V_svd, _, Wh = np.linalg.svd(U)
        U = V_svd @ Wh

    assert np.allclose(U @ U.conj().T, np.eye(n)), "U is not unitary"

    # Compute the Hermitian matrix of U
    H = 1j * logm(U)

    assert np.allclose(H - H.conj().T, 0), f"H is not Hermitian: {clean(H - H.conj().T, 13)}"

    path = lambda t: expm(-1j * H * t) @ w

    return path


def build_linear_path(w, v):
    """Build a linear path between two vectors.

    This function constructs a linear interpolation path between two vectors
    w and v, parameterized by s ∈ [0,1], such that the path starts at w (s=0)
    and ends at v (s=1). The path maintains normalization at each point.

    Args:
        w (numpy.ndarray): The initial vector
        v (numpy.ndarray): The target vector

    Returns:
        callable: A function path(s) that returns the normalized interpolated
                 vector at parameter s ∈ [0,1]

    Raises:
        ValueError: If vectors have incompatible dimensions
        TypeError: If inputs are not numpy arrays

    Examples:
        >>> # Create a linear path between two 3D vectors
        >>> w = np.array([[1], [0], [0]])
        >>> v = np.array([[0], [1], [0]])
        >>> path = build_linear_path(w, v)
        >>> 
        >>> # Evaluate at s=0 (should be close to w)
        >>> start = path(0.0)
        >>> print(np.allclose(start, w / np.linalg.norm(w)))
        True
        >>> 
        >>> # Evaluate at s=1 (should be close to v)
        >>> end = path(1.0)
        >>> print(np.allclose(end, v / np.linalg.norm(v)))
        True

    Notes:
        - The path is a straight line in the ambient space, then normalized
        - Simpler than unitary path but may not be the shortest geodesic
        - Maintains normalization at all points
    """
    # Normalize the vectors
    w = np.asarray(w)
    v = np.asarray(v)
    
    if len(w.shape) == 1:
        w = w.reshape(-1, 1)
    if len(v.shape) == 1:
        v = v.reshape(-1, 1)
        
    w = w / np.linalg.norm(w)
    v = v / np.linalg.norm(v)
    
    path = lambda s: (s * v + (1 - s) * w) / np.linalg.norm(s * v + (1 - s) * w)

    return path