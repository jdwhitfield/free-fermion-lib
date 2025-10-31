"""
Free Fermion Library - Basic Functions For Handling Fermions

This module contains the core quantum physics and linear algebra functions
for free fermion systems, including Jordan-Wigner transformations, symplectic
diagonalization, Gaussian state constructions, and correlation matrix
computations.

Key functionality:
 - Jordan-Wigner fermionic operators (Dirac and Majorana)
 - Symplectic free-fermion diagonalization
 - Gaussian state generation and manipulation
 - Fermionic correlation matrix computations
 - Block matrix construction for coefficient matrices
 - Wick's theorem implementation

Copyright 2025 James.D.Whitfield@dartmouth.edu
"""

import numpy as np
from scipy.linalg import expm, logm, schur
from scipy.stats import special_ortho_group

from .ff_utils import _print, kron_plus


def permutation_to_matrix(permutation):
    """
    Transform a permutation (list) into a permutation matrix using NumPy.

    Args:
        permutation: A list representing a permutation

    Returns:
        A NumPy array representing the permutation matrix.
        Returns None if the input is invalid.
    """
    n = len(permutation)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        if 0 <= permutation[i] < n:
            matrix[permutation[i], i] = 1
        else:
            print("Invalid permutation")
            return None
    return matrix


def pauli_matrices():
    """Define the Pauli matrices as numpy arrays."""
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    return sigma_x, sigma_y, sigma_z


def generate_pauli_group(n, verbose=False):
    """
    Generate all 4^n elements of the Pauli group for n qubits.
    
    This function generates all possible tensor products of the Pauli matrices
    (I, X, Y, Z) for n qubits, creating the complete Pauli group of order 4^n.
    Each element is a 2^n x 2^n matrix representing a Pauli operator acting
    on the n-qubit Hilbert space.
    
    The Pauli group is fundamental in quantum error correction, quantum
    computing, and stabilizer formalism. It consists of all n-fold tensor
    products of single-qubit Pauli operators.
    
    Args:
        n (int): The number of qubits. Must be a positive integer.
        verbose (bool): If True, print the Pauli string name for each operator
                       as it's generated (default: False).
        
    Returns:
        list: A list of 4^n numpy arrays, each of shape (2^n, 2^n),
              representing all Pauli operators for n qubits. The operators
              are ordered lexicographically by their tensor product structure.
              
    Raises:
        TypeError: If n is not an integer.
        ValueError: If n is not positive.
        
    Examples:
        >>> # Single qubit case (n=1)
        >>> pauli_1 = generate_pauli_group(1)
        >>> len(pauli_1)
        4
        >>> pauli_1[0]  # Identity
        array([[1, 0],
               [0, 1]])
        >>> pauli_1[1]  # Pauli-X
        array([[0, 1],
               [1, 0]])
               
        >>> # Two qubit case (n=2) with verbose output
        >>> pauli_2 = generate_pauli_group(2, verbose=True)
        II
        IX
        IY
        IZ
        XI
        XX
        XY
        XZ
        YI
        YX
        YY
        YZ
        ZI
        ZX
        ZY
        ZZ
        >>> len(pauli_2)
        16
        >>> pauli_2[0].shape
        (4, 4)
        
    Notes:
        - The function uses the existing pauli_matrices() function for consistency
        - Memory usage scales as O(4^n * 4^n) = O(16^n), so use with caution for large n
        - For n > 4, consider using sparse representations or symbolic methods
        - The ordering follows itertools.product convention: rightmost index varies fastest
        - Verbose output shows Pauli strings in standard notation (I, X, Y, Z)
    """
    # Input validation
    if not isinstance(n, int):
        raise TypeError(f"Number of qubits must be an integer, got {type(n).__name__}")
    
    if n <= 0:
        raise ValueError(f"Number of qubits must be positive, got {n}")
        
    # Use existing pauli_matrices function for consistency
    sigma_x, sigma_y, sigma_z = pauli_matrices()
    identity = np.eye(2, dtype=complex)
    paulis = [identity, sigma_x, sigma_y, sigma_z]
    pauli_names = ['I', 'X', 'Y', 'Z']
    
    # Import itertools.product for generating all combinations
    from itertools import product as itertools_product
    
    pauli_group = []
    
    # Generate all 4^n combinations of Pauli matrices
    for pauli_combination in itertools_product(paulis, repeat=n):
        # Start with the first Pauli matrix
        tensor_product = pauli_combination[0]
        
        # Compute tensor product for remaining matrices
        for i in range(1, n):
            tensor_product = np.kron(tensor_product, pauli_combination[i])
            
        pauli_group.append(tensor_product)
        
        # Print Pauli string name if verbose mode is enabled
        if verbose:
            # Get corresponding names for this combination
            pauli_string = ""
            for pauli_matrix in pauli_combination:
                # Find which Pauli matrix this is by comparing with our list
                for idx, p in enumerate(paulis):
                    if np.array_equal(pauli_matrix, p):
                        pauli_string += pauli_names[idx]
                        break
            print(pauli_string)
    
    return pauli_group


def jordan_wigner_lowering(n_sites):
    """
    Define annihilation operators using the Jordan-Wigner transform.

    Args:
        n_sites: The number of lattice sites

    Returns:
        A list of the annihilation operators
    """
    sigma_x, sigma_y, sigma_z = pauli_matrices()

    annihilation_operators = []

    for i in range(n_sites):
        a_i = 1  # Initialize as scalar

        Sminus = (sigma_x + 1j * sigma_y) / 2

        for j in range(n_sites):
            if j < i:
                a_i = np.kron(a_i, sigma_z)
            elif j == i:
                a_i = np.kron(a_i, Sminus)  # annihilation operator at site i
            else:
                a_i = np.kron(a_i, np.eye(2))

        annihilation_operators.append(a_i)

    return annihilation_operators


def jordan_wigner_alphas(n_sites):
    """
    Define raising and lowering operators using the Jordan-Wigner transform.
    With alphas = raising + lowering

    Args:
        n_sites: The number of lattice sites

    Returns:
        A list of the [raising, lowering] operators
    """
    lowering = jordan_wigner_lowering(n_sites)
    raising = [c.conj().transpose() for c in lowering]

    return raising + lowering


def jordan_wigner_majoranas(n_sites):
    r"""
    Generate a set of Majorana operators under Jordan-Wigner.

    .. math::

      \gamma_j = (a_j + a_j^\dagger)/\sqrt{2}, \qquad j < n_{sites}

      \gamma_k = i(a_k - a_k^\dagger)/\sqrt{2},\qquad  k < 2n_{sites}


    Args:
        n_sites: The number of lattice sites

    Returns:
        A list of the Majorana operators
    """
    annihilation_operators = jordan_wigner_lowering(n_sites)
    majorana_operators = []

    for j in range(n_sites):
        gamma_j = (
            annihilation_operators[j].conj().T + annihilation_operators[j]
        ) / np.sqrt(2)
        majorana_operators.append(gamma_j)

    for k in range(n_sites):
        gamma_k = (
            -1j
            * (annihilation_operators[k] - annihilation_operators[k].conj().T)
            / np.sqrt(2)
        )
        majorana_operators.append(gamma_k)

    return majorana_operators


def rotate_operators(C, alphas, Left=False, verbose=False):
    r"""
    Transform set of operators using a matrix C per:

    .. math:: \beta_j = \sum_i C_{ji} \alpha_i


    Use Left=True if instead you want

    .. math:: \beta_j = \sum_i  \alpha_i C_{ij}^*

    Args:
        C: A [len(alpha) x len(alpha)] coefficient matrix
        alphas: A list of operators (NumPy matrices)
        Left: Do the transformation left-handed per above (default=False)
        verbose: Print more information (default=False)

    Returns:
        The list of rotated operators
    """
    # check if input is out of order and swap as needed
    if isinstance(C, list):
        (C, alphas) = (alphas, C)

    # Validate dimensions
    n = len(alphas)
    if C.shape[0] != n and C.shape[1] != n:
        print("dimensions are incompatible")
        if not (C.shape[0] > n and C.shape[1] > n):
            return 0

    if Left is True:
        C = np.array(C)
        C = C.conj().T
    else:
        C = np.array(C)

    return _perform_rotation(C, alphas, verbose=verbose)


def _perform_rotation(C, alphas, verbose=False):
    """
    Helper function to perform the actual rotation of operators.

    Args:
        C: The transformation matrix
        alphas: List of operators to transform
        verbose: Whether to print detailed steps

    Returns:
        List of transformed operators
    """
    n = len(alphas)
    b_ops = []

    for j in range(n):
        b_j = np.zeros_like(alphas[0])  # initialize jth rotated operator

        if verbose:
            print("\nj", j, "\n----")

        for i in range(n):
            if abs(C[j, i]) < 1e-9:
                continue

            b_j += C[j, i] * alphas[i]  # accumulate

            if verbose:
                print(
                    "b[",
                    j,
                    "] += C[",
                    j,
                    ",",
                    i,
                    "]  alpha_",
                    i,
                    "= ",
                    np.round(C[j, i], 3),
                    " alpha_",
                    i,
                    "\n",
                )

        b_ops.append(b_j)  # save in array

    return b_ops


def build_V(n_sites, A, Z=None):
    r"""
    Build the generator V matrix of dimension 2*N for N sites.

    .. math::
     V=\begin{bmatrix}
     -Z^* & A \\
     -A^* & Z
     \end{bmatrix}

    Args:
        n_sites: The number of sites
        A: The A coefficient matrix, [N x N]
        Z: The Z coefficient matrix, [N x N] (optional)

    Returns:
        A 2*N x 2*N numpy array representing a generator matrix
    """
    if Z is None:
        Z = np.zeros_like(A)

    V = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)

    V[:n_sites, :n_sites] = -Z.conj()
    V[:n_sites, n_sites:] = A

    V[n_sites:, :n_sites] = -A.conj()
    V[n_sites:, n_sites:] = Z

    return V


def build_H(n_sites, A, B=None):
    r"""
    Build the H block coefficient matrix of dimension 2*N for N sites.

    .. math::
     H=\begin{bmatrix}
     -A^* & B \\
     -B^* & A
     \end{bmatrix}

    If A = A.conj().T and B = -B.T then the output is compatible with
    alphas = jordan_wigner_alphas(n_sites)

    Args:
        n_sites: The number of sites, N
        A: The A coefficient matrix, [N x N]
        B: The B coefficient matrix, [N x N] (optional)

    Returns:
        A [2*N x 2*N] numpy array representing the H matrix
    """
    if B is None:
        B = np.zeros_like(A)

    N = n_sites
    H = np.zeros((2 * N, 2 * N), dtype=complex)

    H[:N, :N] = -(A.conj())
    H[:N, N:] = B
    H[N:, :N] = -(B.conj())
    H[N:, N:] = A
    return H


def random_FF_rotation(n_sites, seed=None, returnH=False, returnOrb=False):
    """Generate a random free fermion rotation matrix

    Args:
        n_sites: The number of sites
        seed: Random seed for reproducibility (default: None)
        returnH: If True, return the generator matrix instead (default: False)
        returnOrb: If True, return the orbital rotation matrix C (default: False)

    Returns:
        A random free fermion rotation matrix of dimension 2*N for N sites
    """
    if seed is not None:
        np.random.seed(seed)

    randO = special_ortho_group.rvs(dim=2 * n_sites)

    Omega = build_Omega(n_sites)

    C = Omega.conj().T @ randO @ Omega
    assert is_symp(C), "Generated matrix is not symplectic"

    if returnOrb:
        return C  # Return the orbital rotation matrix

    z = -1j
    h = -logm(C) / z
    H = -h / 2

    G_op = build_op(n_sites, H, jordan_wigner_alphas(n_sites))

    if returnH:
        return G_op  # Compute the unitary operator and return H

    return expm(-1j * G_op)  # Compute the unitary operator


def random_FF_state(n_sites, fixedN=False, seed=None, returnH=False, pure=False):
    """Generate a Haar random free fermion state using random symplectic rotations.

    This function generates free-fermion states that are uniformly distributed
    over the space of free-fermion states using Haar random symplectic transformations.

    Args:
        n_sites: The number of sites
        fixedN: If True, generator commutes with N_op (default: False)
        seed: Random seed for reproducibility (optional)
        returnH: If True, return the generator matrix along with
                          rho (default: False)
        pure: If True, return a pure state (default: False)
    Returns:
        A normalized Haar random free fermion state, rho. If returnH is True, also
        returns the generator matrix H.
    """
    if seed is not None:
        np.random.seed(seed)

    if pure:
        # For pure states, apply Haar random unitary to vacuum state
        W_op = random_FF_rotation(n_sites, seed=seed)
        zero_state = np.zeros((2**n_sites, 1), dtype=complex)
        zero_state[0, 0] = 1  # Set the first element to 1 (ground state)
        psi = W_op @ zero_state

        # check normalization of the pure state
        assert np.allclose(1, np.linalg.norm(psi))

        if returnH:
            # Get the generator that produced this unitary
            H_op = random_FF_rotation(n_sites, seed=seed, returnH=True)
            return psi, H_op
        else:
            return psi

    # For mixed states, use thermal-like distribution with Haar random Hamiltonian
    if fixedN:
        # For particle number preserving case, use simpler approach
        H = random_H_generator(n_sites, fixedN=True, seed=seed)
        H_op = build_op(n_sites, H, jordan_wigner_alphas(n_sites))
    else:
        # Use Haar random rotation to generate the Hamiltonian
        H_op = random_FF_rotation(n_sites, seed=seed, returnH=True)

    # Generate thermal state with random temperature parameter
    beta = np.random.exponential(1.0)  # Random inverse temperature
    rho = expm(-beta * H_op)
    rho = rho / np.trace(rho)  # Normalize the state density matrix

    # Return both the normalized state density matrix and the generator matrix
    if returnH:
        return rho, H_op
    else:
        return rho


def random_H_generator(n_sites, fixedN=False, seed=None):
    """
    Generate a random generator matrix for free fermions. Entries are drawn
    from a standard normal distribution for both real and imaginary parts.

    Args:
        n_sites: The number of sites
        fixedN: If True, generator will preserve N (default: False)
        seed: Random seed for reproducibility (optional)

    Returns:
        A random generator matrix of dimension 2*N for N sites
    """
    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(n_sites, n_sites) + 1j * np.random.randn(n_sites, n_sites)
    Z = np.random.randn(n_sites, n_sites) + 1j * np.random.randn(n_sites, n_sites)
    A = A + A.conj().T  # make A Hermitian
    Z = Z - Z.T  # make Z skew-symmetric

    if fixedN:
        Z = Z * 0  # make Z zero if fixedN is True
    # Build the generator matrix
    return build_H(n_sites, A, Z)


def kitaev_chain(n_sites, mu, t, delta):
    """Create Kitaev chain Hamiltonian."""
    # Chemical potential term
    A = -mu * np.eye(n_sites)

    # Hopping term
    for i in range(n_sites - 1):
        A[i, i + 1] = -t
        A[i + 1, i] = -t

    # Pairing term
    B = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        B[i, i + 1] = delta
        B[i + 1, i] = -delta

    return build_H(n_sites, A, B)


def build_Omega(N):
    r"""
    Build the Omega matrix of dimension 2*N for N sites.

    .. math::
      \Omega = \frac{1}{\sqrt{2}} \begin{bmatrix}
      \mathbf{1} & \mathbf{1} \\
      i \mathbf{1} & -i \mathbf{1}
      \end{bmatrix}

    Args:
        N: The number of sites

    Returns:
        A 2*N x 2*N numpy array representing the Omega matrix
    """
    Omega = np.zeros((2 * N, 2 * N), dtype=complex)
    nu = 1 / (np.sqrt(2))
    Omega[:N, :N] = np.diag(nu * np.ones(N))
    Omega[:N, N:] = np.diag(nu * np.ones(N))
    Omega[N:, :N] = np.diag(1j * nu * np.ones(N))
    Omega[N:, N:] = np.diag(-1j * nu * np.ones(N))
    return Omega


def build_reordering_xx_to_xp(n_sites):
    """
    A permutation matrix that transforms ordering
    [x_1, x_2... p_1, p_2...] to [x_1, p_1, x_2, p_2, ...]

    Args:
        n_sites: Number of sites

    Returns:
        Permutation matrix for reordering
    """
    P_xx_xp = np.zeros((2 * n_sites, 2 * n_sites))
    for i in range(n_sites):
        P_xx_xp[i, 2 * i] = 1
        P_xx_xp[i + n_sites, 2 * i + 1] = 1

    return P_xx_xp


def build_K(n_sites):
    """Build the symplectic transform from standard to canonical eigenstruct"""
    # 1. the permutation P from L_paired to L_std
    P = build_reordering_xx_to_xp(n_sites)

    # 2. the unitary W from L_paired to L_canonical
    # https://math.stackexchange.com/questions/1439552/
    w = np.zeros((2, 2)) * (0j)
    w[0, 0] = 1
    w[1, 1] = -1
    w[0, 1] = -1j
    w[1, 0] = 1j
    w = w / np.sqrt(2)
    W = np.kron(np.eye(n_sites), w)

    # 3. Phase factors to rotate the second set of vectors to U_sympl form
    Phases = np.ones((2 * n_sites,)) * (1 + 0j)
    Phases[n_sites:] = -1j * Phases[n_sites:]
    Phases = np.diag(Phases)

    # all together
    K = Phases @ P @ W @ build_Omega(n_sites)

    return K


def is_symp(U):
    r"""
    Function checks if U is symplectic

    U is symplectic if U S U^T = S

    A known canonical form is:

    .. math::
     U = \begin{bmatrix}
             s & t^* \\
             t & s^*
         \end{bmatrix}

    where s and t are square matrices of dimension N.

    Args:
        U: Matrix to check for symplectic property

    Returns:
        True if U is symplectic, False otherwise
    """
    n_sites = U.shape[0] // 2

    aa = U[:n_sites, :n_sites]
    bb = U[:n_sites, n_sites:]
    cc = U[n_sites:, :n_sites]
    dd = U[n_sites:, n_sites:]

    S = np.zeros_like(U)
    S[:n_sites, n_sites:] = np.eye(n_sites)
    S[n_sites:, :n_sites] = np.eye(n_sites)

    if np.allclose(aa, dd.conj()) and np.allclose(cc, bb.conj()):
        if np.allclose(U @ S @ U.T, S):
            return True
        else:
            print("Error: Canonical form but not symplectic")
    else:
        if np.allclose(U @ S @ U.T, S):
            print("Not canonical but still symplectic")
            return True

    return False


def check_canonical_form(L):
    """
    Check if a matrix is in standard block diagonal form.

    Args:
        L: A NumPy array representing the matrix

    Returns:
        True if the matrix is in standard block diagonal form, False otherwise
    """
    n = L.shape[0] / 2

    if n != int(n):
        print("Odd dimension for input matrix")
        return False
    else:
        n = int(n)

    # Should be square, real, and skew-symmetric
    if (
        L.shape[1] != L.shape[0]
        or not np.allclose(L, L.conj())
        or not np.allclose(L, -L.T)
    ):
        print("Not square, real, and skew-symmetric")
        return False

    # Copy blocks to compare against given matrix
    M = np.zeros_like(L)
    for j in range(2 * n):
        if j % 2:  # odd
            M[j, j - 1] = L[j, j - 1]
        else:  # even
            M[j, j + 1] = L[j, j + 1]

    if not np.allclose(M, L):
        print("Too many non-zero elements")
        return False

    return True


def generate_gaussian_state(n_sites, H, alphas=None):
    """
    Generate a Gaussian state using Hamiltonian H.

    Args:
        n_sites: The number of orbitals
        H: Quadratic parent Hamiltonian ([2n x 2n] or [2**n x 2**n])
        alphas: The list of 2n fermionic operators (optional)

    Returns:
        The [2**n x 2**n] free fermion state
    """
    H_op = None

    if alphas is None:
        alphas = jordan_wigner_alphas(n_sites)

    if H.shape[0] == 2 * n_sites:
        H_op = build_op(n_sites, H, alphas)

    if H.shape[0] == 2**n_sites:
        H_op = H

    if H_op is None:
        print(H.shape)
        print(
            "Invalid Hamiltonian size ",
            H.shape,
            "\nShould be dim: ",
            2 * n_sites,
            " or ",
            2**n_sites,
        )
        return 0

    rho = expm(-H_op)

    return rho / np.trace(rho)


def build_op(n_sites, R, alphas, verbose=None, direct=False):
    """
    Build the N-body lift of a quadratic coefficient matrix

    R̂ = Σ_ij R_ij α_i† α_j

    Args:
        n_sites: The number of sites
        R: Coefficient matrix [2N x 2N]
        alphas: A list of 2N fermionic operators
        verbose: Increases output if set to True or an integer (default: None)
        direct: Do R_ij α_i α_j (no adjoint) (default: False)

    Returns:
        The N-body Hamiltonian operator as a matrix
    """
    if R.shape[0] != 2 * n_sites or R.shape[1] != 2 * n_sites:
        print("Use `build_H(n_sites,A,np.zeros_like(A))`", "to format input correctly")

    # N-body operators
    R_op = np.zeros_like(alphas[0])

    for i in range(2 * n_sites):
        for j in range(2 * n_sites):
            if np.abs(R[i, j]) < 1e-10:
                continue

            if direct:
                R_op += R[i, j] * (alphas[i] @ alphas[j])
            else:
                R_op += R[i, j] * (alphas[i].conj().T @ alphas[j])

            if verbose:
                print("\nR_op so far:")
                print(R_op)

    return R_op


def compute_cov_matrix(rho, n_sites=None, alphas=None):
    """
    Calculate the fermionic covariance matrix of rho

    K_ij = Tr[rho α_i α_j] - Tr[rho α_j α_i]

    Args:
        rho: should be a [2**n_sites x 2**n_sites] matrix
        n_sites: number of sites
        alphas: a set of 2 n_sites of fermionic operators (optional)

    Returns:
        An [2 n_sites x 2 n_sites] covariance matrix
    """
    if n_sites is None:
        N = rho.shape[0]
        n_sites = np.round(np.log2(N))

    if alphas is None:
        alphas = jordan_wigner_alphas(n_sites)

    Covf = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)
    for i in range(2 * n_sites):
        for j in range(2 * n_sites):
            Covf[i, j] = np.trace(rho @ alphas[i] @ alphas[j]) - np.trace(
                rho @ alphas[j] @ alphas[i]
            )

    return Covf


def correlation_matrix(rho):
    """
    Calculates the following two-point correlation matrix

    .. math:: \Gamma = \langle \vec \alpha \vec \alpha ^t \rangle 
    .. math:: \Gamma_{ij} = Tr[\rho \alpha_i \alpha_j]
    
    for JW fermionic operators in the [a^+ a] ordering

    Args:
        rho: should be a [2**n_sites x 2**n_sites] matrix

    Returns:
        An [2 n_sites x 2 n_sites] correlation matrix
    """
    N = rho.shape[0]
    n_sites = int(np.round(np.log2(N)))
    alphas = jordan_wigner_alphas(n_sites)

    return compute_2corr_matrix(rho, n_sites, alphas, conjugation=None)


def compute_2corr_matrix(rho, n_sites, alphas=None, conjugation=None):
    """
    Calculate the two-point correlation matrix

    .. math:: \Gamma = \langle \vec \alpha \vec \alpha ^t \rangle 
    .. math:: \Gamma_{ij} = Tr[\rho \alpha_i \alpha_j]
    
    By changing the input conjugation parameter, this function also computes
    P = ⟨vec \alpha vec α†⟩ (conjugation == T)
    η = ⟨vec \alpha^\dagger vec α^t⟩ (conjugation < 0)

    By default the operators are not conjugated but if
    `conjugation` is True or positive:
                            P_ij = Tr[ρ α_i α_j†]
    And if `conjugation` is negative:
                            η_ij = Tr[ρ α_i† α_j]

    Args:
        rho: should be a [2**n_sites x 2**n_sites] matrix
        n_sites: number of sites
        alphas: a set of 2 n_sites of fermionic operators (optional)
        conjugation: Indicates if the operators should be conjugated (optional)

    Returns:
        An [2 n_sites x 2 n_sites] correlation matrix
    """
    if alphas is None:
        alphas = jordan_wigner_alphas(n_sites)

    G = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)
    for i in range(2 * n_sites):
        for j in range(2 * n_sites):
            if conjugation is None:
                G[i, j] = np.trace(rho @ alphas[i] @ alphas[j])
            elif int(conjugation) < 0:
                G[i, j] = np.trace(rho @ alphas[i].conj().T @ alphas[j])
            else:
                G[i, j] = np.trace(rho @ alphas[i] @ alphas[j].conj().T)

    return G


def compute_algebra_S(alphas, verbose=False):
    """
    Given operators α_i, compute the S matrix

    α_i α_j = S_ij 𝟙 - α_j α_i

    Args:
        alphas: A list of 2N operators
        verbose: Output more information (default: False)

    Returns:
        S: The algebraic S matrix
    """
    n_ops = len(alphas) // 2

    mat0 = np.zeros_like(alphas[0])

    # pull matrix shape from input ops
    Id = np.eye(alphas[0].shape[0])

    S = np.zeros((2 * n_ops, 2 * n_ops), dtype=complex)
    for i in range(2 * n_ops):
        for j in range(2 * n_ops):
            anticomm_ij = alphas[i] @ alphas[j] + alphas[j] @ alphas[i]

            if np.allclose(anticomm_ij, mat0):
                continue

            s_ij = anticomm_ij[0, 0]

            if verbose:
                print("s_ij=", s_ij)
                print("anticomm_ij.shape", anticomm_ij.shape)

            # verify only scalar anti-commutation
            if not np.allclose(s_ij * Id, anticomm_ij):
                print("Error:")
                print(np.diag(anticomm_ij))
                print(s_ij)

            S[i, j] = s_ij

    if verbose:
        print("\nAlgebra properties\n")
        print("Is symmetric?", np.allclose(np.zeros_like(S), S - S.T))
        print("Rank:", np.linalg.matrix_rank(S))
        print("Trace:", np.trace(S))
        evals = np.linalg.eigvals(S)
        evals.sort()
        print("Eigenvalue max,min:", np.min(evals), np.max(evals))

    return S


def is_matchgate(M, verbose=False):
    """
    Check the matchgate condition for 4x4 input matrices.

    A 4x4 matrix satisfies the matchgate condition if the determinant
    of its inner 2x2 submatrix equals the determinant of its corner
    2x2 submatrix.

    Args:
        M: 4x4 numpy array to test
        verbose: Print detailed information (default: False)

    Returns:
        True if M satisfies the matchgate condition, False otherwise
    """
    if M.shape != (4, 4):
        print("Wrong size - input must be 4x4 matrix")
        return False

    # Extract corner elements
    corner_matrix = np.array([[M[0, 0], M[0, 3]], [M[3, 0], M[3, 3]]])

    # Extract inner 2x2 matrix
    inner_matrix = M[1:3, 1:3]

    # Compute determinants
    det_inner = np.linalg.det(inner_matrix)
    det_corner = np.linalg.det(corner_matrix)

    if np.allclose(det_inner, det_corner):
        if verbose:
            print(
                "Satisfies the matchgate condition, det1=det2=", np.round(det_inner, 4)
            )
        return True
    else:
        if verbose:
            print(
                "Not matchgate\n det1=",
                np.round(det_inner, 4),
                "; det2=",
                np.round(det_corner, 4),
            )
        return False


# # Wick's theorem implementation functions
# def wick_contraction(L, Gamma):
#     """
#     Compute Wick contraction for correlation function using pfaffian.

#     Args:
#         L: List of operator indices
#         Gamma: Two-point correlation matrix

#     Returns:
#         Pfaffian of the contracted covariance matrix
#     """
#     # Extract submatrix
#     tmp = Gamma[L, :]
#     g0 = tmp[:, L]

#     # Correct for contractions
#     # Lower part of overlap matrix for L
#     sL = np.tril(g0 + g0.T)

#     # Normal ordered contractions
#     gn = g0 - sL


def eigh_sp(H):
    r"""
    Return the eigenvectors in symplectic form and the eigenvalues of a complex
    Hermitian (conjugate symmetric) FF coefficient array.

    Returns two objects, a 1-D array containing the eigenvalues of H and a
    2-D array containing a symplectic unitary that diagonalizes H.

    This routine transforms to Majorana forms, gets the canonicalizing
    orthogonal matrix for skew-symmetric real matrices. This is transformed to
    symplectic form using

    .. math:: U_{symp} = \Omega^\dagger O \Omega

    Then a symplectic transformation is constructed to transform from canonical
    eigenstructure

    .. math::
        L_{canonical} = \oplus_k
        \begin{pmatrix} 0 & -\lambda_k\\
        \lambda_k & 0 \end{pmatrix}

    to standard eigenstructure

    .. math::
        L_{standard}
        =
        \begin{pmatrix} -\Sigma & 0\\
        0 & \Sigma \end{pmatrix}

    Parameters:
        H: (2N, 2N) array
            Should be of the form
                [[-A.conj(), B],
                 [-B.conj(), A]]
            with :math:`A = A^\dagger` and :math:`B = -B^T`

    Returns:
        A tuple with
        eigenvalues: (2N) ndarray
        eigenvectors: (2N,2N) ndarray
            Ortho-normal eigenvectors arranged in the form
            :math:`\begin{pmatrix} s & t\\t^* & s^* \end{pmatrix}`
    """
    n_sites = H.shape[0] // 2

    # Get Majorana form
    Omega = build_Omega(n_sites)
    Omega_dag = Omega.conj().T

    # Majorana coefficient matrix form
    ih = 1j * Omega @ H @ Omega_dag

    # these follow if the form of the input H was correct
    assert np.allclose(0, ih + ih.T)
    assert np.allclose(ih, ih.real)

    # cast to real assuming the imag part is zero
    ih = ih.real

    # we want the non-zero eigenvalue to appear first
    def sfunction(x, y=None):
        z = x if y is None else x + y * 1j
        return abs(z) > 1e-10

    # imported from scipy.linalg
    [L, Orth, _] = schur(ih, sort=sfunction)

    # format checking
    _print(L)
    assert check_canonical_form(L)
    assert np.allclose(Orth @ Orth.T, np.eye(Orth.shape[0]))
    assert check_canonical_form(Orth.T @ ih @ Orth)
    assert np.allclose(Orth, Orth.real)
    assert np.allclose(Orth @ L @ Orth.T, ih)

    # re-sort Schur output according to A matrix eig decomp
    A = H[n_sites:, n_sites:]

    # if the key matrix exists, sort the blocks according to the key matrix
    # ordering. Swap within blocks if the sign ordering is incorrect.
    # Apply the same permutation to the orthogonal matrix and check that the
    # resulting eigenvalue structure is still canonical when used with ih
    if not np.allclose(A, 0):

        # get eigs from A and L
        evs_a = np.linalg.eigvalsh(A)
        evs_l = []
        for i in range(n_sites):
            evs_l.append(L[2 * i, 2 * i + 1])
        evs_l = np.array(evs_l)

        # the sorting permutations
        perm_A = np.argsort(np.abs(evs_a))
        perm_l = np.argsort(np.abs(evs_l))

        # L eigenvalues sorted in order of A
        evs_l_mag = evs_l[perm_l[perm_A]]

        # flipping L eigenvalues to match sign structure of A
        Flips = np.eye(2 * n_sites)
        for i in range(n_sites):
            if np.allclose(np.sign(evs_l_mag[i]), np.sign(evs_a[i])):
                Flips[2 * i, 2 * i + 1] = 1
                Flips[2 * i + 1, 2 * i] = 1
                Flips[2 * i, 2 * i] = 0
                Flips[2 * i + 1, 2 * i + 1] = 0

        # permutation matrix for blocks
        P_blocks = np.kron(permutation_to_matrix(perm_l[perm_A]), np.eye(2))

        # The permutation matrix for rearrange L
        P = P_blocks @ Flips

        # Update the orthogonal transformation
        Orth = Orth @ P

    # Converting canonicalizing orthogonal matrix to symplectic unitary form
    U_o = Omega_dag @ Orth @ Omega
    K = build_K(n_sites)
    U = U_o @ K.conj().T

    # Set canonical phase structure
    phasesU = (np.diag(U)).conj()
    for i in range(len(phasesU)):
        if abs(phasesU[i]) > 1e-9:
            phasesU[i] = phasesU[i] / abs(phasesU[i])
        else:
            phasesU[i] = 1

    # eigenvector to be returned
    U = U @ np.diag(phasesU)
    # eigenstruct to be returned
    L = U.conj().T @ H @ U

    return [L, U]


def eigv_sp(V):
    r"""
    Returns the left eigenvectors in symplectic form and eigenvalues in Y-form

    .. math::

        L_{Y} =
        \begin{pmatrix} 0 & -\Sigma\\
        \Sigma & 0 \end{pmatrix}

    of the coefficient matrix in V-form whereby
    :math:`\hat V = \hat V^\dagger = \sum V_{ij}\alpha_i \alpha_j`

    Returns two objects, a 1-D array containing the eigenvalues of V in Y-form
    and a 2-D array containing a symplectic unitary that diagonalizes V.

    .. math::
        L_{standard}
        =
        \begin{pmatrix} -\Sigma & 0\\
        0 & \Sigma \end{pmatrix}

    Parameters:
        V: (2N, 2N) array
            Should be of the form
            :math:`V = \begin{pmatrix}-B^* & A\\ -A^* & B\end{pmatrix}`
            with A = A† and B = -B^T

    Returns:
        A tuple with
        eigenvalues: (2N) ndarray in :math:`L_Y` form
        eigenvectors: (2N,2N) ndarray
            Ortho-normal eigenvectors arranged in the form
            :math:`U = \begin{pmatrix} s & t\\ t^* & s^* \end{pmatrix}`
            such that U.T @ V @ U = L_Y
    """
    n_sites = V.shape[0] // 2

    Id = np.eye(n_sites)
    Zero = np.zeros_like(Id)

    S = np.block([[Zero, Id], [Id, Zero]])

    H = S @ V

    [L, U] = eigh_sp(H)

    return [S @ L, U]


def eigm_sp_can(G):
    r"""
    Returns the orthogonal eigenvectors and the eigenvalues in direct sum
    canonical form for the Majorana coefficient matrix G.

    G should be of form
    .. math::
        G =  i\begin{bmatrix}Im(A+Z) & Re(A+Z)\\
        Re(Z-A)&  Im(A-Z)
        \end{bmatrix}

    where Z = -Z^T and A = A^dagger.

    Returns two objects, the eigenvalues of `G` in canonical form and
    the sympletic orthogonal matrix that transforms `G` to canonical
    eigenstructure form
    .. math::
        L_{canonical} = \oplus_k
        \begin{matrix} 0 & -\lambda_k\\
        \lambda_k & 0 \end{matrix}

    Parameters
    ----------
    G : ( 2N, 2N ) array properly formatted

    Returns
    -------
    A tuple with

    eigenvalues:  (2N) ndarray
        Lc = \oplus_k [[   0  ,  -ev_k ],
                      [ ev_k ,    0   ]]
    eigenvectors: (2N,2N) ndarray
        Ortho-normal eigenvectors arranged in the form
        U = [[  s  ,  t  ]
             [ t^* , s^* ]]
    """
    n_sites = G.shape[0] // 2

    assert np.allclose(G, 1j * G.imag)

    iG = -G.imag

    # from scipy.linalg
    [L, O] = schur(iG)

    # we will sort the states according to eigenvalues of A

    # pull A
    Omega = build_Omega(n_sites)
    H = Omega.conj().T @ G @ Omega
    A = H[n_sites:, n_sites:]

    if not np.allclose(A, 0):
        evs_a = np.linalg.eigvalsh(A)

        # sort eigenvalues by decreasing eigenvalue
        p = np.flip(np.argsort(abs(evs_a)))

        # sort eigenvalues
        evs_a_p = evs_a[p]

        # # Build permutation matrix according to sign of list
        P_mat = None

        id = np.eye(2)
        x = np.matrix([[0, 1], [1, 0]])

        if evs_a_p[0] > 0:
            P_mat = x
        else:
            P_mat = id

        for k in range(1, n_sites):
            if evs_a_p[k] > 0:
                P_mat = kron_plus(P_mat, x)
            else:
                P_mat = kron_plus(P_mat, id)

        # resort eigenspaces
        O = O @ P_mat

    L = O.T @ G @ O

    return [L, O]


def eigm_sp(G):
    r"""
    Returns the orthogonal eigenvectors and the eigenvalues in L_Y form for the
    Majorana coefficient matrix G. G should be of form

    .. math::
        G =  i\begin{bmatrix}Im(A+Z) & Re(A+Z)\\
        Re(Z-A)&  Im(A-Z)
        \end{bmatrix}

    where :math:`Z = -Z^T` and :math:`A = A^\dagger`.

    Returns two objects, the eigenvalues of `G` in canonical form and
    the sympletic orthogonal matrix that transforms `G` to the Y
    eigenstructure form

    .. math::

        L_{Y} =
        \begin{pmatrix} 0 & -\Sigma\\
        \Sigma & 0 \end{pmatrix}

    Parameters
    ----------
    G : ( 2N, 2N ) array properly formatted

    Returns
    -------
    A tuple with

    eigenvalues:  ( 2N ) ndarray
        i*L in L_Y form
    eigenvectors: ( 2N, 2N ) ndarray
        Orthogonal matrix such that  O^T O = Id
    """
    n_sites = G.shape[0] // 2

    # convert to standard form
    Omega = build_Omega(n_sites)
    H = Omega.conj().T @ G @ Omega

    [l, U] = eigh_sp(H)

    # convert symplectic unitary to orthogonal
    Orth = Omega @ U @ Omega.conj().T

    # compute the canonical form
    L = Orth.T @ G @ Orth

    return [L, Orth]
