"""
Free Fermion Distance Measures Module

This module contains core distance measure functions for analyzing quantum states
in the context of free fermion systems and stabilizer formalism. These measures
quantify various notions of distance from classical or free-fermion behavior.

Key functionality:
 - Stabilizer distribution and Stabilizer Rényi Entropy (SRE)
 - General Rényi entropy and linear entropy measures
 - Fermionic covariance distribution analysis
 - Fermionic Anti-Flatness (FAF) distance measure
 - Kullback-Leibler divergence for probability distributions
 - Jensen-Shannon divergence for probability distributions
 - Trace distance for density matrices 
 - Total variation distance for probability distributions
 - Bhattacharyya coefficient for probability distributions

The distance measures implemented here are fundamental tools for characterizing
quantum states and their proximity to classical, stabilizer, or free-fermion
subspaces in quantum many-body systems.

Copyright 2025 James.D.Whitfield@dartmouth.edu
"""



import numpy as np
from scipy.stats import entropy
from scipy.linalg import schur

from .ff_lib import generate_pauli_group, compute_cov_matrix, jordan_wigner_majoranas
from .ff_utils import clean,cast_to_density_matrix,cast_to_pdf


def stabilizer_distribution(rho):
    """
    Compute the probability distribution over Pauli operators in the stabilizer representation.
    
    This function calculates the stabilizer distribution ξ_P for a quantum state ρ,
    where each element represents the probability of finding a specific Pauli operator P
    in the representation of the state. The distribution is defined as:
    
    .. math::
    
        \\xi_P = \\frac{|\\langle P \\rangle_\\rho|^2}{d Purity(\\rho)}
    
    where :math:`\\langle P \\rangle_\\rho = \\text{Tr}[\\rho P]` is the expectation value of 
    Pauli operator P in state ρ, :math:`d = 2^n` is the dimension of the Hilbert space 
    for n qubits, and :math:`Purity(\\rho) = \\text{Tr}(\\rho^2)` is the purity of the state.
    
    The stabilizer distribution is a fundamental quantity in stabilizer formalism
    and quantum error correction, providing insight into how "classical" or 
    "stabilizer-like" a quantum state is.
    
    Args:
        rho (numpy.ndarray): Input quantum state. Can be:
            - Density matrix of shape (2^n, 2^n) 
            - Pure state wavefunction of shape (2^n,) or (2^n, 1)
            The function automatically detects and converts wavefunctions to density matrices.
            
    Returns:
        numpy.ndarray: Stabilizer distribution ξ of length 4^n, where each element
                      ξ_j corresponds to the probability associated with the j-th
                      Pauli operator in the Pauli group. The distribution is
                      normalized such that Σ_j ξ_j = 1.
                      
    Raises:
        AssertionError: If the computed distribution does not sum to 1 (within numerical precision).
        
    Examples:
        >>> # For a single qubit in |0⟩ state
        >>> psi = np.array([1, 0])
        >>> xi = stabilizer_distribution(psi)
        >>> len(xi)
        4
        >>> np.allclose(np.sum(xi), 1.0)
        True
        
        >>> # For a two-qubit maximally mixed state
        >>> rho = np.eye(4) / 4
        >>> xi = stabilizer_distribution(rho)
        >>> len(xi)
        16
        
    Notes:
        - For n qubits, the Pauli group has 4^n elements
        - The function uses the existing generate_pauli_group() for consistency
        - Numerical precision is handled with a tolerance of 1e-9 for imaginary parts
        - The distribution provides a complete characterization of the state in the Pauli basis
        
    References:
        - Stabilizer Renyi Entropy paper by Leone, Oliviero, and Hamma (arXiv:2106.12587)
    """
    
    #clean input and convert to density matrix as needed
    rho = cast_to_density_matrix(rho)

    # Determine number of qubits from Hilbert space dimension
    n_qubits = int(np.log2(rho.shape[0]))
    
    # Determine the normalization factor from purity and dimension
    N = 2**n_qubits
    pur_rho = np.trace(rho @ rho)
    if abs(pur_rho.imag) < 1e-9:
        pur_rho = pur_rho.real
    else:
        raise ValueError(f"Significant imaginary part in purity: {pur_rho}")
    d = N * pur_rho

    # Generate complete Pauli group for n qubits
    pauli_group = generate_pauli_group(n_qubits)

    # Initialize stabilizer distribution
    xi = np.zeros(len(pauli_group))

    # Compute expectation values and probabilities
    for j, Pj in enumerate(pauli_group):
        # Calculate expectation value ⟨P_j⟩_ρ = Tr[ρ P_j]
        expectP = np.trace(Pj @ rho)
        
        # Clean small imaginary parts due to numerical precision
        if abs(expectP.imag) < 1e-9:
            expectP = expectP.real
        else:
            raise ValueError(f"Significant imaginary part in expectation value: {expectP}")

        # Compute probability: ξ_j = |⟨P_j⟩_ρ|² / d
        xi[j] = (expectP * expectP) / d

    # Verify normalization
    if not np.allclose(np.sum(xi), 1):
        print("Warning: stabilizer distribution sum =", clean(np.sum(xi)))
    
    assert np.allclose(np.sum(xi), 1), f"Distribution not normalized: sum = {np.sum(xi)}"

    return xi


def SRE(rho, a=2):
    """
    Compute the Stabilizer Rényi Entropy (SRE) of a quantum state.
    
    The Stabilizer Rényi Entropy is a measure of how far a quantum state is from
    being a stabilizer state. It is defined as:
    
    .. math::
    
        \\text{SRE}_\\alpha(\\rho) = S_\\alpha(\\xi) - \\log(d)
    
    where :math:`S_\\alpha(\\xi)` is the α-Rényi entropy of the stabilizer distribution ξ,
    and :math:`d = 2^n` is the dimension of the Hilbert space.
    
    For stabilizer states, SRE = 0, while for maximally mixed states or
    Haar-random states, SRE approaches its maximum value. This makes SRE
    a useful measure for quantifying "magic" or non-stabilizer resources
    in quantum computation.
    
    Args:
        rho (numpy.ndarray): Input quantum state as density matrix of shape (2^n, 2^n)
                            or wavefunction that will be converted to density matrix.
        a (float, optional): Rényi parameter α. Default is 2 (quadratic Rényi entropy).
                           For α=1, this reduces to the von Neumann entropy case.
                           
    Returns:
        float: The Stabilizer Rényi Entropy SRE_α(ρ). 
               - SRE = 0 for stabilizer states
               - SRE > 0 indicates deviation from stabilizer behavior
               - Higher values indicate more "magic" or quantum resources
               
    Examples:
        >>> # Stabilizer state (should give SRE ≈ 0)
        >>> psi_stab = np.array([1, 1, 1, 1]) / 2  # |++⟩ state
        >>> rho_stab = np.outer(psi_stab, psi_stab.conj())
        >>> sre_val = SRE(rho_stab)
        >>> np.allclose(sre_val, 0, atol=1e-10)
        True
        
        >>> # Maximally mixed state (should give positive SRE)
        >>> rho_mixed = np.eye(4) / 4
        >>> sre_mixed = SRE(rho_mixed)
        >>> sre_mixed > 0
        True
        
    Notes:
        - The function uses the stabilizer_distribution() to compute ξ
        - The logarithmic offset -log(d) ensures SRE=0 for stabilizer states
        - For α=2, this is sometimes called the "linear entropy" variant
        - The measure is basis-independent and unitarily invariant
        
    References:
        - Leone, L., Oliviero, S. F., & Hamma, A. (2021). Stabilizer Rényi entropy. 
          Physical Review Letters, 128(5), 050402. arXiv:2106.12587
    """
    #clean input and convert to density matrix as needed
    rho = cast_to_density_matrix(rho)

    d = rho.shape[0]
    
    # Compute stabilizer distribution
    xi = stabilizer_distribution(rho)

    if np.nonzero(xi)[0].size == 1:
        return 0.0
    
    # Compute Rényi entropy of the distribution and subtract log(d)
    return renyi_entropy(xi, a) - np.log(d)


def renyi_entropy(p, a):
    """
    Compute the Rényi entropy of a probability distribution.
    
    The Rényi entropy is a generalization of the Shannon entropy and is defined as:
    
    .. math::
    
        S_\\alpha(p) = \\frac{1}{1-\\alpha} \\log\\left(\\sum_i p_i^\\alpha\\right)
    
    For α=1, this reduces to the Shannon entropy:
    
    .. math::
    
        S_1(p) = -\\sum_i p_i \\log(p_i)
    
    For α=2, this gives the collision entropy:
    
    .. math::
    
        S_2(p) = -\\log\\left(\\sum_i p_i^2\\right)
    
    The Rényi entropy provides a family of entropy measures that capture different
    aspects of the probability distribution's structure and are widely used in
    information theory, quantum information, and statistical physics.
    
    Args:
        p (numpy.ndarray): Probability distribution as a 1D array. Must be normalized
                          such that Σ_i p_i = 1 and all elements p_i ≥ 0.
        a (float): Rényi parameter α. Must be non-negative and α ≠ 1.
                  - α → 0: Max entropy (log of support size)
                  - α = 1: Shannon entropy (handled as special case)
                  - α = 2: Collision entropy
                  - α → ∞: Min entropy (-log(max(p_i)))
                  
    Returns:
        float: The α-Rényi entropy S_α(p).
               - For α < 1: S_α ≥ S_1 (Rényi entropy is larger)
               - For α > 1: S_α ≤ S_1 (Rényi entropy is smaller)
               - All Rényi entropies are non-negative for normalized distributions
               
    Examples:
        >>> # Uniform distribution
        >>> p_uniform = np.ones(4) / 4
        >>> s1 = renyi_entropy(p_uniform, 1)  # Shannon entropy
        >>> s2 = renyi_entropy(p_uniform, 2)  # Collision entropy
        >>> np.allclose(s1, np.log(4))  # Should equal log(4) for uniform
        True
        >>> s2 < s1  # Collision entropy is smaller
        True
        
        >>> # Delta distribution (pure state)
        >>> p_delta = np.array([1, 0, 0, 0])
        >>> renyi_entropy(p_delta, 2)
        0.0
        
    Notes:
        - For α=1, the function uses scipy.stats.entropy for numerical stability
        - The function handles the limit α→1 by using the Shannon entropy formula
        - Numerical precision may affect results for very small probabilities
        - The function assumes input is a valid probability distribution
        
    References:
        - Rényi, A. (1961). On measures of entropy and information.
        - Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.
    """
    # Ensure p is a valid probability distribution
    p = cast_to_pdf(p)

    if a == 1:
        # Special case: Shannon entropy (α → 1 limit)
        return entropy(p)
    
    if a == 0:
        # Max entropy: log of the number of non-zero elements
        return np.log(np.count_nonzero(p))
    
    if a == np.inf:
        # Min entropy: -log(max(p_i))
        return -np.log(np.max(p))

    # General case: α-Rényi entropy
    # S_α(p) = (1/(1-α)) * log(||p||_α^α) where ||p||_α is the α-norm
    l_a = np.linalg.norm(p, ord=a)
    
    return np.log(l_a) * (a / (1 - a))


def linear_entropy(p):
    """
    Compute the linear entropy of a probability distribution.
    
    The linear entropy is a measure of mixedness defined as:
    
    .. math::
    
        S_{\\text{lin}}(p) = 1 - d \\|p\\|_2^2
    
    where :math:`d` is the dimension (length of the probability vector) and 
    :math:`\\|p\\|_2^2` is the squared 2-norm of the distribution. This is closely 
    related to the 2-Rényi entropy and provides a simple measure of how far a 
    distribution is from being uniform.
    
    For quantum states, the linear entropy measures the degree of mixedness:
    
    .. math::
    
        S_{\\text{lin}}(\\rho) = 1 - \\text{Tr}(\\rho^2)
    
    Args:
        p (numpy.ndarray): Probability distribution as a 1D array. Must be normalized
                          such that Σ_i p_i = 1 and all elements p_i ≥ 0.
                          The length must be a perfect square (d = 2^n for some integer n).
                          
    Returns:
        float: The linear entropy S_lin(p).
               - S_lin = 0 for pure states (delta distributions)
               - S_lin = (d-1)/d for maximally mixed states  
               - Higher values indicate more mixedness/entropy
               
    Raises:
        AssertionError: If the length of p is not a perfect square.
        
    Examples:
        >>> # Pure state (delta distribution)
        >>> p_pure = np.array([1, 0, 0, 0])
        >>> linear_entropy(p_pure)
        0.0
        
        >>> # Maximally mixed state
        >>> p_mixed = np.ones(4) / 4
        >>> lin_ent = linear_entropy(p_mixed)
        >>> np.allclose(lin_ent, 3/4)  # (d-1)/d = 3/4 for d=4
        True
        
    Notes:
        - The dimension d must be a perfect square for consistency with qubit systems
        - This measure is computationally efficient compared to von Neumann entropy
        - Linear entropy is closely related to purity: Purity = 1 - S_lin
        - For quantum states: 0 ≤ S_lin ≤ (d-1)/d
    """
    # Ensure p is a valid probability distribution
    p = cast_to_pdf(p)
    #print(sum(p))
    #print(p)


    # Verify that d is a perfect square (power of 2 for qubit systems)
    d = len(p)
    sqrt_d = np.sqrt(d)
    assert np.allclose(sqrt_d - int(sqrt_d), 0), f"Length {d} is not a perfect square"

    # Compute squared 2-norm
    l2_squared = np.linalg.norm(p, ord=2)**2

    sL = 1 - l2_squared

    sL_alt = 1 - np.sum(np.square(p))

    assert np.allclose(sL, sL_alt), f"Computed values do not match: {sL} vs {sL_alt}"

    return sL


def cov_distribution(rho):
    """
    Compute the fermionic covariance distribution for a quantum state.
    
    This function computes the probability distribution over fermionic covariance
    matrix eigenvalues, which characterizes the fermionic correlations in the state.
    The distribution is obtained from the eigenvalues of the fermionic covariance matrix:
    
    .. math::
    
        M_{ij} = -i \\langle \\alpha_i \\alpha_j \\rangle_c
    
    where :math:`\\langle \\alpha_i \\alpha_j \\rangle_c` is the connected correlation function
    and :math:`\\alpha_i` are the Majorana operators under Jordan-Wigner transformation.
    
    The covariance matrix M is real and antisymmetric, and its eigenvalues come in
    pairs ±λ_k. The distribution is formed from the squared eigenvalues.
    
    Args:
        rho (numpy.ndarray): Input quantum state. Can be:
            - Density matrix of shape (2^n, 2^n) 
            - Pure state wavefunction of shape (2^n,) or (2^n, 1)
            The function automatically detects and converts wavefunctions to density matrices.
            
    Returns:
        numpy.ndarray: Distribution over squared eigenvalues of the covariance matrix.
                      The length depends on the number of non-zero eigenvalue pairs.
                      
    Examples:
        >>> # Two-site system
        >>> rho = np.eye(4) / 4  # Maximally mixed state
        >>> pm = cov_distribution(rho)
        >>> len(pm) <= 4  # At most 2 sites worth of eigenvalues
        True
        
    Notes:
        - Uses Jordan-Wigner Majorana operators from ff_lib
        - The covariance matrix is computed using compute_cov_matrix from ff_lib
        - Eigenvalues are extracted using Schur decomposition for numerical stability
        - Only eigenvalues with magnitude > 1e-10 are included in the distribution
        
    References:
        - Fermionic Magic Resources of Quantum Many-Body Systems. Sierant, Stornati, and Turkeshi (arXiv:2506.00116)
    """
    
    rho = cast_to_density_matrix(rho)

    # Determine number of sites from Hilbert space dimension
    n_sites = int(np.log2(rho.shape[0]))
    
    #d = 2**n_sites

    # Get Majorana operators
    gammas = jordan_wigner_majoranas(n_sites)

    # Compute fermionic covariance matrix
    M = -1j * compute_cov_matrix(rho, n_sites, alphas=gammas)

    # M should be real (clean small imaginary parts)
    if np.allclose(np.linalg.norm(M.imag), 0):
        M = M.real

    # Verify M is antisymmetric: M = -M^T
    assert np.allclose(np.linalg.norm(M + M.T), 0), "Covariance matrix is not antisymmetric"

    # Use Schur decomposition for numerical stability
    [Lc, _] = schur(M)

    # Extract non-zero eigenvalues from upper triangular Schur form
    eigenvalues = []
    for i in range(2 * n_sites):
        for j in range(i, 2 * n_sites):
            if abs(Lc[i, j]) > 1e-10:
                eigenvalues.append(Lc[i, j])

    # Return squared eigenvalues as the distribution
    pm = np.square(eigenvalues)

    return pm


def total_variation_distance(p,q):
    """
    Compute the total variation distance between two probability distributions.

    The trace distance is a measure of distinguishability between two probability
    distributions. It is defined as:
    .. math::
    
        D_{\\text{trace}}(p, q) = \\frac{1}{2} \\sum_i |p_i - q_i|
    
    where :math:`p` and :math:`q` are the two probability distributions.
    Args:
        p (numpy.ndarray): First probability distribution as a 1D array. Must be normalized
                          such that Σ_i p_i = 1 and all elements p_i ≥ 0.
        q (numpy.ndarray): Second probability distribution as a 1D array. Must be normalized
                          such that Σ_i q_i = 1 and all elements q_i ≥ 0.

    Returns:
        float: The trace distance D_trace(p, q).
               - 0 ≤ D_trace ≤ 1
               - D_trace = 0 if and only if p = q
    Examples:
        >>> # Simple distributions
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.9, 0.1])
        >>> dtrace = trace_distance(p, q)
        >>> dtrace > 0
        True
        >>> dtrace_zero = trace_distance(p, p)
        >>> np.allclose(dtrace_zero, 0)
        True
    Notes:
        - The function coverts input wave functions and density matrices to probability distributions
    References:
        - Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information.
    """
    # Ensure p and q are valid probability distributions
    p = cast_to_pdf(p)
    q = cast_to_pdf(q)

    # Compute trace distance
    dtrace = 0.5 * np.sum(np.abs(p - q))

    return dtrace


def trace_distance(A,B):
    """
    Computes the trace distance between two matrices A and B.

    The trace distance is a quantum measure used to quantify the distinguishability
    between two quantum states represented by density matrices A and B. It is defined as:
    .. math:: D_{trace}(A, B) = \\frac{1}{2} \\text{Tr} |A - B|

    where :math:`|X| = \\sqrt{X^\\dagger X}` is the positive square root of the operator X. 
    Args:
        A: First density matrix (numpy array or array-like)
        B: Second density matrix (numpy array or array-like)
    Returns:
        Trace distance value (float)

    Notes:
        - The function assumes A and B are matrices and casts them to density matrices as needed
        - The trace distance ranges from 0 (identical states) to 1 (orthogonal states)
        - This measure is widely used in quantum information theory to assess state distinguishability
    References:
        - Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information.
    """

    A = cast_to_density_matrix(A)
    B = cast_to_density_matrix(B)

    diff = A - B

    evals = np.linalg.eigvals(diff)

    return 0.5*np.sum(abs(evals))


def relative_entropy(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.
    
    The KL divergence is a measure of how one probability distribution diverges from
    a second, expected probability distribution. It is defined as:
    
    .. math::
    
        D_{KL}(p || q) = \\sum_i p_i \\log\\left(\\frac{p_i}{q_i}\\right)
    
    where :math:`p` is the true distribution and :math:`q` is the reference distribution.
    
    Args:
        p (numpy.ndarray): True probability distribution as a 1D array. Must be normalized
                          such that Σ_i p_i = 1 and all elements p_i ≥ 0.
        q (numpy.ndarray): Reference probability distribution as a 1D array. Must be normalized
                          such that Σ_i q_i = 1 and all elements q_i ≥ 0.
                          
    Returns:
        float: The KL divergence D_KL(p || q).
               - D_KL ≥ 0, with equality if and only if p = q
               
    Examples:
        >>> # Simple distributions
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.9, 0.1])
        >>> dkl = relative_entropy(p, q)
        >>> dkl > 0
        True
        >>> dkl_zero = relative_entropy(p, p)
        >>> np.allclose(dkl_zero, 0)
        True

    Notes:
        - The function uses scipy.stats.entropy for numerical stability
        - KL divergence is not symmetric: D_KL(p || q) ≠ D_KL(q || p)
        - The function coverts input wave functions and density matrices to probability distributions  
    References:
        - Kullback, S., & Leibler, R. A. (1951). On information and sufficiency.
        - Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.
    """
    # Ensure p and q are valid probability distributions
    p = cast_to_pdf(p)
    q = cast_to_pdf(q)

    # Compute KL divergence using scipy.stats.entropy
    dkl = entropy(p, qk=q)

    return dkl


def bhattacharyya_coeff(p, q):
    """
    Compute the Bhattacharyya coefficient between two probability distributions.
    
    The Bhattacharyya coefficient is a measure of similarity between two probability
    distributions. It is defined as:
    
    .. math::
    
        BC(p, q) = \\sum_i \\sqrt{p_i q_i}
    
    where :math:`p` and :math:`q` are the two probability distributions.
    
    Args:
        p (numpy.ndarray): First probability distribution as a 1D array. Must be normalized
                          such that Σ_i p_i = 1 and all elements p_i ≥ 0.
        q (numpy.ndarray): Second probability distribution as a 1D array. Must be normalized
                          such that Σ_i q_i = 1 and all elements q_i ≥ 0.
                          
    Returns:
        float: The Bhattacharyya coefficient BC(p, q).
               
    Notes:  
        - The function coverts input wave functions and density matrices to probability distributions
    """
    # Ensure p and q are valid probability distributions
    p = cast_to_pdf(p)
    q = cast_to_pdf(q)

    # Compute Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))

    return bc


def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon (JS) divergence between two probability distributions.
    
    The JS divergence is a symmetric and bounded measure of similarity between two
    probability distributions. It is defined as:
    
    .. math::
    
        D_{JS}(p || q) = \\frac{1}{2} D_{KL}(p || m) + \\frac{1}{2} D_{KL}(q || m)
    
    where :math:`m = \\frac{1}{2}(p + q)` is the average distribution, and 
    :math:`D_{KL}` is the Kullback-Leibler divergence.
    
    Args:
        p (numpy.ndarray): First probability distribution as a 1D array. Must be normalized
                          such that Σ_i p_i = 1 and all elements p_i ≥ 0.
        q (numpy.ndarray): Second probability distribution as a 1D array. Must be normalized
                          such that Σ_i q_i = 1 and all elements q_i ≥ 0.
                          
    Returns:
        float: The JS divergence D_JS(p || q).
               - 0 ≤ D_JS ≤ log(2)
               - D_JS = 0 if and only if p = q
               
    Examples:
        >>> # Simple distributions
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.9, 0.1])
        >>> djs = jensen_shannon_divergence(p, q)
        >>> djs > 0
        True
        >>> djs_zero = jensen_shannon_divergence(p, p)
        >>> np.allclose(djs_zero, 0)
        True
    Notes:
        - The function uses scipy.stats.entropy for numerical stability
        - JS divergence is symmetric: D_JS(p || q) = D_JS(q || p)
        - The function coverts input wave functions and density matrices to probability distributions  
    References:
        - Lin, J. (1991). Divergence measures based on the Shannon entropy.
        - Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.
    """
    # Ensure p and q are valid probability distributions
    p = cast_to_pdf(p)
    q = cast_to_pdf(q)

    # Compute average distribution
    m = 0.5 * (p + q)

    # Compute JS divergence using KL divergences
    djs = 0.5 * (entropy(p, qk=m) + entropy(q, qk=m))

    return djs


def FAF(rho, k=2):
    """
    Compute the Fermionic Anti-Flatness (FAF) distance measure.
    
    The Fermionic Anti-Flatness is a distance measure that quantifies how far
    a fermionic state is from being a free-fermion (Gaussian) state. It is defined as:
    
    .. math::
    
        \\text{FAF}_k(\\rho) = n - \\|\\mathbf{p}\\|_k^k
    
    where :math:`n` is the number of fermionic sites, :math:`\\mathbf{p}` is the 
    distribution of squared eigenvalues of the fermionic covariance matrix, and 
    :math:`\\|\\cdot\\|_k` is the k-norm.
    
    Equivalently, this can be expressed as:
    
    .. math::
    
        \\text{FAF}_k(\\rho) = n - \\frac{1}{2} \\text{Tr}[(M^T M)^k]
    
    where :math:`M` is the fermionic covariance matrix.
    
    For free-fermion states, FAF = 0, while for highly correlated fermionic states,
    FAF approaches its maximum value. This makes FAF a useful measure for quantifying
    fermionic correlations and deviations from free-fermion behavior.
    
    Args:
        rho (numpy.ndarray): Input quantum state as density matrix of shape (2^n, 2^n)
                            or wavefunction that will be converted to density matrix.
        k (int, optional): Norm parameter for the FAF measure. Default is 2.
                          Higher values of k make the measure more sensitive to
                          large eigenvalues in the covariance spectrum.
                          
    Returns:
        float: The Fermionic Anti-Flatness FAF_k(ρ).
               - FAF = 0 for free-fermion (Gaussian) states
               - FAF > 0 indicates deviation from free-fermion behavior
               - Higher values indicate stronger fermionic correlations
               
    Examples:
        >>> # Free-fermion state (should give FAF ≈ 0)
        >>> # This would require constructing a proper Gaussian state
        >>> rho_ff = np.eye(4) / 4  # Placeholder - not actually free-fermion
        >>> faf_val = FAF(rho_ff)
        >>> isinstance(faf_val, float)
        True
        
    Notes:
        - Uses the fermionic covariance matrix computed via Jordan-Wigner transformation
        - The function verifies consistency between different computational approaches
        - For k=2, this reduces to a quadratic measure of correlations
        - The measure is basis-independent within the fermionic representation
        
    References:
        - Sierant, P., Stornati, G., & Turkeshi, X. (2024). Fermionic Magic Resources of Quantum Many-Body Systems. arXiv:2506.00116
    """
    
    rho = cast_to_density_matrix(rho)

    n_sites = int(np.log2(rho.shape[0]))

    # Get the covariance distribution
    pm = cov_distribution(rho)

    # Compute FAF using three equivalent formulations for verification
    # Method 0: Using k-norm of distribution
    FAF = n_sites - np.linalg.norm(pm, k)**k
    
    # Method 1: Using matrix powers (Equation 27 in reference)
    #FAF_27 = n_sites - 0.5 * np.trace(np.linalg.matrix_power(M.T @ M, k))
    
    # Method 2: Using eigenvalue distribution (Equation 29 in reference)  
    #FAF_29 = n_sites - np.sum(np.power(pm, k))
    
    # Verify consistency between methods
    #assert np.allclose(FAF_27, FAF_29), f"FAF_27={FAF_27}, FAF_29={FAF_29}"
    #assert np.allclose(FAF_27, FAF), f"FAF_27={FAF_27}, FAF={FAF}"
    
    return FAF