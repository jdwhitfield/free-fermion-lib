"""
Free Fermion Combinatorial Functions Module

Core combinatorial matrix functions including determinants, pfaffians,
permanents, and hafnians.

Copyright 2025 James.D.Whitfield@dartmouth.edu
Licensed under MIT License.
"""

import itertools
import math

import numpy as np


def sgn(permutation):
    """Computes the sign of a permutation by counting inversions.

    Args:
        permutation: List or array representing a permutation

    Returns:
        1 if even number of inversions, -1 if odd
    """
    sign = 1
    inversions = 0
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if permutation[i] > permutation[j]:
                inversions += 1
    sign = 1 if inversions % 2 == 0 else -1
    return sign


def pf(A):
    r"""Computes pfaffian via combinatorial formula:

    Given :math:`A` of dimension :math:`N = 2 n_e` we have the pfaffian as:

    .. math::
        pf(A)= \sum_{\sigma\in S_N}^{N!} sgn(\sigma) W_\sigma(A)/(2^{n_e} n_e!)

    Where the weight of matching :math:`\sigma` is given by

    .. math:: W_\sigma(A) = \prod_{i=0}^{n_e} A_{\sigma(2i), \sigma(2i+1)}

    Args:
        A: Square matrix (numpy array or array-like)

    Returns:
        Pfaffian value (complex or real)
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    n_verts = A.shape[0]
    if n_verts % 2 != 0:
        return 0

    # number of edges in a perfect match
    n_edges = n_verts // 2

    perms = itertools.permutations(range(n_verts))
    pf_sum = 0

    for perm in perms:
        prod = 1
        for i in range(n_edges):
            prod *= A[perm[2 * i], perm[2 * i + 1]]
        pf_sum += sgn(perm) * prod

    return pf_sum / (2 ** (n_edges) * math.factorial(n_edges))


def hf(A):
    r"""Computes hafnian via combinatorial formula.

    Given :math:`A` of even dimension :math:`N = 2 n_e` we have the hafnian as:

    .. math:: hf(A) =  \sum_{\sigma\in S_N}^{N!} W_\sigma(A)/(2^{n_e} n_e!)


    Where the weight of matching :math:`\sigma` is given by

    .. math:: W_\sigma(A) = \prod_{i=0}^{n_e} A_{\sigma(2i), \sigma(2i+1)}

    Args:
        A: Square matrix (numpy array or array-like)

    Returns:
        Hafnian value (complex or real)
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    n_verts = A.shape[0]
    if n_verts % 2 != 0:
        return 0

    # number of edges in a perfect match
    n_edges = n_verts // 2

    perms = itertools.permutations(range(n_verts))
    hf_sum = 0

    for perm in perms:
        prod = 1
        for i in range(n_edges):
            prod *= A[perm[2 * i], perm[2 * i + 1]]
        hf_sum += prod

    return hf_sum / (2 ** (n_edges) * math.factorial(n_edges))


def pt(A):
    r"""Computes permanent via combinatorial formula.

    Given :math:`A` of dimension :math:`N` we have the permanent as:

    .. math:: pt(A) = \sum_{\sigma\in S_N}^{N!} B_\sigma(A)

    Where the bipartite matching of :math:`\sigma` is given by

    .. math:: B_\sigma(A) = \prod_{i=1}^N A_{i,\sigma(i)}

    Args:
        A: Square matrix (numpy array or array-like)

    Returns:
        Permanent value (complex or real)
    """
    A = np.matrix(A)
    n = len(A)

    perms = itertools.permutations(range(n))
    pt_sum = 0

    for perm in perms:
        prod = 1
        for i in range(n):
            prod *= A[i, perm[i]]
        pt_sum += prod

    return pt_sum


def dt(A):
    r"""Computes determinant via combinatorial formula.

    Given :math:`A` of dimension :math:`N` we have the determinant as:

    .. math:: dt(A) = \sum_{\sigma\in S_N}^{N!} sgn(\sigma) B_\sigma(A)

    Where the bipartite matching of :math:`\sigma` is given by

    .. math:: B_\sigma(A) = \prod_{i=1}^N A_{i,\sigma(i)}

    Args:
        A: Square matrix (numpy array or array-like)

    Returns:
        Determinant value (complex or real)
    """
    A = np.matrix(A)
    n = len(A)

    perms = itertools.permutations(range(n))
    dt_sum = 0

    for perm in perms:
        prod = 1
        for i in range(n):
            prod *= A[i, perm[i]]
        dt_sum += sgn(perm) * prod

    return dt_sum


def dt_eigen(A):
    r"""
    Computes the determinant of a matrix using eigenvalue decomposition.


    .. math:: dt(A) = \prod_{i=0}^{n-1} \lambda_i

    with :math:`\lambda_i` the eigenvalues of :math:`A`.

    Args:
        A: A NumPy array representing the square matrix.

    Returns:
        The determinant of the matrix. Returns None if the input is not a
        square matrix or if eigenvalue decomposition fails.
    """
    try:
        # Check if the matrix is square
        rows, cols = A.shape
        if rows != cols:
            print("Input matrix must be square.")
            return None

        # Perform eigenvalue decomposition
        eigenvalues = np.linalg.eigvals(A)

        # Calculate the determinant as the product of eigenvalues
        determinant = np.prod(eigenvalues)
        return determinant

    except np.linalg.LinAlgError:
        print("Eigenvalue decomposition failed.")
        return None
