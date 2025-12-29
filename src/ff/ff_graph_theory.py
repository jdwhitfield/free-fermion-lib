"""
Graph Theory and Visualization Functions for Free Fermion Library

This module contains graph theory algorithms and visualization functions,
particularly focused on planar graphs and the Pfaffian ordering algorithm
(FKT algorithm).

Key algorithms:
- Pfaffian ordering algorithm (FKT algorithm) for planar graphs
- Graph visualization and plotting functions
- Perfect matching algorithms
- Dual graph construction for planar embeddings

Copyright 2025 James.D.Whitfield@dartmouth.edu
"""

import itertools
import itertools as it
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .ff_combinatorics import dt_eigen
from .ff_utils import clean


def _draw_labeled_multigraph(G, pos=None, ax=None):
    """
    Draw a labeled multigraph with curved edges for multiple connections.

    Length of connectionstyle must be at least that of a maximum number of
    edges between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.

    Args:
        G: NetworkX graph object
        pos: Dictionary of node positions (optional)
        ax: Matplotlib axis object (optional)
    """
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    if not pos:
        pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )


def _draw_labeled_graph(G, pos=None, ax=None):
    """
    Draw a labeled graph using positions (if given).

    Args:
        G: NetworkX graph object
        pos: Dictionary of node positions (optional)
        ax: Matplotlib axis object (optional)
    """
    if not pos:
        pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="grey", ax=ax)

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )


def plot_graph_with_edge_weights(A, matching=None, title=None):
    """
    Plot the graph associated with adjacency matrix A with edge weights.

    Given a real [n x n] matrix A, interpret the entries as directed edge
    weights, then plot the graph with the edge weights.

    Args:
        A: A NumPy array representing the square matrix or NetworkX graph
        matching: List of edges to highlight in red (optional)
        title: Title for the plot (optional)

    Returns:
        NetworkX graph object with edge weights
    """
    if (
        isinstance(A, nx.Graph)
        or isinstance(A, nx.DiGraph)
        or isinstance(A, nx.MultiGraph)
        or isinstance(A, nx.MultiDiGraph)
    ):
        A = nx.to_numpy_array(A, weight="weight")

    n = A.shape[0]

    if np.allclose(A, A.T):
        G = nx.Graph()
    else:
        # If asymmetric, create a directed graph
        G = nx.DiGraph()

    # Add nodes
    for i in range(n):
        G.add_node(i)

    # Add edges with weights
    # if it is a digraph, loop over i and j. If it is a graph loop over i<j
    if G.is_directed():
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0:  # Only add edges with non-zero weights
                    G.add_edge(i, j, weight=round(10e3 * A[i, j]) / 10e3)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] != 0:  # Only add edges with non-zero weights
                    G.add_edge(i, j, weight=round(10e3 * A[i, j]) / 10e3)

    layout_options = [
        nx.rescale_layout,
        nx.random_layout,
        nx.shell_layout,
        nx.fruchterman_reingold_layout,
        nx.spectral_layout,
        nx.kamada_kawai_layout,
        nx.spring_layout,
        nx.circular_layout,
    ]

    layout_opt = layout_options[2]
    pos = layout_opt(G)

    if G.is_directed():
        _draw_labeled_multigraph(G, pos)
    else:
        _draw_labeled_graph(G, pos)

    if matching is not None:
        # Highlight the matching edges
        nx.draw_networkx_edges(G, pos, edgelist=matching, edge_color="red", width=2)

    # title of plot
    if title is not None:
        plt.title(title)
    else:
        plt.title("Graph A with Edge Weights")

    plt.show()
    return G


def generate_random_planar_graph(n, seed=None):
    """
    Generate a random planar graph with n nodes.

    Args:
        n: The number of nodes in the graph
        seed: Random seed for reproducibility (optional)

    Returns:
        A NetworkX graph object representing the planar graph.
        Returns None if a planar graph with n nodes cannot be generated
        within a reasonable number of attempts.
    """
    if seed:
        random.seed(seed)

    max_attempts = 1000  # Limit the number of attempts to find a planar graph
    for _ in range(max_attempts):
        G = nx.gnm_random_graph(n, random.randint(n - 1, 2 * n - 3))

        # Generate a random graph
        if nx.check_planarity(G)[0] and nx.is_connected(G):
            return G
    return None  # Return None if no planar graph found within max_attempts


def plot_planar_embedding(G, pfo=None, title=None):
    """
    Plot the planar embedding of a graph with oriented faces.

    Args:
        G: The original NetworkX graph
        pfo: Pfaffian ordering matrix (optional)
        title: Title for the plot (optional)
    """
    pos = nx.planar_layout(G)  # Use planar layout for better visualization

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=20)

    if pfo is not None:
        plt.title("Planar Embedding with Oriented Faces")
        # Highlight the edges according to orientation in pfo
        for u, v in G.edges():
            L = min(u, v)
            R = max(u, v)

            if pfo[L, R] < 0:
                nx.draw_networkx_edges(
                    nx.DiGraph(G),
                    pos,
                    edgelist=[[L, R]],
                    edge_color="red",
                    arrows=True,
                    width=2,
                )
            elif pfo[L, R] > 0:
                nx.draw_networkx_edges(
                    nx.DiGraph(G),
                    pos,
                    edgelist=[[R, L]],
                    edge_color="red",
                    arrows=True,
                    width=2,
                )
            else:
                continue
    if title:
        plt.title(title)

    plt.show()


def dual_graph_H(G, F, T):
    """
    Create the dual graph according to the PFO algorithm (Vazirani 1987).

    "Construct a new graph H having one vertex corresponding to each face
    (including the infinite face) of G. Two vertices u and v of H are
    connected by an edge iff the corresponding faces share an edge not in T.
    Let r be the vertex in H corresponding to the infinite face of G. Root H
    at r."

    Args:
        G: Original planar graph
        F: List of faces from planar embedding
        T: Spanning tree of G

    Returns:
        NetworkX graph H representing the dual graph
    """
    face_edges = []
    for f in F:
        face_edges.append([tuple(sorted(e)) for e in f])

    edges_not_in_t = G.edges() - T.edges()

    # for storing the edges of the tree
    dual_edges = []

    # go through edges missing from tree
    for e in edges_not_in_t:
        u = None
        v = None

        for f in face_edges:
            if e in f and u is None:
                u = face_edges.index(f)
                continue

            if e in f and u is not None:
                v = face_edges.index(f)
                dual_edges.append([u, v])
                break

        if v is None and u is not None:
            dual_edges.append([u, len(face_edges)])

    H = nx.Graph()
    H.add_nodes_from(range(len(face_edges)))
    H.add_edges_from(dual_edges)

    # It is easy to see that H is a tree.
    nx.is_tree(H)
    return H


def faces(graph, emb=None):
    """
    Return the faces of an embedded graph using the networkx embedding scheme.

    ADAPTED HEAVILY FROM SAGE:
    https://github.com/sagemath/sage/blob/develop/src/sage/graphs/generic_graph.py#L6815

    Args:
        graph: NetworkX graph object
        emb: Planar embedding from NetworkX

    Returns:
        List of faces, where each face is a list of edges
    """
    # Establish set of possible edges
    edgeset = set()
    for u, v in graph.edges:
        edgeset.add((u, v))
        edgeset.add((v, u))

    # Storage for face paths
    faces = []
    minedge = min(edgeset)
    path = [minedge]
    edgeset.discard(minedge)

    if not emb:
        [is_planar, emb] = nx.check_planarity(graph)
        if not is_planar:
            return 0

    # Trace faces
    while edgeset:
        u, v = path[-1]
        neighbors = emb[v]
        next_node = neighbors[u]["cw"]
        e = (v, next_node)

        if e == path[0]:
            faces.append(path)
            minedge = min(edgeset)
            path = [minedge]
            edgeset.discard(minedge)
        else:
            path.append(e)
            edgeset.discard(e)
    if path:
        faces.append(path)

    max_face = None
    for face in faces:
        if max_face is None:
            max_face = face
        elif len(face) > len(max_face):
            max_face = face

    # get the outer_face from networkx
    from networkx.algorithms.planar_drawing import triangulate_embedding

    [emb2, outer_face] = triangulate_embedding(emb, False)

    outer_face.sort()

    for face in faces:
        # get the face as a sorted list of nodes
        fverts = []
        for e in face:
            fverts.append(e[0])
            fverts.append(e[1])
        fverts = list(set(fverts))
        fverts.sort()

        # compare to the networkx outer face
        if outer_face == fverts:
            max_face = face

            # https://www.geeksforgeeks.org/python-move-element-to-end-of-the-list/
            # moving element to end
            # using append() + pop() + index()
            faces.append(faces.pop(faces.index(max_face)))
            break

    return faces


def complete_face(pfo, edges, verbose=False):
    """
    Update one edge while maintaining the PFO condition.

    Args:
        pfo: Pfaffian ordering matrix
        edges: List of edges in the face
        verbose: Print debug information (optional)

    Returns:
        Updated PFO matrix
    """
    sign = 1

    skip_list = []

    for e in edges:
        # reverse sign if edge is reversed in face
        if e[0] > e[1]:
            prefix = -1
        else:
            prefix = 1

        u = min(e[0], e[1])
        v = max(e[0], e[1])

        if np.allclose(np.real(pfo[u, v]), 0):
            skip_list.append(e)
            continue

        sign *= pfo[u, v] * prefix

    if verbose:
        print(skip_list)
        print("sign without edge:", sign)

    if len(skip_list) == 0:
        print("no unassigned edges among:", edges)
        return

    if len(skip_list) > 1:
        print("more than one unassigned edge:", skip_list)

    e = skip_list.pop()

    # reverse sign if edge is reversed in face
    if e[0] > e[1]:
        prefix = -1
    else:
        prefix = 1

    u = min(e[0], e[1])
    v = max(e[0], e[1])

    if sign == -1:
        pfo[u, v] = prefix
        pfo[v, u] = prefix
    else:
        pfo[u, v] = -1 * prefix
        pfo[v, u] = -1 * prefix

    return pfo


def count_perfect_matchings(graph):
    """
    Find all perfect matchings in a graph.

    If planar, this method uses a pfo, otherwise it is computed via brute force

    Args:
        graph: A NetworkX graph

    Returns:
        The number of weighted matching for the given graph.
    """

    if nx.is_planar(graph):
        return count_perfect_matchings_planar(graph)
    else:
        return len(find_perfect_matchings_brute(graph))


def count_perfect_matchings_planar(graph):
    """
    Find all perfect matchings in a planar graph using the PFO algorithm.

    Args:
        graph: A NetworkX graph

    Returns:
        The number of weighted matching for the given graph
    """

    assert nx.is_planar(graph)

    if len(graph) == 0:
        return 0
        # Trivial case, no non-zero matchings

    A = nx.adjacency_matrix(graph).toarray()

    # run the pfo algorithm
    pfo = pfo_algorithm(graph)

    # use the pfo ordering to align the hf and pf
    pfoA = np.multiply(pfo, A)
    pfoA = np.triu(pfoA)
    pfoA = pfoA - pfoA.T

    return clean(np.sqrt(dt_eigen(pfoA)), 10)


def find_perfect_matchings_brute(graph):
    """
    Find all perfect matchings in a graph using a brute force enumeration.

    Args:
        graph: A NetworkX graph

    Returns:
        A list of tuples, where each tuple represents a perfect matching
    """
    n = len(graph.nodes)
    if n % 2 != 0:
        return []  # No perfect matchings possible for odd number of nodes

    if n > 15:
        print("Warning brute force approach too expensive")
        return []

    all_edges = list(graph.edges)
    perfect_matchings = []
    for combination in itertools.combinations(all_edges, n // 2):
        nodes_in_matching = set()
        for u, v in combination:
            nodes_in_matching.add(u)
            nodes_in_matching.add(v)
        if len(nodes_in_matching) == n:
            perfect_matchings.append(combination)

    return perfect_matchings


def pfo_algorithm(graph, verbose=False):
    """
    Pfaffian ordering algorithm for planar graphs (FKT algorithm).

    This algorithm is due to Kasteleyn 1967 by way of Vazirani 1987.

    The algorithm constructs a pfaffian ordering of the edges of a planar graph
    such that the pfaffian of the resulting skew-symmetric matrix equals
    the number of perfect matchings in the graph.

    Args:
        graph: NetworkX planar graph
        verbose: Print debug information and plots (optional)

    Returns:
        NumPy array representing the pfaffian ordering matrix
    """
    # Step 1: Find spanning tree
    T = nx.maximum_spanning_tree(graph)

    if verbose:
        p = -10 * nx.to_numpy_array(T)
        plot_planar_embedding(graph, p, title="T")

    orientation = nx.adjacency_matrix(graph).toarray()
    orientation = orientation * 99j  # mark locations we need to assign

    # Step 2: Build spanning tree for dual graph
    # get embedding from networkx
    [a, emb] = nx.check_planarity(graph)

    # get faces of graph
    F = faces(graph, emb)

    # get dual graph for those faces
    # root is the H[len(F)] vertex corresponding to the infinite face
    H = dual_graph_H(graph, F, T)

    if verbose:
        plot_planar_embedding(H, title="H")
        print(len(F))
        for i, face in enumerate(F[:5]):  # Print first 5 faces
            print(f"F[{i}]", face)

    if not nx.is_tree(H):
        print(H)
        assert nx.is_tree(H), "H is not a tree"

    # Step 3: Orient edges of T
    for e in T.edges():
        u = min(e[0], e[1])
        v = max(e[0], e[1])
        orientation[u, v] = 1
        orientation[v, u] = -1

    if verbose:
        plot_planar_embedding(graph, np.real(orientation))

    # Step 4: The rooted tree H dictates the order in which the rest of the
    # edges of G will be oriented. The orientation starts with the faces
    # corresponding to the leaves of H, and moves up to r.

    # The orientation starts with the faces corresponding to the leaves of H,
    # and moves up to r. Let e be the edge in G corresponding to the edge
    # (u -> v) in H (assuming that all edges in H are directed away from the
    # root). Let f be the face in G corresponding to u. Assume that the faces
    # corresponding to all descendents of v have already been oriented. Then,
    # e is the only unoriented edge in f. Now orient e so that f has an odd
    # number of edges oriented clockwise.

    # we ordered the faces such that the last of the faces is the external face
    root = max(H.nodes())

    while len(H.nodes()) > 1:
        # get leaves (which are not the root)
        leaves = [x for x in H.nodes() if H.degree(x) == 1 and x != root]

        if len(leaves) == 0:
            break

        for leaf in leaves:
            face = F[leaf]
            complete_face(orientation, face)
            H.remove_node(leaf)

            if verbose:
                plot_planar_embedding(
                    graph, np.real(orientation), title="Face {}: {}".format(leaf, face)
                )

    if np.sum(np.sum(np.imag(orientation))) > 1e-16:
        print("Incomplete pfo")
    else:
        print("Complete pfo")

    pfo = np.real(orientation)

    if verbose:
        plot_planar_embedding(graph, pfo, "Constructed PFO")

    return pfo


def compute_tree_depth(graph):
    """
    Compute the depth of a tree graph.

    Args:
        graph: NetworkX tree graph

    Returns:
        Integer representing the maximum depth of the tree
    """
    if not nx.is_tree(graph):
        raise ValueError("Graph must be a tree")

    # Choose an arbitrary root (first node)
    root = list(graph.nodes())[0]

    # Compute shortest path lengths from root to all nodes
    depths = nx.single_source_shortest_path_length(graph, root)

    return max(depths.values())
