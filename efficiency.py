"""
Compute the global and local efficiencies of the graph.

These algorithms work with undirected and directed graphs.

Url: https://groups.google.com/forum/#!msg/networkx-discuss/ycxtVuEqePQ/1VA2us0c7c0J

"""

from __future__ import print_function, division

import networkx as nx


def global_efficiency(G, weight=None):
    """Return the global efficiency of the graph G

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    global_efficiency : float

    Notes
    -----
    The published definition includes a scale factor based on a completely
    connected graph. In the case of an unweighted network, the scaling factor
    is 1 and can be ignored. In the case of a weighted graph, calculating the
    scaling factor requires somehow knowing the weights of the edges required
    to make a completely connected graph. Since that knowlege may not exist,
    the scaling factor is not included. If that knowlege exists, construct the
    corresponding weighted graph and calculate its global_efficiency to scale
    the weighted graph.

    Distance between nodes is calculated as the sum of weights. If the graph is
    defined such that a higher weight represents a stronger connection,
    distance should be represented by 1/weight. In this case, use the invert_
    weights function to generate a graph where the weights are set to 1/weight
    and then calculate efficiency

    References
    ----------
    .. [1] Latora, V., and Marchiori, M. (2001). Efficient behavior of
       small-world networks. Physical Review Letters 87.
    .. [2] Latora, V., and Marchiori, M. (2003). Economic small-world behavior
       in weighted networks. Eur Phys J B 32, 249-263.

    """
    N = len(G)
    if N < 2:
        return 0    # facilitates calculation of local_efficiency although
                    # could reasonably raise nx.NetworkXUnfeasible or
                    # nx.NetworkXPointlessConcept error instead and force
                    # testing to occur in local_efficiency

    inv_lengths = []
    for node in G:
        if weight is None:
            lengths = nx.single_source_shortest_path_length(G, node)
        else:
            lengths = nx.single_source_dijkstra_path_length(G, node,
                                                            weight=weight)

        inv = [1/x for x in lengths.values() if x is not 0]
        inv_lengths.extend(inv)

    return sum(inv_lengths)/(N*(N-1))


def local_efficiency(G, weight=None):
    """Return the local efficiency of each node in the graph G

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    local_efficiency : dict
       the keys of the dict are the nodes in the graph G and the corresponding
       values are local efficiencies of each node

    Notes
    -----
    The published definition includes a scale factor based on a completely
    connected graph. In the case of an unweighted network, the scaling factor
    is 1 and can be ignored. In the case of a weighted graph, calculating the
    scaling factor requires somehow knowing the weights of the edges required
    to make a completely connected graph. Since that knowlege may not exist,
    the scaling factor is not included. If that knowlege exists, construct the
    corresponding weighted graph and calculate its local_efficiency to scale
    the weighted graph.

    References
    ----------
    .. [1] Latora, V., and Marchiori, M. (2001). Efficient behavior of
       small-world networks. Physical Review Letters 87.
    .. [2] Latora, V., and Marchiori, M. (2003). Economic small-world behavior
       in weighted networks. Eur Phys J B 32, 249-263.

    """
    if G.is_directed():
        new_graph = nx.DiGraph
    else:
        new_graph = nx.Graph

    efficiencies = dict()
    for node in G:
        temp_G = new_graph()
        temp_G.add_nodes_from(G.neighbors(node))
        for neighbor in G.neighbors(node):
            for (n1, n2) in G.edges(neighbor):
                if (n1 in temp_G) and (n2 in temp_G):
                    temp_G.add_edge(n1, n2)

        if weight is not None:
            for (n1, n2) in temp_G.edges():
                temp_G[n1][n2][weight] = G[n1][n2][weight]

        efficiencies[node] = global_efficiency(temp_G, weight)

    return efficiencies


def average_local_efficiency(G, weight=None):
    """Return the average local efficiency of all of the nodes in the graph G

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    average_local_efficiency : float

    Notes
    -----
    The published definition includes a scale factor based on a completely
    connected graph. In the case of an unweighted network, the scaling factor
    is 1 and can be ignored. In the case of a weighted graph, calculating the
    scaling factor requires somehow knowing the weights of the edges required
    to make a completely connected graph. Since that knowlege may not exist,
    the scaling factor is not included. If that knowlege existed, a revised
    version of this function would be required.

    References
    ----------
    .. [1] Latora, V., and Marchiori, M. (2001). Efficient behavior of
       small-world networks. Physical Review Letters 87.
    .. [2] Latora, V., and Marchiori, M. (2003). Economic small-world behavior
       in weighted networks. Eur Phys J B 32, 249-263.

    """
    eff = local_efficiency(G, weight)
    total = sum(eff.values())
    N = len(eff)
    return total/N


def invert_weights(G, weight='weight'):
    """Return a graph where the weight of each edge is 1 over the weight of
       the corresponding edge in graph G

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Ginv : NetworkX graph

    Notes
    -----
    This function is meant to address the case where weights represent the
    "strength" of the connection rather than the distance between nodes. In
    this case, the distance would be considered to be 1 over the weight so
    that "weak" connections have correspondingly "long" distances between
    nodes.

    """
    Ginv = nx.create_empty_copy(G)
    for (n1, n2) in G.edges():
        dist = 1/G[n1][n2][weight]
        Ginv.add_edge(n1, n2, {weight: dist})

    return Ginv
