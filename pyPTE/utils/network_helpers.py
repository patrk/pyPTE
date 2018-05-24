import numpy as np
import networkx

def linear_adjacecy(N):
    A = np.identity(N)
    return A

def pairwise_coupling(N):
    assert N % 2 == 0
    a = np.array([[-1, 1], [1, -1]])
    A = np.kron(np.eye(int(N/2)), a)
    return A

def equal_coupling(N):
    A = np.ones((N, N))
    np.fill_diagonal(A, -1.0)
    return A

def circular_adjacency(N):
    graph = networkx.cycle_graph(N)
    A = networkx.adjacency_matrix(graph)
    return A

def dorogovtsev_goltsvev_adjacency(n):
    graph = networkx.generators.dorogovtsev_goltsev_mendes_graph(n)
    A = networkx.adjacency_matrix(graph)
    return A

def erdoes_renyi_adjacency(n):
    graph = networkx.generators.erdos_renyi_graph(n)
    A = networkx.adjacency_matrix(graph)
    return A