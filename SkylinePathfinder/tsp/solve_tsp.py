from model.bases import GraphFilterMixin, NodeList
from model.functions import euclidean_dist_nodes
from itertools import combinations, permutations
import networkx as nx

__all__ = ['ReducedGraph', 'tsp_solver']


class ReducedGraph(GraphFilterMixin):
    """
    A reduced, undirected graph class.
    Represents a complete Euclidean graph.

    A reduced graph is comprised entirely of edges that each
    represent a singly connected, acyclic graph. It is useful
    for reducing repetitive computation for expensive algorithms
    such as the Traveling Salesman Problem, for which this class
    is primarily implemented to support.
    """
    def __init__(self, start=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        # Map from each reduced edge to its corresponding full path.
        self.edge_dict = {}

    def add_reduced_edge(self, u, v, path, **attrs):
        """
        Adds a reduced, undirected edge from node u to node v.

        A reduced edge is an edge that represents a singly connected
        graph with source node u and terminal node v.

        :param u, v: EuclideanNode objects.
        :param path: (list) Path from node u to node v.
        :param attrs: Edge attributes to pass to networkx.Graph().add_edge().
                See networkx.Graph for more info.
        """
        self.add_edge(u, v, **attrs)
        # Euclidean graphs are undirected, so the edges must also
        # be undirected
        self.edge_dict[(u, v)] = path
        self.edge_dict[(v, u)] = path[::-1]

    def expand_path(self, reduced_path):
        """
        Given a path comprised entirely of nodes in the reduced graph,
        return the full expanded path.

        :param reduced_path: (list) Path over nodes in the reduced graph.
        :return: NodeList object.
        """
        full_path = []
        for u, v in zip(reduced_path[:-2], reduced_path[1:-1]):
            full_path.extend(self.edge_dict[(u, v)][:-1])
        return NodeList(full_path + self.edge_dict[(reduced_path[-2], reduced_path[-1])])

    def neighbors(self, n):
        """
        Returns the neighbors of node n as a set.

        :param n: (EuclideanNode) A node in the graph.
        """
        return {*self[n].keys()}


def tsp_solver(graph, nodes, start, algorithm='brute force'):
    """
    Given a Euclidean graph, a set of nodes to visit, and
    a source node "start", generates a complete, reduced graph
    and finds the optimal route that visits all the nodes
    using a TSP algorithm.

    :param graph: (EuclideanGraph) graph object.
    :param nodes: (list) Nodes to visit.
    :param start: (EuclideanNode) Source node.
    :param algorithm: (str) TSP algorithm to use.
            Currently supported:
            - "brute force"
            - "greedy"
    :return: (NodeList) Optimal route from start that visits all the nodes. If a heuristic
            algorithm is chosen (e.g. greedy heuristic), an optimal solution
            cannot be guaranteed.
    """

    reduced = _reduce_graph(graph, nodes, start)

    if algorithm.lower() == 'greedy':
        shortest_path, length = _greedy(reduced)
    elif algorithm.lower() == 'brute force':
        shortest_path, length = _brute_force(reduced)
    else:
        _implemented = ['greedy', 'brute force']
        _bullet_points = ''.join('\n\t- ' + algo for algo in _implemented)
        raise NotImplementedError('"{}" is not recognized as an implemented algorithm. '
                                  'Implemented algorithms: {}'.format(algorithm, _bullet_points))

    full_path = reduced.expand_path(shortest_path)
    return full_path, length


def _reduce_graph(graph, nodes, start):
    """
    Given a graph and a set of critical nodes, generate
    a complete reduced graph.

    See ReducedGraph for more information.

    :param graph: (EuclideanGraph) graph object.
    :param nodes: (list) Critical nodes. Only these nodes will be included as nodes
                in the reduced graph.
    :param start: (EuclideanNode) Source node.
    :return: ReducedGraph object.
    """
    reduced = ReducedGraph(graph.node_dict[start])
    for u, v in combinations(nodes, 2):
        source, target = graph.node_dict[u], graph.node_dict[v]
        path = nx.astar_path(graph, source, target, heuristic=euclidean_dist_nodes)
        path_dist = _path_len(graph, path)
        reduced.add_reduced_edge(source, target, path, weight=path_dist)

    return reduced


def _brute_force(graph: ReducedGraph):
    """
    A brute-force approach to solving the Traveling Salesman Problem
    on a complete graph.

    Requires O(n!) time.

    :param graph: Complete graph (ReducedGraph object).
    :return:
        - (list) Optimal hamiltonian path over the reduced graph.
        - (int) Total euclidean distance of the returned path.
    """
    nodes = [node for node in graph if node is not graph.start]

    shortest_len, shortest_path = float('inf'), None
    for p in permutations(nodes):
        path = [graph.start] + list(p)
        length = _path_len(graph, path)

        if length < shortest_len:
            shortest_path, shortest_len = path, length
    return shortest_path, shortest_len


def _greedy(graph: ReducedGraph):
    """
    A greedy, heuristic approach to solving the Traveling Salesman Problem
    on a complete graph. This approach is not guaranteed to find an optimal path.

    Requires O(n^2) time with a naive nearest-neighbor algorithm.
    Can be reduced to O(nlog(n)) with a space-partitioning structure
    like a KD tree.

    :param graph: (ReducedGraph) graph object. Must be a complete graph.
    :return:
        - (list) Optimal hamiltonian path over the reduced graph.
        - (int) Total euclidean distance of the returned path.
    """
    greedy_path, visited = [graph.start], {graph.start}
    cur = graph.start
    # While unvisited neighbors exist
    while graph.neighbors(cur).difference(visited):
        # Get nearest unvisited neighbor
        nearest = min(graph.neighbors(cur).difference(visited),
                      key=lambda neighbor: euclidean_dist_nodes(cur, neighbor))
        greedy_path.append(nearest)
        visited.add(nearest)
        cur = nearest
    return greedy_path, _path_len(graph, greedy_path)


def _path_len(graph, path):
    """
    Returns the length of a given path in a Graph object.

    :param graph: Graph object.
    :param path: (list) Path over nodes in the graph.
                A path is a singly connected, acyclic graph.
    """
    return sum(graph[u][v].get('weight', 1) for u, v in zip(path[:-1], path[1:]))
