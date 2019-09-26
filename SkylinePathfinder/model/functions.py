from .euclidean_node import EuclideanNode

__all__ = ['euclidean_dist', 'euclidean_dist_nodes', 'naive_knn', 'project']


def euclidean_dist(p1, p2):
    """
    Returns the euclidean distance between points (p1, p2)
    in n-dimensional space.

    Points must have the same number of dimensions.
    """
    if len(p1) != len(p2):
        raise ValueError("Points must have the same number of dimensions.")
    return sum((d1 - d2) ** 2 for d1, d2 in zip(p1, p2)) ** 0.5


def euclidean_dist_nodes(n1, n2):
    """
    Returns the euclidean distance between two EuclideanNode
    objects in n-dimensional space. Points must have the same
    number of dimensions.

    Equivalent to euclidean_dist(n1.coord, n2.coord).
    """
    return euclidean_dist(n1.coord, n2.coord)


def naive_knn(node, data, k, weight_func=euclidean_dist, ignore_self=True, same_floor=True):
    """
    Finds the k nearest neighbors of a source node using a brute-force approach.
    Returns the neighbors sorted from least to greatest by distance,
    using weight_func as a key function.

    :param node: (EuclideanNode) Source node.
    :param data: (list) All nodes in the graph.
    :param k: (int) Number of nearest neighbors to return.
    :param weight_func: A distance function that returns a numerical distance
                        between two points in n-dimensional space. Must have
                        signature function(coord1, coord2).
    :param ignore_self: (bool) If true, ignore cases where weight_func(node, other) == 0.
    :param same_floor: (bool) If true, limits search to nodes on the same floor.
    """
    if same_floor:
        data = [item for item in data if item.floor == node.floor]
    if ignore_self:
        data = [item for item in data if weight_func(node.coord, item.coord) != 0]
    if k > len(data):
        raise ValueError("Data only contains {} total neighbors after filtering".format(len(data)))

    _sorted = sorted(data, key=lambda neighbor: weight_func(neighbor.coord, node.coord))
    return _sorted[:k]


def project(p, q, node):
    """
    Performs an orthogonal projection from (node) onto
    line segment PQ from p to q. Returns the new hallway
    node after projection onto PQ.

    If the projected node is not on the PQ segment,
    the closest point on the segment is returned as a node.

    Notes:
        p, q, and node must all be two-dimensional.
        The new hallway node will be named 'T-' + node.name.

    :param p, q: (EuclideanNode) Hallway nodes representing a line segment PQ.
    :param node: (EuclideanNode) Classroom node to project onto PQ.
    """
    from shapely.geometry import LineString, Point

    if any(len(n.coord) != 2 for n in [p, q, node]):
        raise ValueError("All nodes must be two-dimensional.")

    line = LineString([p.coord, q.coord])
    point = Point(node.coord)
    projected = line.interpolate(line.project(point))

    return EuclideanNode(
        node_type='Hallway',
        name='T-' + node.name,
        floor=node.floor,
        coord=projected.coords[0]
    )
