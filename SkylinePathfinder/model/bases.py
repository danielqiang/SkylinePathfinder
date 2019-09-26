import networkx as nx

__all__ = ['GraphFilterMixin', 'NodeList']


class GraphFilterMixin(nx.Graph):
    """
    A filterable undirected graph.

    Allows simple, top-level node filtering by attribute.

    Supported attribute filters:
        - classrooms (:rtype: NodeList)
        - hallways (:rtype: NodeList)
        - names (:rtype: list)
        - floors (:rtype: list)
        - coords (:rtype: list)
        - coords_3d (:rtype: list)


    Examples
    --------

    Create a filterable graph (see networkx.Graph for more details):

        >>> from SkylinePathfinder.model import EuclideanNode
        >>> f = GraphFilterMixin()
        >>> f.add_node(EuclideanNode('Hallway', 'name1', 'floor1', ('x1', 'y1')))
        >>> f.add_node(EuclideanNode('Hallway', 'name2', 'floor2', ('x2', 'y2')))
        >>> f.add_node(EuclideanNode('Classroom', 'name3', 'floor3', ('x3', 'y3')))
        >>> f.add_node(EuclideanNode('Classroom', 'name4', 'floor4', ('x4', 'y4')))

    Filter nodes by attribute:

        >>> f.coords
        [('x1', 'y1'), ('x2', 'y2'), ('x3', 'y3'), ('x4', 'y4')]
        >>> f.names
        ['name1', 'name2', 'name3', 'name4']

    Attributes can be chained:

        >>> f.classrooms.coords
        [('x3', 'y3'), ('x4', 'y4')]
        >>> f.hallways.coords_3d
        [('x1', 'y1', 'floor1'), ('x2', 'y2', 'floor2')]

    Chained attribute access is only allowed when a new NodeList
    is returned. For example, the following throws an error:

        >>> f.names.names
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        AttributeError: 'list' object has no attribute 'names'
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, attr):
        # Allow top-level attribute access for NodeList filtering
        if attr in NodeList.__dict__:
            return self.nodes.__getattribute__(attr)
        return self.__getattribute__(attr)

    @property
    def nodes(self):
        return NodeList(super().nodes)


class NodeList(list):
    """
    A list of EuclideanNode objects. Used to quickly filter nodes by attributes.

    See GraphFilterMixin for usage details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def classrooms(self):
        return NodeList(node for node in self if node.type == 'Classroom')

    @property
    def hallways(self):
        return NodeList(node for node in self if node.type == 'Hallway')

    @property
    def names(self):
        return [node.name for node in self]

    @property
    def floors(self):
        return [node.floor for node in self]

    @property
    def coords(self):
        return [node.coord for node in self]

    @property
    def coords_3d(self):
        return [node.coord + tuple([int(node.floor)]) for node in self]
