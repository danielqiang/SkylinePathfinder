from . import functions as funcs
from .bases import GraphFilterMixin
from .euclidean_node import EuclideanNode


class EuclideanGraph(GraphFilterMixin):
    """
    An undirected Euclidean graph class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_dict = {}

    def from_csv(self, csv_file):
        """
        Constructs a graph object from a csv file.
        Returns the object.

        The csv file must have the following headers:

            1. NodeType: Node type, e.g. hallway/classroom.
            2. Name: Node name.
            3. Floor: The floor the node is on, if applicable.
                    Used when modeling multi-floor buildings
                    to specify z-coordinate.
            4. Coordinates: Tuple of 2D Euclidean coordinates (x, y).
            5. Neighbors: List of names of the node's neighbors.

        Each row in the csv file will be used to construct
        a EuclideanNode object.

        :param csv_file: (str) filepath for csv file.
        """
        self.node_dict, neighbor_dict = self._process_csv(csv_file)

        for name, node in self.node_dict.items():
            for neighbor_name in neighbor_dict[name]:
                neighbor = self.node_dict[neighbor_name]
                distance = funcs.euclidean_dist_nodes(node, neighbor)
                self.add_edge(node, neighbor, weight=distance)
            self.add_node(node)
        return self

    @staticmethod
    def _process_csv(filename):
        """
        Constructs EuclideanNode objects from each
        row in a csv file.

        Helper method for from_csv().

        :param filename: (str) filepath for csv file.
        """
        import csv

        node_dict, neighbor_dict  = {}, {}

        with open(filename, "r") as csv_file:
            for row in csv.DictReader(csv_file):
                node = EuclideanNode(
                    node_type=row['NodeType'],
                    name=row['Name'],
                    floor=row['Floor'],
                    coord=eval(row['Coordinates'])
                )
                node_dict[row['Name']] = node
                neighbor_dict[row['Name']] = eval(row['Neighbors'])
        return node_dict, neighbor_dict

    def connect_all(self):
        """
        Converts the graph to a connected graph. This is done by
        connecting each disconnected classroom node in the
        graph to the two nearest hallway nodes.

        More explicitly:

            1. Given C where C is an disconnected node,
                find its two nearest connected neighbors A and B.
                AB represents a line segment:

                ( A ) - - - - - - ( B )

                          C

            2. Use an orthogonal projection to map C onto line AB
                as a new node D:

                ( A ) - ( D ) - - ( B )

                          C

                If D does not fall on segment AB, the closest point
                on AB is used.

            3. Connect A <-> D, B <-> D, and C <-> D:

                ( A ) - ( D ) - - ( B )
                          |
                          C

            4. Repeat steps 1-3 for every disconnected node in the graph.
        """
        # All classrooms are disconnected nodes
        for classroom in self.nodes.classrooms:
            a, b = funcs.naive_knn(classroom, self.nodes.hallways, k=2)
            d = funcs.project(a, b, classroom)

            self.add_edge(a, d, weight=funcs.euclidean_dist_nodes(a, d))
            self.add_edge(b, d, weight=funcs.euclidean_dist_nodes(b, d))
            self.add_edge(classroom, d, weight=funcs.euclidean_dist_nodes(classroom, d))

    @staticmethod
    def _edge_coords_3d_iter(edges):
        """
        Given a list of edges, returns an iterator
        over 2-tuple pairs of 3D coordinates.

        Helper method for draw().

        :param edges: (list) A list of edges from networkx.Graph.
                    See networkx.Graph.edges for more info.
        """
        for a, b in edges:
            yield (a.coord + tuple([int(a.floor)]), b.coord + tuple([int(b.floor)]))

    def _draw_path(self, axes, path, color='red'):
        """
        Draws a Euclidean path on a matplotlib Axes object.

        Helper method for draw().

        :param axes: matplotlib Axes object.
        :param path: (list) Euclidean path as a list of nodes.
        :param color: (str) matplotlib plot color.
        """
        edges = zip(path[:-1], path[1:])
        for u, v in self._edge_coords_3d_iter(edges):
            axes.plot(*zip(u, v), c=color, zorder=0, linewidth=3)

    def draw(self, paths=()):
        """
        Plots and shows the graph and any Euclidean paths on a set
        of 3D axes. Euclidean paths are layered on top of the graph.

        Uses matplotlib.

        :param paths: (list) Euclidean paths to plot.
        """
        import matplotlib.pyplot as plt
        # Axes3D is required for 3D plotting
        from mpl_toolkits.mplot3d import Axes3D
        from itertools import cycle
        from typing import Iterable

        # Input can either be a single path of a list of paths.
        if len(paths) > 0 and not isinstance(paths[0], Iterable):
            paths = [paths]
        # Color cycle for plotting
        colors = cycle('rgcmkb')

        ax = plt.axes(projection='3d')
        ax.scatter(*zip(*self.hallways.coords_3d), c='blue', alpha=0.5, zorder=5)
        ax.scatter(*zip(*self.classrooms.coords_3d), c='green', alpha=0.5, zorder=5)
        for u, v in self._edge_coords_3d_iter(self.edges):
            ax.plot(*zip(u, v), c='blue', alpha=0.5, zorder=5)
        for path in paths:
            self._draw_path(ax, path, color=next(colors))

        plt.show()
