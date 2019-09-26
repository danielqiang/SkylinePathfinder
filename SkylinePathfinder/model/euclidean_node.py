class EuclideanNode:
    """
    A hashable node class. EuclideanNode objects
    should not be modified after initialization.

    Can be used as a custom node class for networkx graphs.

    Attributes:
        - node_type: (str) Node type, e.g. hallway/classroom.
        - name: (str) Node name.
        - floor: (int) The floor the node is on, if applicable.
                Used when modeling multi-floor buildings
                to specify z-coordinate.
        - coord: (int, int) Tuple of 2D Euclidean coordinates (x, y).
    """
    def __init__(self, node_type, name, floor, coord):
        self.type = node_type
        self.name = name
        self.floor = floor
        self.coord = coord

    def __repr__(self):
        return "EuclideanNode({} {}: {})".format(self.type, self.name, self.coord)

    def __hash__(self):
        return hash((self.type, self.name, self.floor, self.coord))

    def __eq__(self, other):
        # Two nodes are equal if and only if their fields are equal.
        return type(self) == type(other) and hash(self) == hash(other)
