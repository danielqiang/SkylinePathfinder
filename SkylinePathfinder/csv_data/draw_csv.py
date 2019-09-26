from SkylinePathfinder.model import EuclideanGraph


def draw(csv_file):
    """
    Builds and displays a 3D Euclidean graph from a
    csv file.

    :param csv_file: (str) filepath to csv file.
    """
    graph = EuclideanGraph().from_csv(csv_file)
    graph.draw()


if __name__ == '__main__':
    draw('school.csv')
