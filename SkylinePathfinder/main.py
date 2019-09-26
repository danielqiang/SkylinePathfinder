from SkylinePathfinder.model import EuclideanGraph
from SkylinePathfinder.tsp import tsp_solver
import random


def calc_time(dist, destinations):
    """
    Given a Euclidean path distance and the total number of
    classrooms to visit, return the approximate time needed
    to deliver SkylinePathfinder to every classroom.

    :param dist: (float) Euclidean distance.
    :param destinations: (int) Number of destinations (classrooms).
    """
    from datetime import timedelta

    # Euclidean units per second.
    # It takes about 2 minutes (120 seconds) to traverse the entire school diagonally,
    # and the school grid is a 200 x 300 Euclidean grid -> ~360.55 units diagonally.
    units_per_second = 360.55 / 120

    # Each Singagram takes about 90 seconds.
    return timedelta(seconds=destinations * 90 + dist / units_per_second)


def main(destinations, show_graph=False):
    """
    A program to approximate the time needed to deliver SkylinePathfinder
    to every classroom at my high school.

    Builds a 3D Euclidean Graph to model the school, then uses a greedy TSP
    heuristic to find a near-optimal route. It then calculates the
    approximate time needed by mapping Euclidean distance to seconds.

    :param destinations: (int) Number of random classrooms
            to choose. When used in practice, these classrooms
            are passed directly.
    :param show_graph: (bool) If true, draws the Euclidean graph on
             an interactive 3D plot (matplotlib Axes3D).
    """
    graph = EuclideanGraph().from_csv("./csv_data/school.csv")
    graph.connect_all()

    if len(graph.nodes.classrooms) < destinations:
        raise ValueError("School only has {} classrooms.".format(len(graph.nodes.classrooms)))

    classrooms = random.sample(graph.classrooms.names, k=destinations)

    route, dist = tsp_solver(graph, classrooms, classrooms[0], algorithm='greedy')

    print("Classrooms to visit: {}".format(classrooms))
    print("Route: {}".format(route.names))
    print("Time needed: {}".format(calc_time(dist, destinations)))

    if show_graph:
        graph.draw(paths=route)


if __name__ == '__main__':
    main(20, show_graph=True)
