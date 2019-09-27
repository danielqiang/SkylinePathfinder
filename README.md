# Skyline Pathfinder

Skyline Pathfinder is a 3D routefinding application for classrooms at Skyline High School. It consists of two main components:
1. A backend framework that models all hallways and classrooms as edges and nodes in a graph. Supports interactive 3D visual displays:
  
   ![](https://i.imgur.com/3Z9Jvca.gif)
   
   <sub><sup>Classrooms/hallways displayed as green nodes/blue edges respectively.</sup></sub>

   Route visualization:

   ![](https://i.imgur.com/OlBGsNx.gif)
   
   <sub><sup>Route shown in red.</sup></sub>
     
   And initialization from CSV data.
     
2. A [Traveling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) solver that, depending on the selected algorithm, computes an optimal or near-optimal route which visits each critical node in the graph. This is done in three steps:
    1. Reduce the Euclidean graph model to a smaller, complete Euclidean graph by connecting critical nodes using A* search. Since the resulting reduced graph consists only of critical nodes that must be included in the route, we can directly apply a TSP route optimization algorithm. The optimal paths generated by A* search are cached to expand the graph after finding an optimal route.
    2. Perform a user-selected TSP algorithm to calculate an optimal or near-optimal route over the reduced graph. A brute-force algorithm and heuristic greedy algorithm have been implemented. 
    3. Expand the graph, using cached optimal paths to yield a solution.

## Dependencies
[NetworkX](https://networkx.github.io/), [Matplotlib](https://matplotlib.org/)

## Why Skyline Pathfinder?

In high school, my vocal jazz group did a popular fundraiser called Singagrams every Valentine's Day. We ran around the school for the entire day, going into classrooms and serenading students. However, we consistently ran into the same problem--we spent too much time running between classrooms. I wanted to take a computational approach to solving the problem, but did not possess the necessary technical background at the time. Frustrated, I decided to revisit the project over winter break a few months later. 

This repository was the result. 
