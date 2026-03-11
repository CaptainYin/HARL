import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from SimEnv import SimulationEnvironment_Warehouse
import random


def rrt(start, goal, env, max_iter=50000, step_size=0.1):
    """RRT algorithm to find a path from start to goal."""
    tree = {start: None}
    for i in range(max_iter):
        random_point = (random.uniform(-env.area_size/2, env.area_size/2),
                        random.uniform(-env.area_size/2, env.area_size/2))
        
        nearest_point = min(tree.keys(), key=lambda p: np.linalg.norm(np.array(p) - np.array(random_point)))
        direction = np.array(random_point) - np.array(nearest_point)
        direction = direction / np.linalg.norm(direction)
        new_point = tuple(np.array(nearest_point) + direction * step_size)
        
        if not env.is_collision(new_point):
            tree[new_point] = nearest_point
            if np.linalg.norm(np.array(new_point) - np.array(goal)) < step_size:
                tree[goal] = new_point
                return trace_path(tree, goal)
    
    return None

def trace_path(tree, goal):
    """Trace back the path from goal to start."""
    path = [goal]
    while tree[path[-1]] is not None:
        path.append(tree[path[-1]])
    return path[::-1]

def exist_rrt(env):
    # Start and goal positions
    start = env.positions[0]
    goal = env.positions[1]

    # Run RRT to find a path
    path = rrt(start, goal, env)

    # Plot the environment and the path
    # fig, ax = plt.subplots()
    ax=env.display_environment(show=0)

    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'r-', label="RRT Path")
        ax.plot(*start, 'go', label="Start")
        ax.plot(*goal, 'bo', label="Goal")
        ax.legend()
    else:
        print("No path found.")

    plt.show()
# Create environment
env = SimulationEnvironment_Warehouse()
exist_rrt(env)

