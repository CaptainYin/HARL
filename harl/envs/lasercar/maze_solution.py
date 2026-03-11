import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Define the maze dimensions (odd numbers preferred)
width = 21
height = 21

# Create a grid filled with walls (1 = wall, 0 = path)
maze = [[1 for _ in range(width)] for _ in range(height)]

# Directions: (dx, dy) for North, South, East, West
directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]

def is_valid_move(x, y):
    """Check if the move is within the maze boundaries and on an unvisited wall."""
    return 0 <= x < width and 0 <= y < height and maze[y][x] == 1

def dfs(x, y):
    """Perform DFS to generate the maze."""
    maze[y][x] = 0  # Mark the current cell as part of the path
    random.shuffle(directions)  # Randomize directions

    for dx, dy in directions:
        nx, ny = x + dx, y + dy  # Calculate the coordinates of the neighboring cell
        if is_valid_move(nx, ny):
            maze[ny - dy // 2][nx - dx // 2] = 0  # Carve a path between cells
            dfs(nx, ny)  # Recursively visit the neighbor

# Starting point
start_x, start_y = 1, 1
end_x, end_y = width - 2, height - 2  # Goal is bottom-right corner

maze[start_y][start_x] = 0  # Make the start position part of the path
maze[end_y][end_x] = 0      # Make the goal position part of the path

# Generate the maze
dfs(start_x, start_y)

# BFS to find the shortest path from start to end
def bfs(maze, start, goal):
    """Find the shortest path using BFS."""
    queue = deque([start])  # Queue for BFS, storing (x, y) tuples
    visited = set([start])  # Set to store visited positions
    parent_map = {}         # Map to trace back the path

    while queue:
        x, y = queue.popleft()
        
        # If we reach the goal, stop and trace the path
        if (x, y) == goal:
            return trace_path(parent_map, goal, start)
        
        # Explore neighbors (N, S, E, W)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited and maze[ny][nx] == 0:
                queue.append((nx, ny))
                visited.add((nx, ny))
                parent_map[(nx, ny)] = (x, y)  # Keep track of the parent (for path tracing)

    return None  # Return None if no path found

def trace_path(parent_map, goal, start):
    """Trace the path from goal to start using the parent map."""
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent_map[current]
    path.append(start)
    return path[::-1]  # Return reversed path (from start to goal)

# Find the solution path
solution_path = bfs(maze, (start_x, start_y), (end_x, end_y))

# Visualize the maze and the solution
def plot_maze_with_solution(maze, solution):
    """Plot the maze and the solution path using Matplotlib."""
    maze_copy = np.array(maze)  # Convert maze to numpy array for easier manipulation
    for (x, y) in solution:
        maze_copy[y][x] = 0.5  # Mark the solution path as 0.5 (for a different color)

    plt.figure(figsize=(10, 10))
    plt.imshow(maze_copy, cmap='gray')  # Use 'gray' to show paths and walls
    plt.xticks([])  # Hide the x-axis ticks
    plt.yticks([])  # Hide the y-axis ticks
    plt.show()

# Plot the generated maze and its solution
plot_maze_with_solution(maze, solution_path)
