import random
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

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
start_x, start_y = 0, 0
maze[start_y][start_x] = 0  # Make the start position part of the path

# Generate the maze
dfs(start_x, start_y)

# Modified function to extract the edges of the maze
def get_maze_edges(maze):
    """
    Extract the walls/edges of the maze and return them as a list of LineString objects.
    """
    edges = []
    height = len(maze)
    width = len(maze[0])
    
    # Check horizontal edges (between cells)
    for y in range(height):
        for x in range(width - 1):
            if maze[y][x] == 1 and maze[y][x + 1] == 0:
                # Left wall
                edges.append(LineString([(x, y), (x, y + 1)]))
            if maze[y][x] == 0 and maze[y][x + 1] == 1:
                # Right wall
                edges.append(LineString([(x + 1, y), (x + 1, y + 1)]))

    # Check vertical edges (between cells)
    for y in range(height - 1):
        for x in range(width):
            if maze[y][x] == 1 and maze[y + 1][x] == 0:
                # Top wall
                edges.append(LineString([(x, y), (x + 1, y)]))
            if maze[y][x] == 0 and maze[y + 1][x] == 1:
                # Bottom wall
                edges.append(LineString([(x, y + 1), (x + 1, y + 1)]))
                
    return edges

# Get the edges of the generated maze
maze_edges = get_maze_edges(maze)

# Plotting the maze edges using shapely geometries
def plot_maze_edges(maze_edges):
    """Plot the edges of the maze using Matplotlib."""
    fig, ax = plt.subplots(figsize=(11, 11))

    for edge in maze_edges:
        x, y = edge.xy
        ax.plot(x, y, color='black', linewidth=2)

    # Set plot limits and display
    ax.set_xlim(-1, width+1)
    ax.set_ylim(-1, height+1)
    ax.set_aspect('equal')
    plt.title("Maze with Shapely Edges")
    plt.show()

# Plot the maze edges
plot_maze_edges(maze_edges)
