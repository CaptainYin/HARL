import random
import matplotlib.pyplot as plt
import numpy as np

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
maze[start_y][start_x] = 0  # Make the start position part of the path

# Generate the maze
dfs(start_x, start_y)

# Visualize the maze using matplotlib
def plot_maze(maze):
    """Plot the maze using Matplotlib."""
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='binary')  # 'binary' gives a black and white color map
    plt.xticks([])  # Hide the x-axis ticks
    plt.yticks([])  # Hide the y-axis ticks
    plt.show()

# Plot the generated maze
plot_maze(maze)
# Plot points
x_coords = [p.x for p in points]
y_coords = [p.y for p in points]
ax.scatter(x_coords, y_coords, color='b', label="Points")