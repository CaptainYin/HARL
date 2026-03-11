import random
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString,Point

# Define the maze dimensions (odd numbers preferred)
width = 21
height = 21

# Create a grid filled with walls (1 = wall, 0 = path)
maze = [[1 for _ in range(width)] for _ in range(height)]

# Directions: (dx, dy) for North, South, East, West
directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
# directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

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
start_x, start_y = 10, 10
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
    for y in range(0,height):
        for x in range(0,width-1):
            if maze[y][x] == 1 and maze[y][x + 1] == 0:
                # Left wall
                # edges.append(LineString([(x-(width+1)/2, y-(height+1)/2), (x-(width+1)/2, y-(height+1)/2 + 1)]))
                edges.append(LineString([(x, y), (x, y + 1)]))
            if maze[y][x] == 0 and maze[y][x + 1] == 1:
                # Right wall
                # edges.append(LineString([(x-(width+1)/2 + 1, y-(height+1)/2), (x-(width+1)/2 + 1, y-(height+1)/2 + 1)]))
                edges.append(LineString([(x + 1, y), (x + 1, y + 1)]))

    # Check vertical edges (between cells)
    for y in range(0,height - 1):
        for x in range(0,width):
            if maze[y][x] == 1 and maze[y + 1][x] == 0:
                # Top wall
                # edges.append(LineString([(x-(width+1)/2, y-(height+1)/2), (x -(width+1)/2+ 1, y-(height+1)/2)]))
                edges.append(LineString([(x, y), (x + 1, y)]))
            if maze[y][x] == 0 and maze[y + 1][x] == 1:
                # Bottom wall
                # edges.append(LineString([(x-(width+1)/2, y-(height+1)/2 + 1), (x -(width+1)/2+ 1, y -(height+1)/2+ 1)]))
                edges.append(LineString([(x, y + 1), (x + 1, y + 1)]))
    new_edges,new_edges_list=[],[]
    offset_x=-(width+1)/2
    offset_y=-(height+1)/2
    for edge in edges:
        x, y = edge.xy
        # if x[0]==x[1] and x[0]==-10:
        #     continue
        # if x[0]==x[1] and x[0]==10:
        #     continue
        # if y[0]==y[1] and y[0]==-10:
        #     continue        
        # if y[0]==y[1] and y[0]==10:
        #     continue    
        new_edges.append(LineString([(x[0]+offset_x,y[0]+offset_y),(x[1]+offset_x,y[1]+offset_y)]))
        new_edges_list.append([(x[0]+offset_x,y[0]+offset_y),(x[1]+offset_x,y[1]+offset_y)])
    x_min=-(width-1)/2-1
    x_max=width-(width+1)/2
    y_min=1-(height+1)/2-1
    y_max=height-(height+1)/2
    new_edges.append(LineString([(x_min, y_min), (x_min, y_max)]))
    new_edges_list.append([(x_min, y_min), (x_min, y_max)])
    new_edges.append(LineString([(x_max, y_min), (x_max, y_max)]))
    new_edges_list.append([(x_max, y_min), (x_max, y_max)])
    new_edges.append(LineString([(x_min, y_min), (x_max, y_min)]))
    new_edges_list.append([(x_min, y_min), (x_max, y_min)])
    new_edges.append(LineString([(x_min, y_max), (x_max, y_max)]))    
    new_edges_list.append([(x_min, y_max), (x_max, y_max)])      
    return new_edges,new_edges_list



def generate_random_point(area_size, obstacles, min_distance):
    """Generate a point that is at least min_distance away from any obstacle."""
    while True:
        # Generate a random point within the specified area
        point = Point(random.uniform(-area_size/2, area_size/2), random.uniform(-area_size/2, area_size/2))
        
        # Check the distance to all obstacles
        valid = True
        for obstacle in obstacles:
            if point.distance(obstacle) < min_distance:
                valid = False
                break
        
        if valid:
            return point

# Plotting the maze edges using shapely geometries
def plot_maze_edges(maze_edges):
    """Plot the edges of the maze using Matplotlib."""
    # fig, ax = plt.subplots(figsize=(10, 10))
    fig, ax = plt.subplots()
    lines = []
    for edge in maze_edges:
        x, y = edge.xy
        lines.append([(x[0],y[0]),(x[1],y[1])])
        ax.plot(x, y, color='black', linewidth=2)
    # print(lines)
    # Set plot limits and display
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect('equal')
    plt.title("Maze with Shapely Edges")
    return ax
    

# Plot the maze edges
# Get the edges of the generated maze
# maze_edges,new_edges_list = get_maze_edges(maze)
import pickle
# with open('maze_edges.pkl', 'wb') as f:
#     pickle.dump(new_edges_list, f)
with open('maze_edges_points.pkl', 'rb') as f:
    new_edges_list,coords_list = pickle.load(f)
# print(coords_list)
maze_edges=[]
for ed in new_edges_list:
    maze_edges.append(LineString(ed))
ax=plot_maze_edges(maze_edges)
# points=[]
# while len(points)<100:
#     point = generate_random_point(20, maze_edges, 0.5)
#     collison=0
#     for p in points:
#         if point.distance(p) < 1.5:
#             collison=1
#             break
#     if collison==0:
#         points.append(point)
# print(points)
# Plot points
x_coords = [p[0] for p in coords_list]
y_coords = [p[1] for p in coords_list]
# x_coords = [p.x for p in points]
# y_coords = [p.y for p in points]
# coords_list=[(p.x,p.y) for p in points]
# with open('maze_edges_points.pkl', 'wb') as f:
#     pickle.dump([new_edges_list,coords_list], f)
ax.scatter(x_coords, y_coords, color='b', label="Points")
plt.show()
# plt.savefig('maze_with100point.png')
