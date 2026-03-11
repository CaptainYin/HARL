import random
from shapely.geometry import Point, Polygon,MultiPoint
from shapely.ops import nearest_points
import numpy as np
def generate_convex_obstacle(center, size, area_size,num_vertices=6):
    """Generate a convex polygon obstacle with the given center and size."""
    x_center, y_center = center

    # Generate random points around the center
    points = []
    for _ in range(num_vertices):
        angle = random.uniform(0, 2 * np.pi)  # Random angle
        radius = random.uniform(size * 0.5, size)  # Random radius within size
        x = x_center + np.cos(angle) * radius
        y = y_center + np.sin(angle) * radius
        if x>area_size/2 or x<-area_size/2 or y>area_size/2 or y<-area_size/2:
            continue        
        points.append((x, y))

    # Create a convex hull around these points to ensure convexity
    convex_hull = MultiPoint(points).convex_hull

    return convex_hull
def generate_concave_obstacle(center, size,area_size, num_vertices=8, irregularity_factor=0.3):
    """
    Generate a concave polygon obstacle with the given center and size.
    
    Parameters:
    - center: (x, y) tuple for the center of the polygon.
    - size: Maximum radius of the concave polygon.
    - num_vertices: Number of vertices for the polygon.
    - irregularity_factor: Controls the concavity. The higher the value, the more concave the shape.
    """
    x_center, y_center = center
    points = []
    
    # for _ in range(num_vertices):
    while len(points)<num_vertices:
        # Random angle for polar coordinates
        angle = random.uniform(0, 2 * np.pi)
        
        # Add randomness to the radius to create concavity
        radius = size * (1 + irregularity_factor * (random.random() - 0.5))
        
        # Convert polar coordinates to Cartesian
        x = x_center + np.cos(angle) * radius
        y = y_center + np.sin(angle) * radius
        if x>area_size/2 or x<-area_size/2 or y>area_size/2 or y<-area_size/2:
            continue
        points.append((x, y))

    # Ensure the points form a valid polygon by sorting them by angle around the center
    points.sort(key=lambda p: np.arctan2(p[1] - y_center, p[0] - x_center))

    # Return a concave polygon created from the points
    return Polygon(points)

def generate_obstacle(center, size):
    """Generate a square-shaped obstacle with the given center and size."""
    x, y = center
    half_size = size / 2
    return Polygon([
        (x - half_size, y - half_size),
        (x + half_size, y - half_size),
        (x + half_size, y + half_size),
        (x - half_size, y + half_size)
    ])

def generate_random_point(area_size, obstacles,points, min_distance):
    """Generate a point that is at least min_distance away from any obstacle and existing points."""
    while True:
        # Generate a random point within the specified area
        point = Point(random.uniform(-area_size/2, area_size/2), random.uniform(-area_size/2, area_size/2))
        
        # Check the distance to all obstacles
        valid = True
        for obstacle in obstacles:
            if point.distance(obstacle) < min_distance:
                valid = False
                break
        for p in points:
            if point.distance(p) < min_distance:
                valid = False
                break
        if valid:
            return point

def generate_obstacles_and_points(existing_points,area_size, num_obstacles, num_points, obstacle_size,num_vertices,irregularity_factor, min_distance):
    """Generate a set of obstacles and a set of points that do not overlap and maintain a minimum distance."""
    obstacles = []
    points = []

    # Step 1: Generate obstacles
    for _ in range(num_obstacles):
        while True:
            # Generate a random center for the obstacle
            center = (random.uniform(-area_size/2, area_size/2), random.uniform(-area_size/2, area_size/2))
            # obstacle = generate_obstacle(center, obstacle_size)
            # obstacle = generate_convex_obstacle(center, obstacle_size,area_size)
            obstacle = generate_concave_obstacle(center, obstacle_size,area_size, num_vertices=num_vertices, irregularity_factor=irregularity_factor)
            
            # Ensure no overlap between obstacles
            overlap = False
            for existing_obstacle in obstacles:
                if obstacle.intersects(existing_obstacle) or obstacle.distance(existing_obstacle)<min_distance:
                    overlap = True
                    break
            for existing_point in existing_points:
                p=Point(existing_point)
                if p.distance(obstacle)<min_distance:
                    overlap = True
                    break                    
            if not overlap:
                obstacles.append(obstacle)
                break

    # Step 2: Generate points that are at least `min_distance` away from any obstacle
    for _ in range(num_points):
        point = generate_random_point(area_size, obstacles,points, min_distance)
        points.append(point)

    return obstacles, points
if __name__ == '__main__':
    # Example usage
    area_size = 20  # Define the area size (e.g., a 20x20 meter area)
    num_obstacles = 15  # Number of obstacles
    num_points = 5  # Number of points to generate
    obstacle_size = 2  # Size of each obstacle (e.g., 2 meters square)
    min_distance = 0.6  # Minimum distance between points and obstacles,obstacles and obstacles (0.6 meters)
    num_vertices=8
    irregularity_factor=1
    # Generate obstacles and points
    obstacles, points = generate_obstacles_and_points(area_size-2, num_obstacles, num_points, obstacle_size, num_vertices,irregularity_factor,min_distance)

    # Print the generated obstacles and points
    print("Generated Obstacles (as polygons):")
    for obstacle in obstacles:
        print(obstacle)

    print("\nGenerated Points (coordinates):")
    for point in points:
        print(point)

    # Optional: You can visualize the result using Matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in obstacles:
        patch = MplPolygon(list(obstacle.exterior.coords), closed=True, edgecolor='r', facecolor='gray')
        ax.add_patch(patch)

    # Plot points
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    ax.scatter(x_coords, y_coords, color='b', label="Points")

    # Set limits and display the plot
    ax.set_xlim(-area_size / 2, area_size / 2)
    ax.set_ylim(-area_size / 2, area_size / 2)
    ax.set_aspect('equal', 'box')
    plt.legend()
    plt.show()
