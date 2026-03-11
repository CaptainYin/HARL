# import pyrvo2
import numpy as np
import math, time
from pyrvo2 import *
import matplotlib.pyplot as plt
from SimEnv import line_to_rectangle,get_edges, SimulationEnvironment_Scenario2, SimulationEnvironment_Scenario6, \
    SimulationEnvironment_CubeCollection, SimulationEnvironment_Warehouse, SimulationEnvironment_CA
from shapely.geometry import Polygon, Point, LineString
import os,sys
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.getcwd(), "../../../MinMax-MTSP-master/partition"))  
from env_core import EnvCore

def setupScenario(sim,env):
    goals=[]
    obstale_lists = []
    width=0.01
    for line in env.SimEnv.inner_lines:
        rectangle = line_to_rectangle(line, width)
        obstale_lists.append(rectangle)
    for line in env.SimEnv.border:# for boundary, clockwise
        #reverse order
        obstale_lists.append([line[i] for i in range(len(line)-2,-1,-1)])
    for circ in env.SimEnv.circle_obs:
        po=Point(circ[0],circ[1]).buffer(circ[2],quad_segs=4)
        polist=list(po.exterior.coords)
        polist.reverse()#reverse order
        obstale_lists.append(polist)
        
    for obstacle in env.SimEnv.rectangle_obs:
        obstale_lists.append(list(obstacle.exterior.coords))
    sim.setTimeStep(0.1)
    sim.setAgentDefaults(neighborDist=5, maxNeighbors=32, timeHorizon=5.0, timeHorizonObst=1.0, radius=0.2, maxSpeed=1.0, velocity=np.array([0,0]))
    for car in env.cars:
        sim.addAgent(np.array([car['x'],car['y']]))
        goals.append(np.array([car['goal_x'],car['goal_y']]))
    for obstale_list in obstale_lists:
        sim.addObstacle(obstale_list)
    sim.processObstacle()
    return goals,obstale_lists

def setPreferredVelocities(sim, goals):
    for i in range(sim.getNumAgents()):
        goalVector = goals[i] - sim.getAgentPosition(i)
        if (np.linalg.norm(goalVector) > 1.0):
            goalVector = goalVector/np.linalg.norm(goalVector)
        sim.setAgentPrefVelocity(i, goalVector)

def reachedGoal(sim, goals):
    for i in range(sim.getNumAgents()):
        if (np.linalg.norm(sim.getAgentPosition(i) - goals[i]) > sim.getAgentRadius(i)):
            return False
    return True

def visualize(sim,obstale_lists):
    plt.clf()  # Clear the previous frame
    
    # Plot agents
    for i in range(sim.getNumAgents()):
        pos = sim.getAgentPosition(i)
        radius = sim.getAgentRadius(i)
        circle = plt.Circle((pos[0], pos[1]), radius, color='blue', alpha=0.5)
        plt.gca().add_artist(circle)
    
    # Plot obstacles
    for obstale_list in obstale_lists:
        x = [point[0] for point in obstale_list]
        y = [point[1] for point in obstale_list]
        plt.plot(x, y, 'r-', linewidth=2)
        # Close the obstacle shape
        plt.plot([x[-1], x[0]], [y[-1], y[0]], 'r-', linewidth=2)
    
    # Set plot limits (equivalent to your world_size)
    plt.xlim(-10, 10)  # Adjust based on your world_size
    plt.ylim(-10, 10)  # Adjust based on your world_size
    plt.axis('equal')  # Equal aspect ratio
    plt.grid(True)
    # plt.title(f"Simulation Time: {sim.getGlobalTime():.1f}")
    plt.pause(0.1)  # Pause to show the animation

# Create the environment
env = EnvCore({'scenario':"Warehouse"})
env.seed(42)  # For reproducibility

# Reset the environment
observations = env.reset()
world_size=env.SimEnv.area_size

# Initialize the plot
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(10, 10))

start_time = time.time()
sim = RVOSimulator()
goals,obstale_lists = setupScenario(sim,env)

while not reachedGoal(sim, goals):
    setPreferredVelocities(sim, goals)
    sim.step()
    visualize(sim,obstale_lists)

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final frame visible
