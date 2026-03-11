import pyrvo2
import numpy as np
import math, time
from pyrvo2 import *
import matplotlib.pyplot as plt
from SimEnv import get_edges, line_to_rectangle,SimulationEnvironment_Scenario2, SimulationEnvironment_Scenario6, \
    SimulationEnvironment_CubeCollection, SimulationEnvironment_Warehouse, SimulationEnvironment_CA
from shapely.geometry import Polygon, Point, LineString
from offpolicy.envs.lasercar.env_core import EnvCore

class Args:
    num_sensors: int = 16
    Car_resettype: str = "diag"  # Options: "random_target", "diag"
    num_agents: int = 4
    turnoff_sensor: bool = False
    add_sensor_noise: bool = True
    fixed_theta: bool = True
    scenario: str = "Unmaze"  # Use an available scenario
    share_reward: bool = False
    sensor_gussian_noise: float = 0.1
    save_history: bool = True
    plot_traj: bool = True
    traj_hold: bool = True
    plot_sensor_line: bool = True
    ifi: float = 0.1  # Animation interval

# Set up the parameters
args = Args()

# Create the environment
env = EnvCore(args)
env.seed(42)  # For reproducibility

# Reset the environment
observations = env.reset()
world_size=env.SimEnv.area_size
obstale_lists = []
obstale_list = []
width=0.05

for line in env.SimEnv.border:# for boundary, clockwise, reverse order
    obstale_lists.append([line[i] for i in range(len(line)-2,-1,-1)])
for line in env.SimEnv.inner_lines:
    rectangle = line_to_rectangle(line, width)
    obstale_lists.append(rectangle)
for obstacle in env.SimEnv.rectangle_obs:
    obstale_lists.append(list(obstacle.exterior.coords))
for obstacle in env.SimEnv.circle_obs:
    A=list(obstacle.exterior.coords)
    A.reverse()
    obstale_lists.append(A)

print(f"obstacle num {len(obstale_lists)}")

# exit(0)
def setupScenario(sim):
    goals = []
    start = []
    sim.setTimeStep(0.1)
    sim.setAgentDefaults(5, 32, 5.0, 1.0, 0.2, 1.0, np.array([0,0]))
    for car in env.cars:
        sim.addAgent(np.array([car['x'], car['y']]))
        start.append(np.array([car['x'], car['y']]))
        goals.append(np.array([car['goal_x'], car['goal_y']]))
    for obstale_list in obstale_lists:
        sim.addObstacle(obstale_list)
    sim.processObstacle()
    return start,goals

def setPreferredVelocities(sim, goals):
    # print(len(goals),goals)
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


import matplotlib.animation as animation

def update_frame(frame_num, world_size, agent_positions, goals, radii, anim_ax):
    anim_ax.clear()
    
    # Plot agents
    for i in range(len(agent_positions[frame_num])):
        pos = agent_positions[frame_num][i]
        radius = radii[i]
        circle = plt.Circle((pos[0], pos[1]), radius, color='blue', alpha=0.5)
        anim_ax.add_artist(circle)
    
    # # Plot obstacles
    # for obstale_list in obstale_lists:
    #     x = [p[0] for p in obstale_list]
    #     y = [p[1] for p in obstale_list]
    #     anim_ax.fill(x, y, 'gray', alpha=1)
    
    # Plot filled obstacles
    broder_x=[point[0] for point in obstale_lists[0]]+[obstale_lists[0][0][0]]
    broder_y=[point[1] for point in obstale_lists[0]]+[obstale_lists[0][0][1]]
    anim_ax.plot(broder_x,broder_y, 'black',linewidth=5, alpha=1)
    for obstale_list in obstale_lists[1:]:
        x = [point[0] for point in obstale_list]
        y = [point[1] for point in obstale_list]
        anim_ax.fill(x, y, 'gray', alpha=1)  # Fill the obstacle with red color and 50% transparency
    
    
    # Set plot limits and title
    anim_ax.set_xlim(-world_size/2-1, world_size/2+1)
    anim_ax.set_ylim(-world_size/2-1, world_size/2+1)
    anim_ax.set_aspect('equal')
    anim_ax.grid(True)
    anim_ax.set_title(f"Simulation Time: {frame_num}", fontsize=16, fontweight='bold')
    
    return []

def save_animation(filename="rvo_simulation_unmaze.gif", max_frames=200):
    anim_fig, anim_ax = plt.subplots(figsize=(8, 8))
    
    sim = RVOSimulator()
    start, goals = setupScenario(sim)
    num_agents = sim.getNumAgents()
    radii = [sim.getAgentRadius(i) for i in range(num_agents)]
    
    agent_positions = []
    current_positions = [sim.getAgentPosition(i) for i in range(num_agents)]
    agent_positions.append(current_positions)
    
    frame_count = 0
    reached = False
    while frame_count < max_frames and not reached:
        if reachedGoal(sim, goals):
            reached = True
            break
        setPreferredVelocities(sim, goals)
        sim.step()
        current_positions = [sim.getAgentPosition(i) for i in range(num_agents)]
        agent_positions.append(current_positions)
        frame_count += 1
    
    total_frames = len(agent_positions)
    print(f"Animation saved with {total_frames} frames")
    
    anim = animation.FuncAnimation(
        anim_fig, 
        update_frame, 
        frames=total_frames,
        fargs=(world_size, agent_positions, goals, radii, anim_ax),
        interval=100,
        blit=False
    )
    
    writer = animation.PillowWriter(fps=10) if filename.endswith('.gif') else animation.FFMpegWriter(fps=10)
    anim.save(filename, writer=writer)
    print(f"Animation saved to {filename}")

save_animation(f"rvo_simulation_{args.scenario}.gif")
# Or save as GIF
# save_animation("rvo_simulation.gif")


def visualize(sim,timestep):
    plt.clf()  # Clear the previous frame
    
    # Plot agents
    for i in range(sim.getNumAgents()):
        pos = sim.getAgentPosition(i)
        radius = sim.getAgentRadius(i)
        circle = plt.Circle((pos[0], pos[1]), radius, color='blue', alpha=0.5)
        plt.gca().add_artist(circle)
    
    # Plot filled obstacles
    broder_x=[point[0] for point in obstale_lists[0]]+[obstale_lists[0][0][0]]
    broder_y=[point[1] for point in obstale_lists[0]]+[obstale_lists[0][0][1]]

    plt.plot(broder_x,broder_y, 'black',linewidth=5, alpha=1)
    for obstale_list in obstale_lists[1:]:
        x = [point[0] for point in obstale_list]
        y = [point[1] for point in obstale_list]
        plt.fill(x, y, 'gray', alpha=1)  # Fill the obstacle with red color and 50% transparency
    
    # Set plot limits
    plt.xlim(-world_size/2-1, world_size/2+1)
    plt.ylim(-world_size/2-1, world_size/2+1)
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Simulation Time: {timestep}",fontsize=16,fontweight='bold')
    plt.pause(0.1)  # Pause to show the animation


# # Initialize the plot
# plt.ion()  # Turn on interactive mode
# fig = plt.figure(figsize=(8, 8))

# start_time = time.time()
# sim = RVOSimulator()
# start, goals = setupScenario(sim)
# timestep=0
# while not reachedGoal(sim, goals):
#     setPreferredVelocities(sim, goals)
#     sim.step()
#     timestep+=1
#     visualize(sim,timestep)

# plt.ioff()  # Turn off interactive mode
# plt.show()
