import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
import sys
import os
# sys.path.append(os.path.dirname(__file__))
from offpolicy.envs.lasercar.env_core import EnvCore

# Create a parameters class to hold all the required arguments
@dataclass
class Args:
    num_sensors: int = 16
    Car_resettype: str = "random_target"  # Options: "random_target", "diag"
    num_agents: int = 4
    turnoff_sensor: bool = False
    add_sensor_noise: bool = True
    fixed_theta: bool = True
    scenario: str = "scenario6"  # Use an available scenario
    share_reward: bool = False
    sensor_gussian_noise: float = 0.1
    save_history: bool = True
    plot_traj: bool = True
    traj_hold: bool = True
    plot_sensor_line: bool = True
    ifi: float = 0.1  # Animation interval

# Simple policy that moves toward the goal with obstacle avoidance
def simple_policy(observation, num_sensors):
    # For visualization-only simulation, we'll use a simple rule-based policy
    # Extract relevant information from observation
    if len(observation) > num_sensors + 7:  # Check if we have sensor data and state
        # Extract sensor readings from the observation
        sensors = observation[:num_sensors]
        
        # Extract car state information
        theta = observation[num_sensors*3]  # Current orientation
        goal_x = observation[-2]
        goal_y = observation[-1]
        x = observation[-5]
        y = observation[-4]
        
        # Calculate direction to goal
        angle_to_goal = np.arctan2(goal_y - y, goal_x - x)
        
        # Calculate angle difference
        angle_diff = (angle_to_goal - theta) % (2 * np.pi)
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
            
        # Check for obstacles directly ahead
        front_sensors = sensors[7:10]  # Middle sensors
        
        # Base speed
        speed = 0.8  # Default speed
        
        # Obstacle avoidance
        if np.min(front_sensors) < 1.0:
            # Obstacle detected, turn away from it
            min_sensor_idx = np.argmin(sensors)
            turn_direction = 1.0 if min_sensor_idx > num_sensors // 2 else -1.0
            return np.array([0.3, turn_direction * 0.8])  # Slow down and turn
        
        # Normal goal-seeking behavior
        return np.array([speed, np.clip(angle_diff * 1.5, -1.0, 1.0)])
    else:
        # Fallback if observation doesn't contain expected data
        return np.array([0.5, 0.0])


# Set up the parameters
args = Args()

# Create the environment
env = EnvCore(args)
env.seed(42)  # For reproducibility

# Reset the environment
observations = env.reset()

# Create lists to store trajectories for each agent
trajectories = [[] for _ in range(args.num_agents)]

# Run the simulation for a fixed number of steps
total_steps = 100  # Adjust as needed
print("Starting simulation...")

done = False
step_count = 0

while not done and step_count < total_steps:
    # Get actions using our simple policy
    actions = [simple_policy(obs, args.num_sensors) for obs in observations]
    
    # Take a step in the environment
    observations, rewards, dones, infos = env.step(actions)
    
    
    # Store trajectory data for visualization
    for i, info in enumerate(infos):
        if 'history' in info and len(info['history']) > 0:
            trajectories[i].append(info['history'][-1])

    
    step_count += 1
    done = all(dones)
    
    # Print status periodically
    if step_count % 10 == 0:
        print(f"Step {step_count}, rewards: {rewards}")

print("Simulation completed!")
print(f"Final rewards: {rewards}")
print(f"Final statuses: {['Goal reached' if car['reached_goal'] else 'Collision' if car['obs_collision'] or car['car_collision'] else 'Timeout' for car in infos]}")

# Render the trajectories
print("Rendering simulation...")
env.render(trajectory=trajectories, mode='human', gif_name="car_simulation.gif")
