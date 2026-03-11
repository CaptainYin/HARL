import matplotlib.pyplot as plt
import numpy as np
import os,sys
sys.path.append(os.path.join(os.getcwd(), "../")) 
from harl.envs.lasercar.SimEnv import SimulationEnvironment_Unmaze,SimulationEnvironment_CA,SimulationEnvironment_CubeCollection,SimulationEnvironment_Warehouse,\
    SimulationEnvironment_Scenario1,SimulationEnvironment_Scenario2,SimulationEnvironment_Scenario3,SimulationEnvironment_Scenario4,\
    SimulationEnvironment_Scenario6,SimulationEnvironment_Scenario7,SimulationEnvironment_maze
    
# env=SimulationEnvironment_Unmaze()
# env.display_environment(show=0,savefig='fig/Environment_Unmaze.png')
# env=SimulationEnvironment_CA()
# env.display_environment(show=0,savefig='fig/Environment_CA.png')
# env=SimulationEnvironment_CubeCollection()
# env.display_environment(show=0,savefig='fig/Environment_Cube.png')
# env=SimulationEnvironment_Warehouse()
# env.display_environment(show=0,savefig='fig/Environment_Warehouse.png')
# env=SimulationEnvironment_Scenario1()
# env.display_environment(show=0,savefig='fig/Environment_Scenario1.png')
# env=SimulationEnvironment_Scenario2()
# env.display_environment(show=0,savefig='fig/Environment_Scenario2.png')
# env=SimulationEnvironment_Scenario3()
# env.display_environment(show=0,savefig='fig/Environment_Scenario3.png')
# env=SimulationEnvironment_Scenario6()
# env.display_environment(show=0,savefig='fig/Environment_Scenario6.png')


env=SimulationEnvironment_Unmaze()
env.display_environment(show=0,savefig='fig/Environment_Unmaze_position.png')
env=SimulationEnvironment_CA()
env.display_environment(show=0,savefig='fig/Environment_CA_position.png')
env=SimulationEnvironment_CubeCollection()
env.display_environment(show=0,savefig='fig/Environment_Cube_position.png')
env=SimulationEnvironment_Warehouse()
env.display_environment(show=0,savefig='fig/Environment_Warehouse_position.png')
env=SimulationEnvironment_Scenario1()
env.display_environment(show=0,savefig='fig/Environment_Scenario1_position.png')
env=SimulationEnvironment_Scenario2()
env.display_environment(show=0,savefig='fig/Environment_Scenario2_position.png')
env=SimulationEnvironment_Scenario3()
env.display_environment(show=0,savefig='fig/Environment_Scenario3_position.png')
env=SimulationEnvironment_Scenario6()
env.display_environment(show=0,savefig='fig/Environment_Scenario6_position.png')
