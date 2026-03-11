
import sys,os,time,numba,gymnasium,random,imageio,lap,torch
sys.path.append(os.path.join(os.getcwd(), "../../MinMax-MTSP-master/partition"))  

from lib.layers.Actor_low import CMPNN as CMPNNAMARLmodel
from lib.layers.Actor_low_topk import Model as AMARLmodel
from lib.layers.AMARL_edgeaware import Model as EAGAT
from lib.layers.Actor_low_CMPNNGAT import CMPNNGAT 
from lib.layers.partitionNet import mtspsq
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Point
import multiprocessing as mp
from harl.envs.lasercar.pyrvo2 import *
from numba.typed import List
from gym import spaces
from harl.envs.lasercar.SimEnv1 import sigmoid,check_collision,greedyassignment,line_to_rectangle,get_edges,SimulationEnvironment_Unmaze,SimulationEnvironment_CA,SimulationEnvironment_CubeCollection,SimulationEnvironment_Warehouse,\
    SimulationEnvironment_Scenario1,SimulationEnvironment_Scenario2,SimulationEnvironment_Scenario3,SimulationEnvironment_Scenario4,\
    SimulationEnvironment_Scenario6,SimulationEnvironment_Scenario7,SimulationEnvironment_maze
scenario_list={'maze':SimulationEnvironment_maze,'CA':SimulationEnvironment_CA,'Cube':SimulationEnvironment_CubeCollection,'Warehouse':SimulationEnvironment_Warehouse,'Unmaze':SimulationEnvironment_Unmaze,\
    'scenario1':SimulationEnvironment_Scenario1,'scenario2':SimulationEnvironment_Scenario2,'scenario3':SimulationEnvironment_Scenario3,'scenario6':SimulationEnvironment_Scenario6}
for i in range(5):scenario_list['scenario4_'+str(i)]=(SimulationEnvironment_Scenario4,i)
for i in range(25):scenario_list['scenario7_'+str(i)]=(SimulationEnvironment_Scenario7,i)
for i in range(46):scenario_list['Warehouse_'+str(i)]=(SimulationEnvironment_Warehouse,i)
for i in range(22):scenario_list['Cube_'+str(i)]=(SimulationEnvironment_CubeCollection,i)
import get_sensor_data as get_sensor_data_C
from get_sensor_data import check_car_collision
class EnvArgs:
    """Configuration arguments for the EnvCore class."""
    def __init__(self, **kwargs):
        # Default values
        self.num_sensors = 16
        self.Car_resettype = "random_target"  # Options: random_target, diag, diag_fixed
        self.num_agents = 20
        self.goal_num=20
        self.turnoff_sensor = False
        self.add_sensor_noise = False
        self.fixed_theta = False
        self.scenario = "Cube"
        self.share_reward = False
        self.sensor_gussian_noise = 0.1
        self.save_history = False
        self.plot_traj = True
        self.traj_hold = False
        self.plot_sensor_line = True
        self.ifi = 0.1
        self.reward_type = "shaping"  # Options: binary, shaping, pR, lp_sn, lp_wosn
        self.modelfile =''
        self.state_type= 'FP'
        self.task='MTSP1'
        self.num_samples=100
        self.k=20
        self.metric="euclidean"
        self.assignment= "greedy1"
        self.assignmentEachStep = False
        self.fixed_maxsteps = False
        self.obs_goal=True
        self.reachgoalset = True
        self.neighborDist = 2
        self.maxNeighbors = 10
        self.timeHorizon = 1
        self.timeHorizonObst = 1
        self.delta=0.1       
        self.default_speed = 2
        self.default_omega = np.pi
        self.safety_distance = 0.2
        self.emergency_dis = 0.6
        self.PID_calaction_dis = 3.5  # Distance threshold for PID action calculation
        self.disk_radius = 0.2
        self.sensor_range = 3.5
        self.EnvId=0
        self.reborn=False
        # Override defaults with provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Ignoring unknown parameter '{key}'")
    @classmethod
    def from_dict(cls, arg_dict):
        """Create an EnvArgs instance from a dictionary"""
        return cls(**arg_dict)                
class EnvCore(gymnasium.Env):
    """
    # 环境中的智能体
    """
    metadata = {"render.modes": ["human", "txt"]}
    def __init__(self,all_args):
        # args = EnvArgs(num_sensors=32, scenario="CA", num_agents=8)
        all_args = EnvArgs.from_dict(all_args)
        #=============================Env=============================
        self.EnvId=all_args.EnvId
        self.delta = all_args.delta#second
        self.episode = 0
        self.task=all_args.task
        self.reachgoalset=all_args.reachgoalset
        self.obs_goal=all_args.obs_goal
        self.metric=all_args.metric
        self.assignment=all_args.assignment
        self.assignmentEachStep=all_args.assignmentEachStep    
        self.scenario=all_args.scenario
        
        self.reborn=all_args.reborn
        self.fixed_maxsteps=all_args.fixed_maxsteps
        self.obs_crash_count=0
        self.car_crash_count=0
        self.mrta_count=0
        # print("reborn",self.reborn)
        
        if isinstance(scenario_list[self.scenario],tuple):
            self.SimEnv=scenario_list[self.scenario][0](idx=scenario_list[self.scenario][1])
        else:
            self.SimEnv=scenario_list[self.scenario]() 
        self.SimEnv.counter=self.EnvId
        self.area_size = self.SimEnv.area_size
        self.Env_circle_obs=self.SimEnv.circle_obs
        self.Env_rectangle_obs=self.SimEnv.rectangle_obs
        self.Env_obstacles = self.SimEnv.obstacles
        self.Env_Stringline=self.SimEnv.Stringline
        self.Env_edges_list=self.SimEnv.Env_edges_list
        self.Env_edges_list_len=len(self.Env_edges_list)
        #=============================agent=============================
        self.num_agent=all_args.num_agents
        self.n_agents=all_args.num_agents
        self.safety_distance = all_args.safety_distance
        self.emergency_dis = all_args.emergency_dis
        self.PID_calaction_dis = all_args.PID_calaction_dis  # Distance threshold for PID action calculation
        self.disk_radius = all_args.disk_radius
        self.fixed_theta=all_args.fixed_theta
        self.Car_resettype=all_args.Car_resettype #random_target,diag
        self.default_speed = all_args.default_speed
        self.default_omega = all_args.default_omega        
        #=============================sensor=============================
        self.num_sensors = all_args.num_sensors
        self.turnoff_sensor=all_args.turnoff_sensor
        self.add_sensor_noise=all_args.add_sensor_noise
        self.sensor_range = all_args.sensor_range
        self.sensor_gussian_noise=all_args.sensor_gussian_noise
        self.device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.device=torch.device("cuda:{}".format(all_args.EnvId%4) if torch.cuda.is_available() else "cpu")
        # self.device=torch.device("cpu")
        #=============================obs dim=============================
        if self.reachgoalset:
            self.goal_num=all_args.goal_num
            self.num_samples=all_args.num_samples
            self.k=all_args.k
            self.SimEnv.build_roadmap(num_samples=self.num_samples,k=self.k)
            self.goal_obs_dim=3 #(rel_x,rel_y,reach_flag)
            self.other_obs_dim=2*self.goal_num #(rel_x,rel_y,theta,v,omega)
            self.obs_dim_ego=self.num_sensors if not self.turnoff_sensor else 0
            self.his_len=1
            # self.obs_dim = self.obs_dim_ego*self.his_len+3+self.other_obs_dim*(self.num_agent-1)+ 2*self.goal_num if not self.obs_goal else self.obs_dim_ego+5
            self.obs_dim = self.obs_dim_ego*self.his_len+3+ 2 if not self.obs_goal else self.obs_dim_ego+5
            self.state_dim_peragent=5 
            self.share_observation_dim= self.state_dim_peragent*self.num_agent +self.goal_obs_dim*self.goal_num
            self.max_steps = self.goal_num/self.num_agent*2*self.area_size/(self.delta*self.default_speed) # Maximum number of steps per episode, depending on the environment and task
            self.cost_matrix=np.zeros((self.num_agent,self.goal_num),dtype=np.float32)
        else:
            self.modelfile=all_args.modelfile
            self.goal_num=all_args.goal_num
            self.num_samples=all_args.num_samples
            self.k=all_args.k
            if self.metric=='dijstra' and self.assignment in ['greedy','lapjv','AMARL','EAGAT','CMPNN','CMPNNGAT','DisPn']:
                self.SimEnv.build_roadmap(num_samples=self.num_samples,k=self.k)
            self.other_obs_dim=7
            self.state_dim_peragent=7
            self.obs_dim_ego=self.num_sensors if not self.turnoff_sensor else 0
            self.his_len=1
            self.obs_dim = self.obs_dim_ego*self.his_len+5
            self.share_observation_dim= self.state_dim_peragent*self.num_agent
            self.max_steps = self.goal_num/self.num_agent*2*self.area_size/(self.delta*self.default_speed) # Maximum number of steps per episode, depending on the environment and task
            # self.max_steps = self.goal_num*2*self.area_size/(self.delta*self.default_speed)
        if self.fixed_maxsteps:
            self.max_steps=10000
        self.cars = self.generate_cars_Env()
        self.obs_his=np.zeros((self.num_agent,self.his_len,self.obs_dim_ego),dtype=np.float32)# his_length=3,0-2:t-2,t-1,t
        self.state=np.zeros(self.share_observation_dim,dtype=np.float32)
        self.action_dim = 2  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional
        self.action_space = [spaces.Box(low=np.array([0, -self.default_omega],dtype=np.float32), high=np.array([self.default_speed, self.default_omega],dtype=np.float32), dtype=np.float32) for _ in range(self.num_agent)]
        # self.action_space = [spaces.Tuple((spaces.Box(low=np.array([0, -self.default_omega],dtype=np.float32), high=np.array([self.default_speed, self.default_omega],dtype=np.float32), dtype=np.float32),spaces.Discrete(2))) for _ in range(self.num_agent)] # the second part is whether to use NH-ORCA
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]
        self.share_observation_space = [
            # spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim * self.num_agent,), dtype=np.float32) #shared observation is concatenated observation of all agents
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.share_observation_dim,), dtype=np.float32) #shared observation is environment state
            for _ in range(self.num_agent)]
        #=============================reinforcement learning=============================
        self.reward_type=all_args.reward_type#binary,shaping,pR,lp_sn,lp_wosn
        self.share_reward=all_args.share_reward
        #=============================NH_ORCA=============================
        self.D = self.default_speed/self.default_omega-self.disk_radius # effective_distance
        self.L = 2*self.disk_radius #wheelDist
        self.neighborDist = all_args.neighborDist
        self.maxNeighbors = all_args.maxNeighbors
        self.timeHorizon = all_args.timeHorizon
        self.timeHorizonObst = all_args.timeHorizonObst
        self.startVel = np.array([0,0])
        self.simulator = RVOSimulator()
        self.simulator.setTimeStep(self.delta)
        self.simulator.setAgentDefaults(self.neighborDist, self.maxNeighbors, self.timeHorizon,self.timeHorizonObst, self.disk_radius, self.default_speed, self.startVel)
        self.NHORCA_setupScenario()
        #=============================render=============================
        self.save_history=all_args.save_history
        if self.save_history: 
            self.history=[[] for _ in range(self.num_agent)]
            self.goal_reach_flag_his=[]
        self.plot_traj=all_args.plot_traj
        self.traj_hold=all_args.traj_hold
        self.plot_sensor_line=all_args.plot_sensor_line
        self.ifi=all_args.ifi
        # if self.plot_traj:
        #     self.fig, self.ax = plt.subplots()
        #=============================performance test=============================

    def NHORCA_setupScenario(self):
        goals,obstale_lists=[],[]
        width=0.01
        for line in self.SimEnv.inner_lines:
            rectangle = line_to_rectangle(line, width)
            obstale_lists.append(rectangle)
        for line in self.SimEnv.border:# for boundary, clockwise
            obstale_lists.append([line[i] for i in range(len(line)-2,-1,-1)])
        for circ in self.SimEnv.circle_obs:
            po=Point(circ[0],circ[1]).buffer(circ[2],quad_segs=4)
            polist=list(po.exterior.coords)
            polist.reverse()#reverse order
            obstale_lists.append(polist)
        for obstacle in self.SimEnv.rectangle_obs:
            obstale_lists.append(list(obstacle.exterior.coords))
        for car in self.cars:
            self.simulator.addAgent(np.array([car['x'],car['y']]))
            goals.append(np.array([car['goal_x'],car['goal_y']]))
        for obstale_list in obstale_lists:
            self.simulator.addObstacle(obstale_list)
        self.simulator.processObstacle()
        return goals,obstale_lists
        
    def NHORCA_cal_effective_cmd(self, pref_vel,theta):
        A = 0.5*np.cos(theta)+self.D*np.sin(theta)/self.L
        B = 0.5*np.cos(theta)-self.D*np.sin(theta)/self.L
        C = 0.5*np.sin(theta)-self.D*np.cos(theta)/self.L
        D = 0.5*np.sin(theta)+self.D*np.cos(theta)/self.L
        vx = pref_vel[0]
        vy = pref_vel[1]
        vr = (vy-C/A*vx)/(D-B*C/A)
        vl = (vx-B*vr)/A

        omega =   (vr-vl)/self.L
        omega= np.clip(-self.default_omega,omega,self.default_omega)
        vel =0.5*(vl+vr)
        vel = np.clip(0,vel,self.default_speed)
        return np.array([vel, omega])

    def NHORCA_getEffectivePos(self,pos,theta):
        return np.array(pos)+self.D*np.array([np.cos(theta),np.sin(theta)])

    def NHORCA_getEffectiveVel(self,w,v,theta):
        vr = v+0.5*w*self.L
        vl = 2*v - vr
        x_vel = (0.5*np.cos(theta)+self.D*np.sin(theta)/self.L)*vl +  (0.5*np.cos(theta)-self.D*np.sin(theta)/self.L)*vr
        y_vel = (0.5*np.sin(theta)-self.D*np.cos(theta)/self.L)*vl +  (0.5*np.sin(theta)+self.D*np.cos(theta)/self.L)*vr
        return np.array([x_vel,y_vel])

    def NHORCA_calaction(self):
        for i in range(self.num_agent):
            # self.simulator.setAgentPosition(i, self.NHORCA_getEffectivePos([self.cars[i]['x'],self.cars[i]['y']],self.cars[i]['theta']))
            # self.simulator.setAgentVelocity(i, self.NHORCA_getEffectiveVel(self.cars[i]['omega'],self.cars[i]['v'],self.cars[i]['theta']))
            self.simulator.setAgentPosition(i, np.array([self.cars[i]['x'],self.cars[i]['y']]))
            v=self.cars[i]['v']
            theta=self.cars[i]['theta']
            self.simulator.setAgentVelocity(i, np.array([v*np.cos(theta),v*np.sin(theta)]))
            goalVector = np.array([self.cars[i]['goal_x']-self.cars[i]['x'],self.cars[i]['goal_y']-self.cars[i]['y']])
            if  np.linalg.norm(goalVector) > 0.1:
                goalVector = goalVector/np.linalg.norm(goalVector)
            if self.cars[i]['obs_collision'] or self.cars[i]['car_collision'] or self.cars[i]['reached_goal'] or self.cars[i]['timeout']:
                goalVector = np.array([0,0])
            self.simulator.setAgentPrefVelocity(i, self.default_speed*goalVector)
        self.simulator.step()
        action=[]
        for i in range(self.num_agent):# get simulated velocity for each robot
            velocity =  self.simulator.getAgentVelocity(i)
            # action.append(self.NHORCA_cal_effective_cmd(velocity,self.cars[i]['theta']))
            theta=self.cars[i]['theta']
            omega=(np.arctan2(velocity[1],velocity[0])-theta)/self.delta
            action.append(np.array([np.linalg.norm(velocity),omega]))
        action = np.array(action)
        return action
    
    def SRLORCA_calaction(self,prefVel):
        for i in range(self.num_agent):
            self.simulator.setAgentPosition(i, np.array([self.cars[i]['x'],self.cars[i]['y']]))
            v=self.cars[i]['v']
            theta=self.cars[i]['theta']
            self.simulator.setAgentVelocity(i, np.array([v*np.cos(theta),v*np.sin(theta)]))
            
            if self.cars[i]['obs_collision'] or self.cars[i]['car_collision'] or self.cars[i]['reached_goal'] or self.cars[i]['timeout']:
                goalVector = np.array([0,0])
            else:
                v_p,w_p=prefVel[i]
                new_theta=self.cars[i]['theta']+w_p*self.delta
                goalVector = np.array([v_p * np.cos(new_theta),v_p * np.sin(new_theta)])
            self.simulator.setAgentPrefVelocity(i, goalVector)
        self.simulator.step()
        action=[]
        for i in range(self.num_agent):# get simulated velocity for each robot
            velocity =  self.simulator.getAgentVelocity(i)
            theta=self.cars[i]['theta']
            omega=(np.arctan2(velocity[1],velocity[0])-theta)/self.delta
            action.append(np.array([np.linalg.norm(velocity),omega]))
        action = np.array(action)
        return action
    
    def PID_calaction(self):
        action=[]
        for i in range(self.num_agent):
            if self.cars[i]['obs_collision'] or self.cars[i]['car_collision'] or self.cars[i]['reached_goal'] or self.cars[i]['timeout']:
                action.append(np.array([0,0]))
                continue
            rho,phi =self.cars[i]['rho'],self.cars[i]['phi']
            phi = (phi + np.pi) % (2 * np.pi) - np.pi
            if abs(phi)<0.0001:
                action.append(np.array([np.clip(rho/self.delta,-self.default_speed,self.default_speed),0]))
            else:
                action.append(np.array([0,np.clip(phi/self.delta,-self.default_omega,self.default_omega)]))
        return np.array(action)
    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
            np.random.seed(1)
        else:
            random.seed(seed)
            np.random.seed(seed)
            # self.EnvId=seed
    def new_car(self,x,y,theta,goal_x,goal_y,i):
        car = {'x': x, 'y': y, 'theta': theta,'assigned':1,'changegoal':0,'goal_x': goal_x, 'goal_y': goal_y, 'goal_list':[],'omega': 0, 'v': 0,'previous_x':0,'previous_y':0,'rho':0,'phi':0,'prev_rho':0,'prev_phi':0,
                    'min_laser': 0, 'prev_min_laser': 0,'min_laser_index': 0, 'prev_min_laser_index': 0,'updated':1,
                'obs_collision': False, 'car_collision': False, 'reached_goal': False, 'timeout': False, 
                'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_agent+i,'history':[]}
        if self.reachgoalset:
            car['rho']=[np.sqrt((-car['x']+goal[0])**2 + (-car['y']+goal[1])**2) for goal in self.goals]
            car['phi']=[np.arctan2(-car['y']+goal[1],-car['x']+goal[0]) - car['theta'] for goal in self.goals]
            car['phi']=[ (phi + np.pi) % (2 * np.pi) - np.pi for phi in car['phi']]
        else:
            car['rho'],car['phi'] =np.sqrt((-car['x']+car['goal_x'])**2 + (-car['y']+car['goal_y'])**2),np.arctan2(-car['y']+car['goal_y'],-car['x']+car['goal_x']) - car['theta']
        return car
    def cal_cost_matrix(self,src_num,goal_num,src_pos,goal_pos):
        # print(len(src_pos),len(goal_pos),src_pos)
        cost_matrix=np.zeros((src_num,goal_num),dtype=np.float32)
        if self.metric=='dijstra':
            src,goal=[],[]
            for i in range(src_num):
                for j in range(goal_num):
                    src.append(src_pos[i])
                    goal.append(goal_pos[j])
            prm_path = self.SimEnv.find_path(start_set=src_pos,goal_set=goal_pos,start=src, goal=goal)
            for i in range(src_num):
                for j in range(goal_num):
                    cost_matrix[i][j]=prm_path[i*goal_num+j]
        elif self.metric=='euclidean':
            for i in range(src_num):
                for j in range(goal_num):
                    cost_matrix[i][j]=np.sqrt((src_pos[i][0]-goal_pos[j][0])**2+(src_pos[i][1]-goal_pos[j][1])**2)                  
        return cost_matrix
    def assignment_cost_matrix(self,src_num,goal_num,src_pos,goal_pos):
        cost_matrix=self.cal_cost_matrix(src_num,goal_num,src_pos,goal_pos)                
        # solve the assignment problem
        if self.assignment=='lapjv':
            total_cost, goal_assignment,agent_assignment = lap.lapjv(cost_matrix, extend_cost=True)
        elif self.assignment=='greedy':
            total_cost, goal_assignment,agent_assignment= greedyassignment(cost_matrix)
        return total_cost, goal_assignment,agent_assignment
    def reborn_car(self):
        for car_id in range(self.num_agent):
            if self.cars[car_id]['obs_collision'] or self.cars[car_id]['car_collision']:
                if self.cars[car_id]['obs_collision']: self.obs_crash_count+=1
                if self.cars[car_id]['car_collision']: self.car_crash_count+=1
                original_goal_x,original_goal_y=self.cars[car_id]['goal_x'],self.cars[car_id]['goal_y']
                orig_step=self.cars[car_id]['STEP']
                x,y=self.src[car_id]
                theta=self.get_init_theta(x,y,original_goal_x,original_goal_y)
                self.cars[car_id]=self.new_car(x,y,theta,original_goal_x,original_goal_y,car_id)
                self.cars[car_id]['STEP']=orig_step
    def generate_cars_Env(self):
        cars = []
        self.goal_reach_flag=np.zeros(self.goal_num,dtype=np.float32)# 0 not reached, k reached by agent k-1
        if self.reachgoalset:

            position=self.SimEnv.sample_MTSP_from_graph(num_agent=self.num_agent,num_goal=self.goal_num)
            self.src=position[:self.num_agent]
            self.goals=position[self.num_agent:]
            self.reached_all=False

            for i,pos in enumerate(self.src):
                x,y=pos
                goal_x,goal_y=self.goals[0]# the goal not used
                cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i))
        else:
            # src_pos=random.sample(position,self.num_agent)
            if self.Car_resettype=='diag': #random_target,diag
                self.position=self.SimEnv.positions.copy()
                self.src=self.position[:self.num_agent]
                self.goals=[]
                for i,pos in enumerate(self.src):
                    x,y=pos
                    pos_id=i
                    target_id=pos_id+1 if pos_id%2==0 else pos_id-1
                    self.goals.append(self.SimEnv.positions[target_id])
                    goal_x,goal_y = self.SimEnv.positions[target_id][0],self.SimEnv.positions[target_id][1]
                    cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i))
                self.goal_assignment=[i for i in range(self.num_agent)]
            elif self.Car_resettype=='random_target':
                if self.metric=='dijstra' and self.assignment in ['greedy','lapjv'] or self.assignment in ['AMARL','EAGAT','CMPNN','CMPNNGAT','DisPn']:
                    if self.fixed_theta:
                        self.position=self.SimEnv.sample_MTSP_from_graph_fix(num_agent=self.num_agent,num_goal=self.goal_num)
                    else:
                        self.position=self.SimEnv.sample_MTSP_from_graph(num_agent=self.num_agent,num_goal=self.goal_num)
                else:
                    self.position=self.SimEnv.sample_MTSP(num_agent=self.num_agent,num_goal=self.goal_num)
                self.src=self.position[:self.num_agent]
                self.goals=self.position[self.num_agent:]                    
                
                if self.assignment in ['AMARL','EAGAT','CMPNN','CMPNNGAT','DisPn']:
                    self.mrta_count+=1
                    self.reached_all=False
                    position=torch.tensor(self.position).unsqueeze(0).to(self.device)
                    if self.assignment in ['AMARL','EAGAT']:
                        if self.assignment=='AMARL':
                            self.model =EAGAT(input_node_dim=2, num_agents=self.num_agent, hidden_node_dim=128, input_edge_dim=1, hidden_edge_dim=16, topk=10,conv_laysers=4,use_encoder_edge_cost=False, use_decoder_cost_bias=False,device=self.device) 
                        else:
                            self.model =EAGAT(input_node_dim=4, num_agents=self.num_agent, hidden_node_dim=128, input_edge_dim=1, hidden_edge_dim=16, topk=10,conv_laysers=4,use_encoder_edge_cost=False, use_decoder_cost_bias=True,device=self.device)            
                        checkpoint = torch.load(self.modelfile, map_location=self.device,weights_only=True)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.model.eval()
                        self.cost_matrix=self.cal_cost_matrix(self.num_agent+self.goal_num,self.num_agent+self.goal_num,self.position,self.position)
                        self.cost_matrix=torch.tensor(self.cost_matrix).unsqueeze(0)
                        self.cost_matrix.diagonal(dim1=-2, dim2=-1).zero_()
                        logits, tours = self.model(position,steps=position.size(1),greedy=True,anum=self.num_agent,cost_matrix=self.cost_matrix)
                        
                    elif self.assignment in ['CMPNN','CMPNNGAT']:
                        if self.assignment=='CMPNNGAT':
                            self.model = CMPNNGAT(input_node_dim=2, num_agents=4, hidden_node_dim=128, topk=1, device=self.device)
                        else:
                            self.model = CMPNNAMARLmodel(input_node_dim=2, num_agents=4, hidden_node_dim=128, topk=1, device=self.device)
                        checkpoint = torch.load(self.modelfile, map_location=self.device,weights_only=True)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.model.eval()
                        self.cost_matrix=self.cal_cost_matrix(self.num_agent+self.goal_num,self.num_agent+self.goal_num,self.position,self.position)
                        knn=np.argsort(self.cost_matrix,axis=1)[:,:self.k]
                        knn=torch.tensor(knn[:,1:]).unsqueeze(0).to(self.device)
                        logits, tours = self.model(position,knn,steps=position.size(1),greedy=True, instance_num=1,anum=self.num_agent)#CMPNN                        
                    elif self.assignment=='DisPn':
                        self.model = mtspsq(1, [2, 16, 16, 64, 64, 128, 128], [2, 16], self.num_agent, self.num_agent+self.goal_num, False, False).to(self.device)
                        checkpoint = torch.load(self.modelfile, map_location=self.device,weights_only=True)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.model.eval()
                        self.cost_matrix=self.cal_cost_matrix(self.num_agent+self.goal_num,self.num_agent+self.goal_num,self.position,self.position)
                        knn=np.argsort(self.cost_matrix,axis=1)[:,:self.k]
                        knn=torch.tensor(knn[:,1:]).unsqueeze(0).to(self.device)
                        logits, tours = self.model(position.permute(0,2,1), knn, maxsample=True, instance_num=2,anum=self.num_agent)#DISPN

                    
                    position = position[0].cpu().numpy()
                    tours = tours[0][0]
                    # print(tours)
                    self.agent_goal = [[] for _ in range(self.num_agent)]
                    self.goal_assignment=[]
                    for i in range(self.num_agent):
                        x,y=self.position[tours[i][0]]
                        if len(tours[i])==1:
                            goal_x,goal_y = x,y
                            car =self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i)
                            car['assigned']=0
                            self.agent_goal[i]=[]
                            cars.append(car)
                            self.goal_assignment.append(-1)
                        else:
                            if len(tours[i])>2:
                                self.agent_goal[i]=[position[tours[i][j]] for j in range(2,len(tours[i]))]
                            else:
                                self.agent_goal[i]=[]
                            goal_x,goal_y = self.position[tours[i][1]]
                            cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i)) 
                            self.goal_assignment.append(self.goals.index((goal_x,goal_y)))
                elif self.assignment in ['greedy','lapjv']:
                    self.mrta_count+=1
                    total_cost, self.goal_assignment,agent_assignment= self.assignment_cost_matrix(self.num_agent,self.goal_num,self.src,self.goals)
                    for i,pos in enumerate(self.src):
                        x,y=pos
                        goal_x,goal_y=self.goals[self.goal_assignment[i]]
                        cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i))   
                else:
                    for i in range(self.num_agent):
                        x,y=self.position[i]
                        goal_x,goal_y = self.goals[i]
                        cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i))
                    self.goal_assignment=[i for i in range(self.num_agent)]

            elif self.Car_resettype=='diag_fixed':
                src_pos_ids=self.SimEnv.fixed_eval_task[self.EnvId%len(self.SimEnv.fixed_eval_task)]
                for i,pos_id in enumerate(list(src_pos_ids)):
                    target_id=pos_id+1 if pos_id%2==0 else pos_id-1
                    x,y=self.SimEnv.positions[pos_id][0],self.SimEnv.positions[pos_id][1]
                    goal_x,goal_y = self.SimEnv.positions[target_id][0],self.SimEnv.positions[target_id][1]
                    cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i))
            else:
                raise NotImplementedError  
        return cars
    def get_init_theta(self,x,y,goal_x,goal_y):
        if self.fixed_theta:
            return np.arctan2(-y+goal_y,-x+goal_x)
        else:
            return np.random.uniform(-np.pi, np.pi)
    def update_state(self, actions):
        self.crashedOrreached=False# record if any car reached the goal or crashed this step
        self.crashed = False
        for car, action in zip(self.cars, actions):
            if car['obs_collision'] or car['car_collision'] or car['timeout'] or car['reached_goal']:
                car['updated']=0
                continue

            car['updated']=1
            car['STEP']+=1
            # print(f"step {car['STEP']} car_id:{car['car_id']} ")
            if car['STEP']>=self.max_steps:
                car['timeout']=True
            v, w = action if car['assigned'] else (0, 0)
            # v, w = action
            car['previous_omega'],car['previous_v']=car['omega'],car['v']
            car['v'],car['omega'],car['previous_x'],car['previous_y'],theta0 = v,w,car['x'],car['y'],car['theta'] 
            new_x,new_y = car['x'],car['y']
            for _ in range(4):
                car['theta'] += w * self.delta/4
                new_x = new_x + v * np.cos(car['theta']) * self.delta/4
                new_y = new_y + v * np.sin(car['theta']) * self.delta/4
            car['theta'] = (car['theta'] + np.pi) % (2 * np.pi) - np.pi
            car['x'],car['y'] = np.clip(new_x, -self.area_size/2, self.area_size/2),np.clip(new_y, -self.area_size/2, self.area_size/2)

            if not self.reachgoalset:
                car['prev_rho'],car['prev_phi']=car['rho'],car['phi']
                car['rho'],car['phi'] =np.sqrt((-car['x']+car['goal_x'])**2 + (-car['y']+car['goal_y'])**2),np.arctan2(-car['y']+car['goal_y'],-car['x']+car['goal_x']) - car['theta']
                car['phi']= (car['phi'] + np.pi) % (2 * np.pi) - np.pi      
            else:
                car['prev_rho']=car['rho'].copy()
                car['prev_phi']=car['phi'].copy()
                car['rho']=[np.sqrt((-car['x']+goal[0])**2 + (-car['y']+goal[1])**2) for goal in self.goals]
                car['phi']=[np.arctan2(-car['y']+goal[1],-car['x']+goal[0]) - car['theta'] for goal in self.goals]
                car['phi']=[ (phi + np.pi) % (2 * np.pi) - np.pi for phi in car['phi']]          
            if check_collision(car['x'], car['y'],self.Env_rectangle_obs,self.Env_circle_obs,self.Env_Stringline,self.safety_distance):
                car['obs_collision'] = True
                self.crashedOrreached=True
                self.crashed=True
            elif check_car_collision([car['x'], car['y']], [[c['x'],c['y']] for c in self.cars if c != car],self.safety_distance):
                car['car_collision'] = True
                self.crashedOrreached=True
                self.crashed=True
            else:
                if not self.reachgoalset:
                    if car['rho'] < self.safety_distance:
                        # print(self.goals,[car['goal_x'],car['goal_y']])
                        # print(np.nonzero(np.linalg.norm(np.array(self.goals)-np.array([car['goal_x'],car['goal_y']]),axis=1)<0.01))
                        # print(f"step:{car['STEP']}, car_id:{car['car_id']} reach goal {np.nonzero(np.linalg.norm(np.array(self.goals)-np.array([car['goal_x'],car['goal_y']]),axis=1)<0.01)[0][0]}",car['assigned'],self.agent_goal[car['car_id']% self.num_agent])
                        if car['assigned'] and self.goal_reach_flag[np.nonzero(np.linalg.norm(np.array(self.goals)-np.array([car['goal_x'],car['goal_y']]),axis=1)<0.01)[0][0]]==0:#or not car['assigned']
                            car['reached_goal'] = True
                            self.crashedOrreached = True
                            self.goal_reach_flag[self.goal_assignment[car['car_id'] % self.num_agent]] = car['car_id'] % self.num_agent + 1
                else:
                    car_rho_=car['rho'].copy()
                    for i in range(self.goal_num):
                        if self.goal_reach_flag[i]!=0:
                            car_rho_[i]=np.inf
                    if np.min(car_rho_) < self.safety_distance:
                        car['reached_goal'] = True
                        self.crashedOrreached=True
                        self.goal_reach_flag[np.argmin(car_rho_)]=car['car_id']%self.num_agent+1
                    elif np.min(car_rho_)==np.inf:
                        self.reached_all=True

    def reasign_goal(self):
        # for car in self.cars:
        #     car['changegoal']=0
        #if any goal is reached, or any car crashed in this step, change goal assignment
        # if (self.crashedOrreached or self.assignmentEachStep) and self.task=='MTSP' and not self.reachgoalset:
        #     raise NotImplementedError('see version 20250512')
        if self.crashedOrreached and not self.reachgoalset and self.assignment in ["AMARL",'EAGAT',"CMPNN","CMPNNGAT","DisPn"]:
            finished=False
            if not self.crashed:
                for i, car in enumerate(self.cars):
                    if car['reached_goal'] and car['updated']:
                        if len(self.agent_goal[i])==0:
                            finished=True
                            self.goal_assignment[i]=-1
                            break
                        else:
                            self.cars[i]['goal_x'],self.cars[i]['goal_y']=self.agent_goal[i].pop(0)
                            # if car['car_id'] ==0:
                                # print(f"step:{car['STEP']}, car_id:{car['car_id']} new goal {np.nonzero(np.linalg.norm(np.array(self.goals)-np.array([self.cars[i]['goal_x'],self.cars[i]['goal_y']]),axis=1)<0.01)[0][0]}",self.agent_goal[i])
                            self.cars[i]['reached_goal']=False
                            # self.goal_assignment[i]=self.goals.index((self.cars[i]['goal_x'],self.cars[i]['goal_y']))
                            self.goal_assignment[i]=np.nonzero(np.linalg.norm(np.array(self.goals)-np.array([self.cars[i]['goal_x'],self.cars[i]['goal_y']]),axis=1)<0.01)[0][0]
                            # self.cars[i]['changegoal']=1
            if self.crashed or finished:
                remain_agent_index=[i for i in range(self.num_agent) if self.cars[i]['car_collision']==0 and self.cars[i]['obs_collision']==0 and self.cars[i]['timeout']==0]
                remain_goal_index=[i for i in range(self.goal_num) if self.goal_reach_flag[i]==0]
                if len(remain_goal_index)!=0 and len(remain_agent_index)!=0:
                    curpos=[[self.cars[idx]['x'],self.cars[idx]['y']] for idx in remain_agent_index]
                    # goal=torch.tensor(self.position[self.num_agent:],dtype=torch.float32)[torch.tensor(remain_goal_index)]
                    goal=[self.goals[i] for i in remain_goal_index]
                    position_list=curpos+goal
                    position=torch.tensor(position_list,dtype=torch.float32).unsqueeze(0).to(self.device)
                    position_list=[tuple(pos) for pos in position_list]
                    self.mrta_count+=1
                    if self.assignment in ['AMARL','EAGAT']:
                        self.cost_matrix=self.cal_cost_matrix(position.size(1),position.size(1),position_list,position_list)
                        self.cost_matrix=torch.tensor(self.cost_matrix).unsqueeze(0)
                        self.cost_matrix.diagonal(dim1=-1,dim2=-2).zero_()
                        logits, tours = self.model(position,steps=position.size(1),greedy=True,anum=len(remain_agent_index),cost_matrix=self.cost_matrix)
                    elif self.assignment in ['CMPNN','CMPNNGAT']:
                        self.cost_matrix=self.cal_cost_matrix(position.size(1),position.size(1),position_list,position_list)
                        knn=np.argsort(self.cost_matrix,axis=1)[:,:self.k]
                        knn=torch.tensor(knn[:,1:]).unsqueeze(0).to(self.device)
                        logits, tours = self.model(position,knn,steps=position.size(1),greedy=True, instance_num=1,anum=len(remain_agent_index))#CMPNN                     
                    elif self.assignment=='DisPn':
                        self.cost_matrix=self.cal_cost_matrix(position.size(1),position.size(1),position_list,position_list)
                        knn=np.argsort(self.cost_matrix,axis=1)[:,:self.k]
                        knn=torch.tensor(knn[:,1:]).unsqueeze(0).to(self.device)
                        logits, tours = self.model(position.permute(0,2,1), knn, maxsample=True, instance_num=2,anum=len(remain_agent_index))#DISPN 
                                    
                    position = position[0].cpu().numpy()
                    tours = tours[0][0]
                    
                    # def tours_invalid(tours):
                    #     someagentempty=False
                    #     agentempty=-1
                    #     someagentoverflow=False
                    #     agentoverflow=-1
                    #     for i, tour in enumerate(tours):
                    #         if len(tour)==1:
                    #             someagentempty=True
                    #             agentempty=i
                    #         elif len(tour)>2:
                    #             someagentoverflow=True
                    #             agentoverflow=i
                    #     if someagentoverflow and someagentempty:
                    #         return agentempty, agentoverflow, True
                    #     else:
                    #         return agentempty, agentoverflow,False
                    # agentempty, agentoverflow,invalid_tour =tours_invalid(tours)
                    # while invalid_tour:
                    #     tours[agentempty].append(tours[agentoverflow].pop())
                    #     agentempty, agentoverflow,invalid_tour =tours_invalid(tours)    
                                          
                    # tours_c=[]
                    # for i, idx in enumerate(remain_agent_index):
                    #     tours_c.append([])
                    #     if len(tours[i])==1:
                    #         tours_c[i].append(tours[i][0])
                    #         continue
                    #     else:
                    #         tours_c[i].append(tours[i][0])
                    #         for j in range(1,len(tours[i])):
                    #             # print(np.array(position[tours[i][j]]),np.nonzero(np.linalg.norm(np.array(self.goals)-np.array(position[tours[i][j]]),axis=1)<0.01))
                    #             tours_c[i].append(np.nonzero(np.linalg.norm(np.array(self.goals)-np.array(position[tours[i][j]]),axis=1)<0.01)[0][0])
                    # print(self.cars[0]['STEP'],tours_c)
                    # 
                    for i, idx in enumerate(remain_agent_index):
                        if len(tours[i])==1:
                            self.cars[idx]['assigned']=0
                            self.agent_goal[idx]=[]
                            self.cars[idx]['reached_goal']=False
                            self.goal_assignment[idx]=-1
                            # someagentempty=True
                        else:
                            if len(tours[i])>2:
                                self.agent_goal[idx]=[position[tours[i][j]] for j in range(2,len(tours[i]))]
                                # if someagentempty:
                                #     print(f"remain_agent_index:{len(remain_agent_index)},remain goal num {len(remain_goal_index)}")
                            else:
                                self.agent_goal[idx]=[]
                            self.cars[idx]['goal_x'],self.cars[idx]['goal_y']=position[tours[i][1]]
                            self.cars[idx]['reached_goal']=False
                            self.cars[idx]['assigned']=1
                            # print(self.goals)
                            # print(tours[i], len(remain_agent_index),[self.cars[idx]['goal_x'],self.cars[idx]['goal_y']],position[tours[i][1]], np.nonzero(np.linalg.norm(np.array(self.goals)-np.array([self.cars[idx]['goal_x'],self.cars[idx]['goal_y']]),axis=1)<0.01))
                            self.goal_assignment[idx]=np.nonzero(np.linalg.norm(np.array(self.goals)-np.array([self.cars[idx]['goal_x'],self.cars[idx]['goal_y']]),axis=1)<0.01)[0][0]
                else:
                    if len(remain_goal_index)==0:
                        self.reached_all=True
                        for car in self.cars:car['reached_goal']=True# for test only
        if self.crashedOrreached and not self.reachgoalset and self.assignment in ["lapjv","greedy"]:
            remain_agent_index=[i for i in range(self.num_agent) if self.cars[i]['car_collision']==0 and self.cars[i]['obs_collision']==0 and self.cars[i]['timeout']==0]
            remain_goal_index=[i for i in range(self.goal_num) if self.goal_reach_flag[i]==0]
            if len(remain_goal_index)!=0 and len(remain_agent_index)!=0:
                self.mrta_count+=1
                start_set=[(self.cars[remain_agent_index[i]]['x'],self.cars[remain_agent_index[i]]['y']) for i in range(len(remain_agent_index))]
                goal_set=[self.goals[remain_goal_index[i]] for i in range(len(remain_goal_index))]
                _, goal_assignment,_= self.assignment_cost_matrix(len(remain_agent_index),len(remain_goal_index),start_set,goal_set)
                for i in range(len(remain_agent_index)):
                    if goal_assignment[i]==-1:#means no goal assigned
                        self.cars[remain_agent_index[i]]['assigned']=0
                        self.cars[remain_agent_index[i]]['goal_x'],self.cars[remain_agent_index[i]]['goal_y']=self.cars[remain_agent_index[i]]['x'],self.cars[remain_agent_index[i]]['y']
                    else:
                        goal_assignment[i]=remain_goal_index[goal_assignment[i]]
                        self.cars[remain_agent_index[i]]['assigned']=1
                        self.cars[remain_agent_index[i]]['reached_goal']=False
                        self.cars[remain_agent_index[i]]['goal_x'],self.cars[remain_agent_index[i]]['goal_y']=self.goals[goal_assignment[i]]
            else:
                if len(remain_goal_index)==0:
                    self.reached_all=True
                    for car in self.cars:car['reached_goal']=True# for test only
    def get_observation(self, car):
        if car['updated']==0:
            return np.full(self.obs_dim,0)
        car['prev_min_laser']=car['min_laser']
        car['prev_min_laser_index']=car['min_laser_index']
        if not self.turnoff_sensor:
            sensor_data = get_sensor_data_C.get_sensor_data1(car['car_id'],self.Env_edges_list,self.num_sensors,self.sensor_range,car['theta'],car['x'], car['y'],\
                [(ocar['x'],ocar['y'],self.disk_radius) for ocar in self.cars]+ self.Env_circle_obs) 
            if self.add_sensor_noise:
                sensor_data=np.clip(sensor_data+np.random.normal(loc=0, scale=self.sensor_gussian_noise,size=sensor_data.shape),self.disk_radius,self.sensor_range) #add gaussian noise to sensor
            car['min_laser_index'] = np.argmin(sensor_data)
            car['min_laser'] = sensor_data[car['min_laser_index']]
            self.obs_his[car['car_id']%self.num_agent,:-1,:]=self.obs_his[car['car_id']%self.num_agent,1:,:].copy()
            self.obs_his[car['car_id']%self.num_agent,-1,:]=sensor_data.copy()
            if car['STEP']==0 and self.his_len>1:
                for i in range(self.his_len-1):
                    self.obs_his[car['car_id']%self.num_agent,i,:]=self.obs_his[car['car_id']%self.num_agent,-1,:].copy()
            observation=self.obs_his[car['car_id']%self.num_agent,:,:].reshape(-1)
            if self.reachgoalset and not self.obs_goal:
                # obs_other=np.zeros((self.num_agent-1)*self.other_obs_dim,dtype=np.float32)
                # i=0
                # for other_car in self.cars:
                #     if other_car['car_id']!=car['car_id']:
                #         obs_other[i*self.other_obs_dim:(i+1)*self.other_obs_dim]=np.array(other_car['rho']+other_car['phi'])
                #         i+=1
                # observation = np.concatenate((observation,[car['theta'],car['omega'],car['v']],obs_other,car['rho'],car['phi']))
                idx=np.argmin(car['rho'])

                # observation = np.concatenate((observation,[car['theta'],car['omega'],car['v']],car['rho'],car['phi']))
                observation = np.concatenate((observation,[car['theta'],car['omega'],car['v']],[car['rho'][idx],car['phi'][idx]]))
            else:
                observation = np.concatenate((observation,[car['theta'],car['omega'],car['v'],car['rho'],car['phi']]))
            if self.save_history:
                self.history[car['car_id']%self.num_agent].append(np.concatenate((sensor_data,np.array([car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y'],\
                    car['timeout'],car['obs_collision'], car['car_collision'],car['reached_goal'],car['omega'],car['v']]))))
        else:
            raise NotImplementedError("see version 20250512")
        return observation
    def get_state(self):
        if self.reachgoalset:
        # self.state_dim_peragent=self.num_sensors+5 #(x,y,theta,v,omega)
        #self.share_observation_dim= self.state_dim_peragent*self.num_agent +self.goal_obs_dim*self.goal_num
            for i,car in enumerate(self.cars):
                if car['updated']==0:
                    self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[0]*self.state_dim_peragent
                else:
                    # observation=self.obs_his[car['car_id']%self.num_agent,:,:].reshape(-1)
                    # self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=np.concatenate((observation,[car['theta'],car['rho'],car['phi'],car['omega'],car['v'],car['x'],car['y']]))
                    self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=np.array(([car['theta'],car['rho'],car['phi'],car['omega'],car['v'],car['x'],car['y']])) if not self.reachgoalset else np.array(([car['theta'],car['omega'],car['v'],car['x'],car['y']]))
                    # self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=np.array(([car['theta'],car['rho'],car['phi'],car['omega'],car['v']]))
                    # self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=np.array(([car['theta'],car['rho'],car['phi'],car['omega'],car['v'],car['min_laser'],car['min_laser_index'],car['x'],car['y']]))
                    
            for i in range(self.goal_num):
                goal_x,goal_y=self.goals[i]
                self.state[self.num_agent*self.state_dim_peragent+i*self.goal_obs_dim:self.num_agent*self.state_dim_peragent+(i+1)*self.goal_obs_dim]=[goal_x,goal_y,self.goal_reach_flag[i]]
        else:
            for i,car in enumerate(self.cars):
                if car['updated']==0:#out
                    self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[0]*self.state_dim_peragent
                else:
                    self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[car['theta'],car['rho'],car['phi'],car['omega'],car['v'],car['min_laser'],car['min_laser_index']]
        # return np.expand_dims(self.state,axis=0).repeat(self.num_agent,axis=0)
        return self.state

    def reset(self):
        self.cars = self.generate_cars_Env()
        if self.save_history: 
            self.history=[[] for _ in range(self.num_agent)]
            self.goal_reach_flag_his=[]
        self.episode += 1
        self.obs_crash_count=0
        self.car_crash_count=0
        self.mrta_count=0
        observations = [self.get_observation(car) for car in self.cars]
        return observations,self.get_state(),None
        # return observations

    def get_reward(self):
        rewards=np.zeros([self.num_agent,1],dtype=np.float32)
        for i,car in enumerate(self.cars):
            if car['updated']==0:#  if car['obs_collision'] or car['car_collision'] or car['timeout'] in last step and current step have not updated, then no reward
                continue
            if self.reachgoalset:
                if self.reward_type=='reachall':
                    rewards[i][0]=(car['prev_rho']-car['rho'])*3 if car['changegoal']==0 else 0
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]-=100
                    elif car['reached_goal']:
                        rewards[i][0]+=10
                        car['reached_goal']=False
                    if self.reached_all:
                        rewards[i][0]+=100
                elif self.reward_type=='shaping':

                    prev_rho=np.min(car['prev_rho'])
                    rho=np.min(car['rho'])
                    rewards[i][0]=(prev_rho-rho)*3
                    # ##Eq 12 in "End-to-End Deep Reinforcement Learning for Decentralized Task Allocation and Navigation for a Multi-Robot System"
                    # rho_=np.array(car['rho'].copy())
                    # # print(rho_,self.goal_reach_flag!=0)
                    # rho_[self.goal_reach_flag!=0]=np.inf
                    # goalidx=np.argmin(rho_)
                    # othercar_dist_min=np.min([ocar['rho'][goalidx] for ocar in self.cars if ocar!=car])
                    # rewards[i][0]=rewards[i][0]+0.35*np.log10(othercar_dist_min)+0.3409
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]-=10
                    elif car['reached_goal']:
                        rewards[i][0]+=100
                        car['reached_goal']=False
                    # if self.reached_all:
                    #     rewards[i][0]+=100
                elif self.reward_type=='binary':
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]-=100
                    elif car['reached_goal']:
                        rewards[i][0]+=10
                        car['reached_goal']=False
  
                elif self.reward_type=='binaryreachall':
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]-=100
                    elif car['reached_goal']:
                        rewards[i][0]+=10
                        car['reached_goal']=False
                    if self.reached_all:
                        rewards[i][0]+=100       
                elif self.reward_type=='CYB':
                    rewards[i][0]=(car['prev_rho']-car['rho'])*2-1.5 if car['changegoal']==0 else 0
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]-=50
                    elif car['reached_goal']:
                        rewards[i][0]+=120
                        car['reached_goal']=False
                    # if self.reached_all:
                    #     rewards[i][0]+=100                
            else:
                if self.reward_type=='binary':
                    # if car['reached_goal']:
                    #     rewards[i][0]=0
                    # else:
                    #     rewards[i][0]=-1
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]+=-10
                    if car['reached_goal']:
                        rewards[i][0]+=100                    
                elif self.reward_type=='shaping':
                    rewards[i][0]=(car['prev_rho']-car['rho'])*3 if car['changegoal']==0 else 0
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]+=-10
                    if car['reached_goal']:
                        rewards[i][0]+=100
                elif self.reward_type=='long2018':
                    rewards[i][0]=(car['prev_rho']-car['rho'])*2.5 if car['changegoal']==0 else 0
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]+=-15
                    if car['reached_goal']:
                        rewards[i][0]+=15
                    if abs(car['omega'])>0.7*self.default_omega:
                        rewards[i][0]+=-0.1*abs(car['omega'])/self.default_omega
                elif self.reward_type=='sn':
                    reward_laser = -np.exp( -18* (np.clip(car['min_laser']/self.sensor_range,self.safety_distance/self.sensor_range,np.inf)-0.1)) * 0.2577 * ( 3 - 2.5 * (1 - car['min_laser_index']/self.num_sensors ))#(-1,0)
                    rewards[i][0]=(car['prev_rho']-car['rho'])*3+reward_laser
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]+=-10
                    if car['reached_goal']:
                        rewards[i][0]+=100
                else:
                    raise NotImplementedError
            car['changegoal']=0
        
        # return rewards
        if self.share_reward:
            return [[np.mean([r.sum() for r in rewards])]]*self.num_agent
        else:
            return [[r.sum()] for r in rewards]
    
    def step(self, Actions=None):
        if self.task=="NHORCA" or Actions is None:
            actions=self.NHORCA_calaction()
        else:
            actions=10*Actions[:,:2]
            actions[:,0]=sigmoid(actions[:,0])*self.default_speed
            np.tanh(actions[:,1],out=actions[:,1])
            actions[:,1]=actions[:,1]*self.default_omega
            if self.task=='hybrid':
                safe_actions = self.NHORCA_calaction()
                pid_actions = self.PID_calaction()
                for i,car in enumerate(self.cars):
                    if car['min_laser']>self.PID_calaction_dis or car['min_laser']<car['rho']:
                        actions[i]=pid_actions[i]
                    elif car['min_laser']<self.emergency_dis:
                        actions[i]=safe_actions[i]
            elif self.task=='hybridorca':
                safe_actions = self.NHORCA_calaction()
                # pid_actions = self.PID_calaction()
                for i,car in enumerate(self.cars):
                    if car['min_laser']<self.emergency_dis:
                        actions[i]=safe_actions[i]
            elif self.task == 'Hybrid':
                safe_actions = self.NHORCA_calaction()
                pid_actions = self.PID_calaction()
                for i,car in enumerate(self.cars):
                    if Actions[i,2]==0:
                        actions[i]=pid_actions[i]
                    elif Actions[i,2]==1:
                        actions[i]=safe_actions[i]
            elif self.task=='Hybridorca':
                safe_actions = self.NHORCA_calaction()
                for i,car in enumerate(self.cars):
                    if Actions[i,2]==0:
                        actions[i]=safe_actions[i]
            elif self.task == 'SRLORCA':
                actions=self.SRLORCA_calaction(prefVel=actions)

        self.update_state(actions)
        observations = [self.get_observation(car) for car in self.cars]
        if self.save_history:
            self.goal_reach_flag_his.append(self.goal_reach_flag.copy())
        # observations=[obs.astype(np.float16) for obs in observations]
        rewards = self.get_reward()
        shared_obs=self.get_state()
        if self.reborn:
            self.reborn_car()
        self.reasign_goal()
        if self.reachgoalset:
            dones=[car['obs_collision'] or car['car_collision'] or car['timeout'] or self.reached_all for car in self.cars]
            if np.all(dones) and self.save_history:
                max_step,car_step,success_rate,obs_collision_rate,car_collision_rate,car_timeout_rate,mrta_count=self.render(mode='rgb_array')
                infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': True if self.reached_all else False,'timeout': car['timeout'],'goal_reach':self.goal_reach_flag,'episode_info':[max_step,car_step,success_rate,obs_collision_rate,car_collision_rate,car_timeout_rate,mrta_count]} for car in self.cars]
            else:
                infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': True if self.reached_all else False,'timeout': car['timeout'],'goal_reach':self.goal_reach_flag} for car in self.cars]
            return [observations, shared_obs,rewards, dones, infos, None]
        else:
            
            dones=[car['obs_collision'] or car['car_collision'] or car['reached_goal'] or car['timeout'] for car in self.cars]
            if np.all(dones) and self.save_history:
                max_step,car_step,success_rate,obs_collision_rate,car_collision_rate,car_timeout_rate,mrta_count=self.render(mode='rgb_array')
                infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': car['reached_goal'], 'timeout': car['timeout'],'episode_info':[max_step,car_step,success_rate,obs_collision_rate,car_collision_rate,car_timeout_rate,mrta_count]} for car in self.cars]
            else:                                                                                                      
                infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': car['reached_goal'], 'timeout': car['timeout']} for car in self.cars]
            return [observations, shared_obs,rewards, dones, infos,None]

    def render(self, mode='human', gif_name=None):
        colors = ["#e4a68B", "#4a9ab0", "#b0c6d8", "#97bb58","#800080", "#FFA500", "#228B22", "#87CEEB","#2F4F4F", "#FFD700", "#000000", "#00BFFF","#FFD300", "#008000", "#9400D3", "#AFEEEE"]
        self.plot_sensor_line = self.plot_sensor_line if not self.turnoff_sensor else False
        trajectory = self.history
        self.goal_reach_flag_his.append(self.goal_reach_flag.copy())
        goal_reach_flag = self.goal_reach_flag_his
        max_step = max([len(traj) for traj in trajectory])
        obs_collision = [car['obs_collision'] for car in self.cars]
        car_collision = [car['car_collision'] for car in self.cars]
        car_timeout = [car['timeout'] for car in self.cars]
        success_rate = np.sum(np.array(self.goal_reach_flag) != 0) / len(self.goal_reach_flag)
        obs_collision_rate = np.sum(obs_collision) / len(obs_collision)
        car_collision_rate = np.sum(car_collision) / len(car_collision)
        car_timeout_rate = np.sum(car_timeout) / len(car_timeout)
        car_step = [car['STEP'] for car in self.cars if car['reached_goal'] == 1]
        # print([max_step,car_step, success_rate,obs_collision_rate,car_collision_rate,car_timeout_rate])
        
        if self.reborn:
            obs_collision_rate=self.obs_crash_count/self.num_agent
            car_collision_rate=self.car_crash_count/self.num_agent
        if success_rate < 1 or car_collision_rate+obs_collision_rate>0 or car_timeout_rate>0:  
            return [max_step,car_step, success_rate,obs_collision_rate,car_collision_rate,car_timeout_rate,self.mrta_count]
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
        ax.set_aspect('equal')  # Maintain aspect ratio
        self.count_success = 0

        def update(frame):
            ax.clear()
            self.SimEnv.display_environment(ax=ax)
            # ax.set_title("time step {}".format(frame),y=0.98, fontsize=16)
            # ax.set_xlabel("X (m)", fontsize=16)
            # ax.set_ylabel("Y (m)", fontsize=16)
            # Remove axes ticks and labels to save space
            ax.set_xticks([])
            ax.set_yticks([])            
            linewidth=4
            markersize=15
            if self.turnoff_sensor:
                offset = 0
            else:
                offset = self.num_sensors
            # text number of goal
            goals_n = np.array(self.goals)
            # for idx, (goalx, goaly) in enumerate(goals_n):
                # ax.text(goalx, goaly, str(idx), color='black', fontsize=12, ha='center', va='center')
            for idx, traj in enumerate(trajectory):
                if frame < len(traj):
                    state = traj[frame]
                    sensor_data = state[0:offset]
                    x, y, theta, goalx, goaly = state[offset], state[offset + 1], state[offset + 2], state[offset + 3], state[offset + 4]
                    timeout, obs_collision, car_collision, reached_goal = state[offset + 5], state[offset + 6], state[offset + 7], state[offset + 8]
                    
                    ax.plot(x, y, marker='o', color=colors[idx % len(colors)], markersize=markersize)
                    # ax.text(x, y, str(idx), color='red', fontsize=12, ha='center', va='center')
                    # Plot trajectory
                    if self.plot_traj:
                        x_his = [s[offset] for s in traj[:frame + 1]]
                        y_his = [s[offset + 1] for s in traj[:frame + 1]]
                        ax.plot(x_his, y_his, color=colors[idx % len(colors)],linewidth=linewidth)
                        ax.plot(x_his[0], y_his[0], 's', color=colors[idx % len(colors)], markersize=markersize)
                        
                    # Plot sensor data
                    if self.plot_sensor_line:
                        angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_sensors)
                        for distance, angle in zip(sensor_data, angles):
                            sensor_angle = theta + angle
                            sx = x + distance * np.cos(sensor_angle)
                            sy = y + distance * np.sin(sensor_angle)
                            ax.plot([x, sx], [y, sy], 'g-')
                            
                    # Plot car's goal position
                    ax.plot(goalx, goaly, '*', color=colors[idx % len(colors)], markersize=markersize)

                elif self.plot_traj:
                    lastpos = traj[-1]
                    depot = traj[0]
                    ax.plot(depot[offset], depot[offset+1], 's', color=colors[idx % len(colors)], markersize=markersize)
                    ax.plot(lastpos[offset], lastpos[offset + 1], 'o', color=colors[idx % len(colors)], markersize=markersize)
                    ax.plot([s[offset] for s in traj[:]], [s[offset + 1] for s in traj[:]],color=colors[idx % len(colors)],linewidth=linewidth)
            # Plot reached goals
            for idx, agent_idx in enumerate(self.goal_reach_flag_his[frame]):
                if agent_idx ==0: 
                    goal_x, goal_y = self.goals[idx]
                    ax.plot(goal_x, goal_y, '*', color='darkgrey', markersize=markersize)                    
            for idx, agent_idx in enumerate(self.goal_reach_flag_his[frame]):
                if agent_idx != 0:
                    goal_x, goal_y = self.goals[idx]
                    ax.plot(goal_x, goal_y, '*', color=colors[(int(agent_idx) - 1)% len(colors)], markersize=markersize)
            ax.axis('off')  # Completely removes axes

        if mode == 'human':
            # For human mode: create and display animation
            ani = FuncAnimation(fig, update, frames=max_step, repeat=False, interval=int(self.ifi * 1000))
            plt.show()
            
        elif mode == 'rgb_array' and gif_name is not None:
            # Ensure gif_name has proper extension
            if not gif_name.endswith('.gif'):
                gif_name = gif_name + '.gif'
                
            # For rgb_array mode: generate frames and save as gif (no display)
            all_frames = []
            
            for frame in range(max_step):
                print(f"Rendering frame {frame+1}/{max_step}", end='\r')
                update(frame)
                # Force drawing and flush events
                fig.canvas.draw()
                fig.canvas.flush_events()
                # Capture frame
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                all_frames.append(image)
                # fig.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0, dpi=200)
                # temp_image = imageio.imread('temp_frame.png')
                # all_frames.append(temp_image)
                
            if all_frames:  # Only save if we have frames
                imageio.mimsave(gif_name, all_frames, fps=1/self.ifi)
                # Also save the last frame as PNG
                try:
                    from PIL import Image
                    last_frame = all_frames[-1]
                    png_name = gif_name.rsplit('.', 1)[0] + '.png'
                    Image.fromarray(last_frame).save(png_name)
                    print(f"Saved GIF: {gif_name} and PNG: {png_name}")
                except ImportError:
                    print('PIL not installed, cannot save last frame as PNG')
            else:
                print(f"Warning: No frames generated for {gif_name}")

        plt.close(fig)
        return [max_step,car_step, success_rate,obs_collision_rate,car_collision_rate,car_timeout_rate,self.mrta_count]
    
            
    def close(self):
        plt.close()