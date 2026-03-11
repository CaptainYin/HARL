
import sys,os,time,numba,gymnasium,random,imageio,lap
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Point
import multiprocessing as mp
from numba.typed import List
from gym import spaces
from SimEnv import get_edges,SimulationEnvironment_Unmaze,SimulationEnvironment_CA,SimulationEnvironment_CubeCollection,SimulationEnvironment_Warehouse,\
    SimulationEnvironment_Scenario1,SimulationEnvironment_Scenario2,SimulationEnvironment_Scenario3,SimulationEnvironment_Scenario4,\
        SimulationEnvironment_Scenario6,SimulationEnvironment_Scenario7
scenario_list={'CA':SimulationEnvironment_CA,'Cube':SimulationEnvironment_CubeCollection,'Warehouse':SimulationEnvironment_Warehouse,'Unmaze':SimulationEnvironment_Unmaze,\
    'scenario1':SimulationEnvironment_Scenario1,'scenario2':SimulationEnvironment_Scenario2,'scenario3':SimulationEnvironment_Scenario3,'scenario6':SimulationEnvironment_Scenario6}
for i in range(5):scenario_list['scenario4_'+str(i)]=(SimulationEnvironment_Scenario4,i)
for i in range(25):scenario_list['scenario7_'+str(i)]=(SimulationEnvironment_Scenario7,i)
for i in range(46):scenario_list['Warehouse_'+str(i)]=(SimulationEnvironment_Warehouse,i)
for i in range(22):scenario_list['Cube_'+str(i)]=(SimulationEnvironment_CubeCollection,i)
import get_sensor_data as get_sensor_data_C
from get_sensor_data import check_car_collision
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def check_collision(x, y,Env_rectangle_obs,Env_circle_obs,Env_Stringline,safety_distance):
    car_point = Point(x, y)
    for obstacle in Env_rectangle_obs:
        if car_point.distance(obstacle) < safety_distance:
            return True
    for circle in Env_circle_obs:
        if np.sqrt((circle[0]-x)**2+(circle[1]-y)**2) < safety_distance+circle[2]:
            return True
    for line in Env_Stringline:
        if car_point.distance(line) < safety_distance:
            return True
    return False  
def greedyassignment(cost_matrix):
    """
    贪心算法分配任务
    :param cost_matrix: 成本矩阵
    :return: 分配结果
    """
    cost_matrix = cost_matrix.T
    num_tasks = len(cost_matrix)
    num_agents = len(cost_matrix[0])
    
    # 初始化分配结果
    assignment = [-1] * num_agents
    goal_assignment = [-1] * num_tasks
    # 记录已分配的任务
    assigned_tasks = set()
    total_cost = 0
    for agent in range(num_agents):
        min_cost = float('inf')
        best_task = -1
        
        for task in range(num_tasks):
            if task not in assigned_tasks and cost_matrix[task][agent] < min_cost:
                min_cost = cost_matrix[task][agent]
                best_task = task
        
        # 更新分配结果和已分配任务集合
        assignment[agent] = best_task
        goal_assignment[best_task] = agent
        assigned_tasks.add(best_task)
        total_cost += min_cost
    
    return total_cost,assignment,goal_assignment

class EnvArgs:
    """Configuration arguments for the EnvCore class."""
    
    def __init__(self, **kwargs):
        # Default values
        self.num_sensors = 16
        self.Car_resettype = "diag"  # Options: random_target, diag, diag_fixed
        self.num_agents = 4
        self.turnoff_sensor = False
        self.add_sensor_noise = False
        self.fixed_theta = False
        self.scenario = "Unmaze"
        self.share_reward = False
        self.sensor_gussian_noise = 0.1
        self.save_history = False
        self.plot_traj = True
        self.traj_hold = False
        self.plot_sensor_line = True
        self.ifi = 0.1
        self.reward_type = "wonderingpenalty"  # Options: binary, shaping, pR, lp_sn, lp_wosn
        
        self.task='MTSP'
        self.num_samples=100
        self.k=20
        self.metric="euclidean"
        self.assignment= "greedy"
        self.assignmentEachStep = False
        self.obs_goal=True
        self.goal_num=2
        self.reachgoalset = True
        
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
        args = EnvArgs(num_sensors=32, scenario="CA", num_agents=8)
        all_args = EnvArgs.from_dict(all_args)
        #=============================Env=============================
        self.EnvId=0
        self.delta = 0.1#second
        self.episode = 0
        self.task=all_args.task
        self.obs_goal=all_args.obs_goal
        self.metric=all_args.metric
        self.assignment=all_args.assignment
        self.assignmentEachStep=all_args.assignmentEachStep
    
        self.scenario=all_args.scenario
        if isinstance(scenario_list[self.scenario],tuple):
            self.SimEnv=scenario_list[self.scenario][0](idx=scenario_list[self.scenario][1])
        else:
            self.SimEnv=scenario_list[self.scenario]() 

        self.area_size = self.SimEnv.area_size
        self.Env_circle_obs=self.SimEnv.circle_obs
        self.Env_rectangle_obs=self.SimEnv.rectangle_obs
        self.Env_obstacles = self.SimEnv.obstacles
        self.Env_Stringline=self.SimEnv.Stringline
        self.Env_edges_list=self.SimEnv.Env_edges_list
        self.Env_edges_list_len=len(self.Env_edges_list)
        #=============================agent=============================
        self.num_cars = all_args.num_agents
        self.num_agent=all_args.num_agents
        self.n_agents=self.num_agent
        self.safety_distance = 0.2
        self.disk_radius = 0.2
        self.fixed_theta=all_args.fixed_theta
        self.Car_resettype=all_args.Car_resettype #random_target,diag

        self.default_speed = 2
        self.default_omega = np.pi
        
        #=============================sensor=============================
        self.num_sensors = all_args.num_sensors
        self.turnoff_sensor=all_args.turnoff_sensor
        self.add_sensor_noise=all_args.add_sensor_noise
        self.sensor_range = 3.5
        self.sensor_gussian_noise=all_args.sensor_gussian_noise
        
        #=============================obs dim=============================
        if self.task=='MTSP':
            
            self.goal_num=all_args.goal_num
            self.num_samples=all_args.num_samples
            self.k=all_args.k
            self.reachgoalset=all_args.reachgoalset
            self.SimEnv.build_roadmap(num_samples=self.num_samples,k=self.k)

            self.goal_obs_dim=3 #(rel_x,rel_y,reach_flag)
            self.other_obs_dim=5 #(rel_x,rel_y,theta,v,omega)
            self.obs_dim_ego=self.num_sensors if not self.turnoff_sensor else 0
            self.his_len=1
            self.obs_dim = self.obs_dim_ego*self.his_len+5+self.other_obs_dim*(self.num_agent-1)+ self.goal_obs_dim*self.goal_num if not self.obs_goal else self.obs_dim_ego+5
            
            self.state_dim_peragent=5 if self.reachgoalset else 7 #self.num_sensors+ #(x,y,theta,v,omega)
            self.share_observation_dim= self.state_dim_peragent*self.num_agent +self.goal_obs_dim*self.goal_num
            self.max_steps = self.goal_num*2*self.area_size/(self.delta*self.default_speed) # Maximum number of steps per episode, depending on the environment and task
            
        else:
            self.reachgoalset=False
            self.other_obs_dim=7
            self.state_dim_peragent=7
            self.obs_dim_ego=self.num_sensors if not self.turnoff_sensor else 0
            self.his_len=1
            self.obs_dim = self.obs_dim_ego*self.his_len+5
            self.share_observation_dim= self.state_dim_peragent*self.num_agent
            self.max_steps = 2*self.area_size/(self.delta*self.default_speed) # Maximum number of steps per episode, depending on the environment and task
        self.cars = self.generate_cars_Env()
            
        self.obs_his=np.zeros((self.num_agent,self.his_len,self.obs_dim_ego),dtype=np.float32)# his_length=3,0-2:t-2,t-1,t
        self.state=np.zeros(self.share_observation_dim,dtype=np.float32)
        self.action_dim = 2  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional
        self.action_space = [spaces.Box(low=np.array([0, -self.default_omega],dtype=np.float32), high=np.array([self.default_speed, self.default_omega],dtype=np.float32), dtype=np.float32) for _ in range(self.num_agent)]
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]
        self.share_observation_space = [
            # spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim * self.num_agent,), dtype=np.float32) #shared observation is concatenated observation of all agents
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.share_observation_dim,), dtype=np.float32) #shared observation is environment state
            for _ in range(self.num_agent)]
        # self.dim_info={agentid:[self.observation_space.shape[0],self.action_space.shape[0]] for agentid in range(self.num_cars)}

        #=============================reinforcement learning=============================
        self.reward_type=all_args.reward_type#binary,shaping,pR,lp_sn,lp_wosn
        self.share_reward=all_args.share_reward
        #=============================render=============================
        self.save_history=all_args.save_history
        if self.save_history: self.history=[[] for _ in range(self.num_agent)]
        self.plot_traj=all_args.plot_traj
        self.traj_hold=all_args.traj_hold
        self.plot_sensor_line=all_args.plot_sensor_line
        self.ifi=all_args.ifi
        # if self.plot_traj:
        #     self.fig, self.ax = plt.subplots()
        #=============================performance test=============================
        self.time_c=np.zeros(4,dtype=np.float32)
        self.count=0
        self.time_c1=np.zeros(5,dtype=np.float32)
        self.count1=0

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
            np.random.seed(1)
        else:
            random.seed(seed)
            np.random.seed(seed)
            self.EnvId=seed
    def new_car(self,x,y,theta,goal_x,goal_y,i):
        car = {'x': x, 'y': y, 'theta': theta,'assigned':1,'goal_x': goal_x, 'goal_y': goal_y, 'omega': 0, 'v': 0,'previous_x':0,'previous_y':0,'rho':0,'phi':0,'prev_rho':0,'prev_phi':0,
                    'min_laser': 0, 'prev_min_laser': 0,'min_laser_index': 0, 'prev_min_laser_index': 0,'updated':1,
                'obs_collision': False, 'car_collision': False, 'reached_goal': False, 'timeout': False, 
                'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i,'history':[]}
        # if self.reachgoalset:
        car['rho']=[np.sqrt((-car['x']+goal[0])**2 + (-car['y']+goal[1])**2) for goal in self.goals]
        # else:
        #     car['rho'],car['phi'] =np.sqrt((-car['x']+car['goal_x'])**2 + (-car['y']+car['goal_y'])**2),np.arctan2(-car['y']+car['goal_y'],-car['x']+car['goal_x']) - car['theta']
        return car
    def generate_cars_Env(self):
        cars = []
        if self.task=='MTSP':
            position=self.SimEnv.sample_MTSP_from_graph(num_agent=self.num_agent,num_goal=self.goal_num)
            self.src=position[:self.num_agent]
            self.goals=position[self.num_agent:]
            self.goal_reach_flag=np.zeros(self.goal_num,dtype=np.float32)# 0 not reached, k reached by agent k-1
            self.reached_all=False
            if self.reachgoalset:
                for i,pos in enumerate(self.src):
                    x,y=pos
                    goal_x,goal_y=self.goals[0]# the goal not used
                    cars.append(self.new_car(x,y,np.random.uniform(0, 2 * np.pi),goal_x,goal_y,i))
            else:
                #calculate the cost matrix
                cost_matrix=np.zeros((self.num_agent,self.goal_num),dtype=np.float32)
                if self.metric=='dijstra':
                    src,goal=[],[]
                    for i in range(self.num_agent):
                        for j in range(self.goal_num):
                            src.append(position[i])
                            goal.append(position[self.num_agent+j])
                    prm_path = self.SimEnv.find_path(start_set=self.src,goal_set=self.goals,start=src, goal=goal)
                    for i in range(self.num_agent):
                        for j in range(self.goal_num):
                            cost_matrix[i][j]=prm_path[i*self.goal_num+j]
                elif self.metric=='euclidean':
                    for i in range(self.num_agent):
                        for j in range(self.goal_num):
                            cost_matrix[i][j]=np.sqrt((self.src[i][0]-self.goals[j][0])**2+(self.src[i][1]-self.goals[j][1])**2)                  
                # solve the assignment problem
                if self.assignment=='lapjv':
                    total_cost, self.goal_assignment,agent_assignment = lap.lapjv(cost_matrix, extend_cost=True)
                elif self.assignment=='greedy':
                    total_cost, self.goal_assignment,agent_assignment=greedyassignment(cost_matrix)
                
                for i,pos in enumerate(self.src):
                    x,y=pos
                    goal_x,goal_y=self.goals[self.goal_assignment[i]]
                    cars.append(self.new_car(x,y,np.random.uniform(0, 2 * np.pi),goal_x,goal_y,i))
        else:
            self.goals=[]#goal tuple
            self.goal_assignment=[]#indexed by agent
            self.goal_reach_flag=[]#indexed by goal
            # position=self.SimEnv.positions.copy()
            position=self.SimEnv.sample_MTSP(num_agent=self.num_cars,num_goal=self.num_cars)
            src_pos=random.sample(position,self.num_cars)
            des_src=[]
            if self.Car_resettype=='diag': #random_target,diag
                for i,pos in enumerate(src_pos):
                    x,y=pos
                    pos_id=position.index(pos)
                    target_id=pos_id+1 if pos_id%2==0 else pos_id-1
                    goal_x,goal_y = self.SimEnv.positions[target_id][0],self.SimEnv.positions[target_id][1]
                    cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i))
                    self.goals.append((goal_x,goal_y))
                    self.goal_assignment.append(i)
                    self.goal_reach_flag.append(0)
            elif self.Car_resettype=='random_target':
                for i,pos in enumerate(src_pos):
                    x,y=pos
                    tar_position=position.copy()
                    tar_position.remove(pos)
                    goal_x,goal_y = random.sample(tar_position,1)[0]
                    cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i))
                    self.goals.append((goal_x,goal_y))
                    self.goal_assignment.append(i)
                    self.goal_reach_flag.append(0)
            elif self.Car_resettype=='diag_fixed':
                src_pos_ids=self.SimEnv.fixed_eval_task[self.EnvId%len(self.SimEnv.fixed_eval_task)]
                for i,pos_id in enumerate(list(src_pos_ids)):
                    target_id=pos_id+1 if pos_id%2==0 else pos_id-1
                    x,y=self.SimEnv.positions[pos_id][0],self.SimEnv.positions[pos_id][1]
                    goal_x,goal_y = self.SimEnv.positions[target_id][0],self.SimEnv.positions[target_id][1]
                    cars.append(self.new_car(x,y,self.get_init_theta(x,y,goal_x,goal_y),goal_x,goal_y,i))
                    self.goals.append((goal_x,goal_y))
                    self.goal_assignment.append(i)
                    self.goal_reach_flag.append(0)
            else:
                raise NotImplementedError  
            for car in cars:
                car['rho']=[np.sqrt((-car['x']+goal[0])**2 + (-car['y']+goal[1])**2) for goal in self.goals]
        return cars
    def get_init_theta(self,x,y,goal_x,goal_y):
        if self.fixed_theta:
            return np.arctan2(-y+goal_y,-x+goal_x)
        else:
            return np.random.uniform(0, 2 * np.pi)
    def update_state(self, actions):
        self.crashedOrreached=False# record if any car reached the goal or crashed this step
        for car, action in zip(self.cars, actions):
            if car['obs_collision'] or car['car_collision'] or car['timeout'] or car['reached_goal']:
                car['updated']=0
                # car['x'], car['y'] =999,999
                continue
            car['updated']=1
            car['STEP']+=1
            if car['STEP']>=self.max_steps:
                car['timeout']=True
            v, w = action
            car['previous_omega'],car['previous_v']=car['omega'],car['v']
            car['v'],car['omega'],car['previous_x'],car['previous_y'],theta0 = v,w,car['x'],car['y'],car['theta'] 
            new_x,new_y = car['x'],car['y']
            for _ in range(4):
                car['theta'] += w * self.delta/4
                new_x = new_x + v * np.cos(car['theta']) * self.delta/4
                new_y = new_y + v * np.sin(car['theta']) * self.delta/4

            car['x'],car['y'] = np.clip(new_x, -self.area_size/2, self.area_size/2),np.clip(new_y, -self.area_size/2, self.area_size/2)

            car['prev_rho']=car['rho'].copy()
            car['rho']=[np.sqrt((-car['x']+goal[0])**2 + (-car['y']+goal[1])**2) for goal in self.goals]

            if check_collision(car['x'], car['y'],self.Env_rectangle_obs,self.Env_circle_obs,self.Env_Stringline,self.safety_distance):
                car['obs_collision'] = True
                
                self.crashedOrreached=True
            elif check_car_collision([car['x'], car['y']], [[c['x'],c['y']] for c in self.cars if c != car],self.safety_distance):
                car['car_collision'] = True
                self.crashedOrreached=True
            else:
                if not self.reachgoalset:
                    if car['assigned'] and car['rho'][self.goal_assignment[car['car_id']%self.num_agent]] < self.safety_distance:
                        car['reached_goal'] = True
                        self.crashedOrreached=True
                        self.goal_reach_flag[self.goal_assignment[car['car_id']%self.num_agent]]=car['car_id']%self.num_agent+1        
                else:
                    for i in range(self.goal_num):
                        if self.goal_reach_flag[i]!=0:
                            car['rho'][i]=np.inf
                    if np.min(car['rho']) < self.safety_distance:
                        car['reached_goal'] = True 
                        self.crashedOrreached=True
                        self.goal_reach_flag[np.argmin(car['rho'])]=car['car_id']%self.num_agent+1
                    elif np.min(car['rho'])==np.inf:
                        self.reached_all=True


    def reasign_goal(self):
        #if any goal is reached, or any car crashed in this step, change goal assignment
        if (self.crashedOrreached or self.assignmentEachStep) and self.task=='MTSP' and not self.reachgoalset:
            remain_goal_index=[i for i in range(self.goal_num) if self.goal_reach_flag[i]==0]
            remain_agent_index=[i for i in range(self.num_agent) if self.cars[i]['car_collision']==0 and self.cars[i]['obs_collision']==0 and self.cars[i]['timeout']==0]
            if len(remain_goal_index)>0 and len(remain_agent_index)>0:

                cost_matrix=np.zeros((len(remain_agent_index),len(remain_goal_index)),dtype=np.float32)
                if self.metric=='dijstra':
                    start_set=[(self.cars[remain_agent_index[i]]['x'],self.cars[remain_agent_index[i]]['y']) for i in range(len(remain_agent_index))]
                    goal_set=[self.goals[remain_goal_index[i]] for i in range(len(remain_goal_index))]
                    n_src,n_goal=[],[]
                    for i in range(len(remain_agent_index)):
                        for j in range(len(remain_goal_index)):
                            n_src.append(start_set[i])
                            n_goal.append(goal_set[j])
                    prm_path = self.SimEnv.find_path(start_set=start_set,goal_set=goal_set,start=n_src, goal=n_goal)
                    for i in range(len(remain_agent_index)):
                        for j in range(len(remain_goal_index)):                
                            cost_matrix[i][j]=prm_path[i*len(remain_goal_index)+j]
                elif self.metric=='euclidean':
                    for i in range(len(remain_agent_index)):
                        for j in range(len(remain_goal_index)):
                            cost_matrix[i][j]=np.sqrt((self.cars[remain_agent_index[i]]['x']-self.goals[remain_goal_index[j]][0])**2+(self.cars[remain_agent_index[i]]['y']-self.goals[remain_goal_index[j]][1])**2)
                # ## solve the assignment problem
                if self.assignment=='lapjv':
                    total_cost, goal_assignment,agent_assignment = lap.lapjv(cost_matrix, extend_cost=True)
                elif self.assignment=='greedy':
                    total_cost, goal_assignment,agent_assignment = greedyassignment(cost_matrix)
                
                for i in range(len(remain_agent_index)):
                    if goal_assignment[i]==-1:#means no goal assigned
                        # self.cars[remain_agent_index[i]]['goal_x'],self.cars[remain_agent_index[i]]['goal_y'],self.cars[remain_agent_index[i]]['updated']=99999,99999,0
                        # assert i to the nearst goal
                        # goal_assignment[i]=remain_goal_index[np.argmin(cost_matrix[i])]
                        #not assigned, means no goal assigned
                        self.cars[remain_agent_index[i]]['assigned']=0
                        # self.cars[remain_agent_index[i]]['x'],self.cars[remain_agent_index[i]]['y']=99999,99999
                        self.cars[remain_agent_index[i]]['goal_x'],self.cars[remain_agent_index[i]]['goal_y']=self.cars[remain_agent_index[i]]['x'],self.cars[remain_agent_index[i]]['y']
                    else:
                        goal_assignment[i]=remain_goal_index[goal_assignment[i]]
                        self.cars[remain_agent_index[i]]['assigned']=1
                        self.cars[remain_agent_index[i]]['goal_x'],self.cars[remain_agent_index[i]]['goal_y']=self.goals[goal_assignment[i]]
                # update the self.goal_assignment
                for i in range(self.num_agent):
                    if self.cars[i]['car_collision'] or self.cars[i]['obs_collision'] or self.cars[i]['timeout']:
                        self.goal_assignment[i]=-1
                    else:
                        assert i in remain_agent_index
                        self.goal_assignment[i]=goal_assignment[remain_agent_index.index(i)]
            else:
                if len(remain_goal_index)==0:
                    self.reached_all=True
                            
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
            if self.task=='MTSP' and not self.obs_goal:
                obs_other=np.zeros((self.num_agent-1)*self.other_obs_dim,dtype=np.float32)
                i=0
                for other_car in self.cars:
                    if other_car['car_id']!=car['car_id']:
                        obs_other[i*self.other_obs_dim:(i+1)*self.other_obs_dim]=[other_car['x'],other_car['y'],other_car['theta'],other_car['v'],other_car['omega']]
                        # r_rho=np.sqrt((-other_car['x']+car['x'])**2 + (-other_car['y']+car['y'])**2)
                        # r_phi=np.arctan2(other_car['y']-car['y'],other_car['x']-car['x']) - car['theta']
                        # obs_other[i*self.other_obs_dim:(i+1)*self.other_obs_dim]=[r_rho,r_phi,other_car['theta'],other_car['v'],other_car['omega']]
                        i+=1
                obs_goal=np.zeros(self.goal_num*self.goal_obs_dim,dtype=np.float32)
                for i in range(self.goal_num):
                    
                    # r_rho=np.sqrt((-goal_x+car['x'])**2 + (-goal_y+car['y'])**2)
                    # r_phi=np.arctan2(goal_y-car['y'],goal_x-car['x']) - car['theta']
                    # obs_goal[i*self.goal_obs_dim:(i+1)*self.goal_obs_dim]=[r_rho,r_phi,1 if self.goal_reach_flag[i] else 0]
                    # obs_goal[i*self.goal_obs_dim:(i+1)*self.goal_obs_dim]=[r_rho,r_phi,self.goal_reach_flag[i]]
                    obs_goal[i*self.goal_obs_dim:(i+1)*self.goal_obs_dim]=[self.goals[i][0],self.goals[i][1],self.goal_reach_flag[i]]
                observation = np.concatenate((observation,[car['theta'],car['omega'],car['v'],car['x'],car['y']],obs_other,obs_goal))
                # observation = np.concatenate((observation,[car['theta'],car['omega'],car['v']],obs_other,obs_goal))
            else:
                observation = np.concatenate((observation,[car['theta'],car['omega'],car['v'],car['rho'][self.goal_assignment[car['car_id']%self.num_agent]],car['phi']]))
            
            if self.save_history:
                # car['history'].append(np.concatenate((sensor_data,np.array([car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y'],\
                #     car['timeout'],car['obs_collision'], car['car_collision'],car['reached_goal'],car['omega'],car['v']]))))
                self.history[car['car_id']%self.num_agent].append(np.concatenate((sensor_data,np.array([car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y'],\
                    car['timeout'],car['obs_collision'], car['car_collision'],car['reached_goal'],car['omega'],car['v']]))))
        else:
            if self.task=='MTSP':
                obs_other=np.zeros(((self.num_agent-1)*self.other_obs_dim),dtype=np.float32)
                i=0
                for other_car in self.cars:
                    if other_car['car_id']!=car['car_id']:
                        r_rho=np.sqrt((-other_car['x']+car['x'])**2 + (-other_car['y']+car['y'])**2)
                        r_phi=np.arctan2(other_car['y']-car['y'],other_car['x']-car['x']) - car['theta']
                        obs_other[i*self.other_obs_dim:(i+1)*self.other_obs_dim]=[r_rho,r_phi,other_car['theta'],other_car['v'],other_car['omega']]
                        i+=1
                obs_goal=np.zeros(self.goal_num*self.goal_obs_dim,dtype=np.float32)
                for i in range(self.goal_num):
                    goal_x,goal_y=self.goals[i]
                    r_rho=np.sqrt((-goal_x+car['x'])**2 + (-goal_y+car['y'])**2)
                    r_phi=np.arctan2(goal_y-car['y'],goal_x-car['x']) - car['theta']
                    # obs_goal[i*self.goal_obs_dim:(i+1)*self.goal_obs_dim]=[r_rho,r_phi,1 if self.goal_reach_flag[i] else 0]
                    obs_goal[i*self.goal_obs_dim:(i+1)*self.goal_obs_dim]=[r_rho,r_phi,self.goal_reach_flag[i]]
                observation = np.concatenate(([car['theta'],car['omega'],car['v']],obs_other,obs_goal))
            else:
                observation = np.concatenate(([car['theta'],car['omega'],car['v'],car['rho'],car['phi']]))
            if self.save_history:
                # car['history'].append((np.array([car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y'],\
                #     car['timeout'],car['obs_collision'], car['car_collision'],car['reached_goal'],car['omega'],car['v']])))
                self.history[car['car_id']%self.num_agent].append(np.array([car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y'],\
                    car['timeout'],car['obs_collision'], car['car_collision'],car['reached_goal'],car['omega'],car['v']]))
        return observation
    def get_state(self):
        if self.task=='MTSP':
        # self.state_dim_peragent=self.num_sensors+5 #(x,y,theta,v,omega)
        #self.share_observation_dim= self.state_dim_peragent*self.num_agent +self.goal_obs_dim*self.goal_num
            for i,car in enumerate(self.cars):
                if car['updated']==0:
                    self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[0]*self.state_dim_peragent
                else:
                    # observation=self.obs_his[car['car_id']%self.num_agent,:,:].reshape(-1)
                    # self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=np.concatenate((observation,[car['theta'],car['rho'],car['phi'],car['omega'],car['v'],car['x'],car['y']]))
                    if not self.reachgoalset:
                        if car['assigned']:
                            rho,phi=car['rho'][self.goal_assignment[i]],car['phi']
                        else:
                            rho,phi=0,0
                        self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=np.array(([car['theta'],rho,phi,car['omega'],car['v'],car['x'],car['y']]))
                    else:
                        self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=np.array(([car['theta'],car['omega'],car['v'],car['x'],car['y']]))
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
                    self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[car['theta'],car['rho'][self.goal_assignment[i]],car['phi'],car['omega'],car['v'],car['min_laser'],car['min_laser_index']]
        # return np.expand_dims(self.state,axis=0).repeat(self.num_agent,axis=0)
        return self.state

    def reset(self):
        self.cars = self.generate_cars_Env()
        if self.save_history: self.history=[[] for _ in range(self.num_agent)]
        self.episode += 1
        observations = [self.get_observation(car) for car in self.cars]
        return observations,self.get_state(),None
        # return observations

    def get_reward(self):
        rewards=np.zeros([self.num_cars,1],dtype=np.float32)
        for i,car in enumerate(self.cars):
            if car['updated']==0:#  if car['obs_collision'] or car['car_collision'] or car['timeout'] in last step and current step have not updated, then no reward
                continue
            if car['assigned']==0:#if not assigned, should stay there
                # rewards[i][0]=-(car['v']*car['omega'])**2
                # print('errot')
                continue
            if self.task=='MTSP':
                if self.reward_type=='reachall':
                    rewards[i][0]=(car['prev_rho']-car['rho'])*3 
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]-=100
                    elif car['reached_goal']:
                        rewards[i][0]+=10
                        car['reached_goal']=False
                    if self.reached_all:
                        rewards[i][0]+=100
                elif self.reward_type=='shaping':
                    if not self.reachgoalset:
                        rewards[i][0]=(car['prev_rho'][self.goal_assignment[i]]-car['rho'][self.goal_assignment[i]])*3 
                    else:
                        prev_rho=np.min(car['prev_rho'])
                        rho=np.min(car['rho'])
                        if prev_rho==np.inf or rho==np.inf:
                            rewards[i][0]=0
                        else:
                            rewards[i][0]=(prev_rho-rho)*3
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
                    rewards[i][0]=(car['prev_rho']-car['rho'])*2-1.5 
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]-=50
                    elif car['reached_goal']:
                        rewards[i][0]+=120
                        car['reached_goal']=False
                    # if self.reached_all:
                    #     rewards[i][0]+=100                          
            else:
                if self.reward_type=='binary':
                    if car['reached_goal']:
                        rewards[i][0]=0
                    else:
                        rewards[i][0]=-1
                elif self.reward_type=='shaping':
                    rewards[i][0]=(car['prev_rho'][self.goal_assignment[i]]-car['rho'][self.goal_assignment[i]])*3 
                    if car['obs_collision'] or car['car_collision']:
                        rewards[i][0]+=-10
                    if car['reached_goal']:
                        rewards[i][0]+=100
                else:
                    raise NotImplementedError
        
        # return rewards
        if self.share_reward:
            return [[np.mean([r.sum() for r in rewards])]]*self.num_agent
        else:
            return [[r.sum()] for r in rewards]
    
    def step(self, actions):

        actions[:,0]=sigmoid(actions[:,0])*self.default_speed
        np.tanh(actions[:,1],out=actions[:,1])
        actions[:,1]=actions[:,1]*self.default_omega

        # self.count+=1
        # start=time.time()
        self.update_state(actions)
        # self.time_c[0]+=time.time()-start
        # start=time.time()
        observations = [self.get_observation(car) for car in self.cars]
        # self.time_c[1]+=time.time()-start
        # start=time.time()
        # observations=[obs.astype(np.float16) for obs in observations]
        rewards = self.get_reward()
        # self.time_c[2]+=time.time()-start
        # start=time.time()

        shared_obs=self.get_state()
        self.reasign_goal()
        # self.time_c[3]+=time.time()-start
        # if self.count%1000==0:
        #     print(f"step time:{self.time_c/np.sum(self.time_c)}")
        if self.task=='MTSP':
            # if self.save_history:
            #     infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': True if self.reached_all else False,'timeout': car['timeout'], 'history': car['history']} for car in self.cars]
            # else:
            infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': True if self.reached_all else False,'timeout': car['timeout']} for car in self.cars]

            # print([observations, shared_obs,rewards, [car['obs_collision'] or car['car_collision'] or car['timeout'] or self.reached_all for car in self.cars], infos, None])
            return [observations, shared_obs,rewards, [car['obs_collision'] or car['car_collision'] or car['timeout'] or self.reached_all for car in self.cars], infos, None]
        else:
            # if self.save_history:
            #     infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': car['reached_goal'], 'timeout': car['timeout'], 'history': car['history']} for car in self.cars]
            # else:
            infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': car['reached_goal'], 'timeout': car['timeout']} for car in self.cars]
            return [observations, shared_obs,rewards, [car['obs_collision'] or car['car_collision'] or car['reached_goal'] or car['timeout'] for car in self.cars], infos,None]
        # return [observations, rewards, [car['obs_collision'] or car['car_collision'] or car['reached_goal'] or car['timeout'] for car in self.cars], infos]
    
    def render(self, mode='human',gif_name=None):
        colors = [
            "#e4a68B", "#4a9ab0", "#b0c6d8", "#97bb58",
            "#800080", "#FFA500", "#228B22", "#87CEEB",
            "#2F4F4F", "#FFD700", "#000000", "#00BFFF",
            "#FFD300", "#008000", "#9400D3", "#AFEEEE"
        ]
        # trajectory should be a list of num_agent of [sensor,car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y']] tuples
        fig, ax = plt.subplots()
        self.plot_sensor_line = self.plot_sensor_line if not self.turnoff_sensor else False
        step = 0
        trajectory=self.history
        max_step = max([len(traj) for traj in trajectory])
        print(f"max step: {max_step}")
        all_frames = []
        def update(frame):
            ax.clear()
            self.SimEnv.display_environment(ax=ax)
            ax.set_title("Car Trajectories after time step {}".format(frame), fontsize=16)
            ax.set_xlabel("X (m)", fontsize=16)
            ax.set_ylabel("Y (m)", fontsize=16)
            # plot all unreach goal and reach goal
            for i in range(self.goal_num):
                goalx,goaly=self.goals[i]
                ax.plot(goalx, goaly, 'o', color='#3b609b', markersize=10)
                if self.goal_reach_flag[i]!=0:
                    ax.plot(goalx, goaly, 'o', color='#2b8b8c', markersize=5)
            if self.turnoff_sensor:
                offset=0
            else:
                offset=self.num_sensors          
            for idx,traj in enumerate(trajectory):
                if frame < len(traj):
                    state = traj[frame]

                    sensor_data = state[0:offset]
                    x,y,theta,goalx,goaly = state[offset],state[offset + 1],state[offset + 2],state[offset + 3],state[offset + 4]
                    timeout,obs_collision, car_collision,reached_goal=state[offset + 5],state[offset + 6],state[offset + 7],state[offset + 8]
                    # if timeout:print("{} timeout".format(idx))
                    # if obs_collision:print("{} obs_collision".format(idx))
                    # if car_collision:print("{} car_collision".format(idx))
                    # if reached_goal:print("{} reached_goal".format(idx))
                    # print([idx,state[-1],state[-2],goalx,goaly])
                    ax.plot(x, y, marker='s',color=colors[idx%len(colors)],markersize=10)
                    # Plot trajectory
                    if self.plot_traj:
                        x_his = [s[offset] for s in traj[:frame + 1]]
                        y_his = [s[offset + 1] for s in traj[:frame + 1]]
                        ax.plot(x_his, y_his)
                    
                    # Plot sensor data
                    if self.plot_sensor_line:
                        angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_sensors)
                        for distance, angle in zip(sensor_data, angles):
                            sensor_angle = theta + angle
                            sx = x + distance * np.cos(sensor_angle)
                            sy = y + distance * np.sin(sensor_angle)
                            ax.plot([x, sx], [y, sy], 'g-')
                    # Plot car's goal position
                    ax.plot(goalx, goaly, 'o', color=colors[idx%len(colors)],markersize=10)
                elif self.plot_traj:
                    state=traj[-1]
                    goalx = state[offset + 3]
                    goaly = state[offset + 4]
                    ax.plot(goalx, goaly, 'o',color=colors[idx%len(colors)], markersize=10)
                    x_his = [s[offset] for s in traj[:]]
                    y_his = [s[offset + 1] for s in traj[:]]
                    ax.plot(x_his, y_his,color=colors[idx%len(colors)])
            if mode == 'rgb_array':
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                all_frames.append(image)
        ani = FuncAnimation(fig, update, frames=max_step, repeat=False)
        
        # if mode == 'human':
        plt.show()
        if mode == 'rgb_array' and gif_name!=None:
            # Save the frames as a video
            imageio.mimsave(gif_name, all_frames, fps=1/self.ifi)
        
        plt.close(fig)
            
    def close(self):
        plt.close()