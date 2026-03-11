
import sys,os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Point
import time
import multiprocessing as mp
import numba
from numba.typed import List
import gymnasium
from gym import spaces
import random,imageio
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
@numba.jit(nopython=True) 
def ray_intersects_segment(x, y, angle, x1, y1, x2, y2):
    dx = np.cos(angle)
    dy = np.sin(angle)
    segment = ((x1, y1), (x2, y2))
    p1 = np.array([x, y])
    p2 = np.array([x + dx, y + dy])
    p3 = np.array([x1, y1])
    p4 = np.array([x2, y2])
    denominator = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    if denominator == 0:
        return False,0,0  # Lines are parallel

    ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denominator
    ub = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / denominator

    if 0 <= ua and 0 <= ub <= 1:
        intersection = p1 + ua * (p2 - p1)
        return True, intersection[0], intersection[1]
    return False,0,0
@numba.jit(nopython=True) 
def get_sensor_data_single(edges_list,sensor_range,car_theta,car_x, car_y,angle,circle_obs):
    #circle_obs is list of circle obs, each element is [x,y,radius]
    sensor_angle = car_theta + angle
    min_distance = sensor_range
    contact_edge=-1
    edlen=len(edges_list)
    for edge_ind in range(edlen):
        x1, y1, x2, y2=edges_list[edge_ind]
        intersection, ix, iy= ray_intersects_segment(car_x, car_y, sensor_angle, x1, y1, x2, y2)
        if intersection:
            distance = np.sqrt((ix - car_x) ** 2 + (iy - car_y) ** 2)
            if distance<min_distance:
                min_distance=distance
                contact_edge=edge_ind
    for circle in circle_obs:
        # laser ray intersects with circle
        b = np.sqrt((car_x - circle[0]) ** 2 + (car_y - circle[1]) ** 2)
        angle_to_circle = np.arctan2(circle[1] - car_y, circle[0] - car_x)
        angle_diff = (angle_to_circle - sensor_angle + np.pi) % (2 * np.pi) - np.pi
        if abs(angle_diff) < np.arcsin(circle[2] / b):
            distance = np.sqrt(circle[2]**2 + b**2 - 2*circle[2]*b*np.cos(abs(angle_diff)))
            if distance<min_distance:
                min_distance=distance
            
    return min_distance,contact_edge

@numba.jit(nopython=True) 
def get_sensor_data(egoid,edges_list,edlen,Env_edges_list_len,num_sensors,sensor_range,car_theta,car_x, car_y):
    sensor_data = np.full(num_sensors, sensor_range)
    contact_edge_type = np.full(num_sensors, -2,dtype=np.int8)#-2 for none contact, 0-car_num-1 for car, -1 for other static obstacle
    angles = np.linspace(-np.pi / 2, np.pi / 2, num_sensors)
    for i in range(num_sensors):
        sensor_data[i],contact_edge = get_sensor_data_single(edges_list,edlen,sensor_range,car_theta,car_x,car_y,angles[i])
        if contact_edge>=0:# have contact something
            if contact_edge<Env_edges_list_len:
                contact_edge_type[i]=-1
            else:
                contact_edge_type[i]=(contact_edge-Env_edges_list_len)//16 # ind of other agent, the index have omit ego agent, for example: ego index is 2, the other car index will be 0, 1, 3
                contact_edge_type[i]= contact_edge_type[i] if contact_edge_type[i]<egoid else contact_edge_type[i]+1
    return sensor_data,contact_edge_type
@numba.jit(nopython=True) 
def get_sensor_data1(egoid,edges_list,num_sensors,sensor_range,car_theta,car_x, car_y,env_circle_obs):
    # the car_list is the list of all cars except ego car
    circle_obs=env_circle_obs
    circle_obs1=[]
    for circle in circle_obs:
        if circle[0]==car_x and circle[1]==car_y:# the same car
            continue
        if np.sqrt((circle[0]-car_x)**2+(circle[1]-car_y)**2)>circle[2]+sensor_range:# can not reach
            continue
        else:
            circle_obs1.append([circle[0],circle[1],circle[2]])
    sensor_data = np.full(num_sensors, sensor_range)
    angles = np.linspace(-np.pi / 2, np.pi / 2, num_sensors)
    for i in range(num_sensors):
        sensor_data[i],contact_edge = get_sensor_data_single(edges_list,sensor_range,car_theta,car_x,car_y,angles[i],circle_obs1)
    return sensor_data

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
        self.num_sensors = all_args.num_sensors
        self.Car_resettype=all_args.Car_resettype #random_target,diag
        self.num_cars = all_args.num_agents
        self.num_agent=all_args.num_agents
        self.n_agents=self.num_agent
        self.turnoff_sensor=all_args.turnoff_sensor
        self.add_sensor_noise=all_args.add_sensor_noise
        self.fixed_theta=all_args.fixed_theta
        self.other_obs_dim=7
        self.state_dim_peragent=7
        self.obs_dim_ego=self.num_sensors
        # self.obs_dim = self.obs_dim_ego+self.other_obs_dim*(self.num_agent-1)  # 设置智能体的观测维度 # set the observation dimension of agents
        self.his_len=1
        self.obs_dim = self.obs_dim_ego*self.his_len+7 if not self.turnoff_sensor else 7
        self.share_observation_dim= self.state_dim_peragent*self.num_agent
        self.obs_his=np.zeros((self.num_agent,self.his_len,self.obs_dim_ego),dtype=np.float32)# his_length=3,0-2:t-2,t-1,t
        self.state=np.zeros(self.share_observation_dim,dtype=np.float32)
        self.action_dim = 2  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional

        self.EnvId=0
        self.delta = 0.1#second
        
        self.scenario=all_args.scenario
        if isinstance(scenario_list[self.scenario],tuple):
            self.SimEnv=scenario_list[self.scenario][0](idx=scenario_list[self.scenario][1])
        else:
            self.SimEnv=scenario_list[self.scenario]() 
        self.area_size = self.SimEnv.area_size
        self.sensor_range = 3.5
        self.safety_distance = 0.2
        self.disk_radius = 0.2
        
        self.Env_circle_obs=self.SimEnv.circle_obs
        self.Env_rectangle_obs=self.SimEnv.rectangle_obs
        self.Env_obstacles = self.SimEnv.obstacles
        self.Env_Stringline=self.SimEnv.Stringline
        self.Env_edges_list=self.SimEnv.Env_edges_list
        self.Env_edges_list_len=len(self.Env_edges_list)
        # self.outofbound_edgelist=get_edges(list(Point(-10000,-10000).buffer(self.disk_radius,4).exterior.coords))
        # self.default_obstacle_edgelist=List([edge for edge in self.Env_edges_list]+[edge for edge in self.outofbound_edgelist]*self.num_cars)
        # self.default_obstacle_edgelist=[edge for edge in self.Env_edges_list]+[edge for edge in self.outofbound_edgelist]*(self.num_cars-1)
        self.cars = self.generate_cars_Env()
        # self.generate_dynamic_obstacles_edgelist()
        
        self.default_speed = 2
        self.default_omega = np.pi
        self.max_steps = 2*self.area_size/(self.delta*self.default_speed) # Maximum number of steps per episode, depending on the environment and task
        
        self.episode = 0
        self.action_space = [spaces.Box(low=np.array([0, -self.default_omega],dtype=np.float32), high=np.array([self.default_speed, self.default_omega],dtype=np.float32), dtype=np.float32) for _ in range(self.num_agent)]
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]
        self.share_observation_space = [
            # spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim * self.num_agent,), dtype=np.float32) #shared observation is concatenated observation of all agents
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.share_observation_dim,), dtype=np.float32) #shared observation is environment state
            for _ in range(self.num_agent)]
        # self.dim_info={agentid:[self.observation_space.shape[0],self.action_space.shape[0]] for agentid in range(self.num_cars)}

        self.sensor_lines = []
        self.reward_type="shaping"#binary,shaping,pR,lp_sn,lp_wosn
        self.share_reward=all_args.share_reward
        self.sensor_gussian_noise=all_args.sensor_gussian_noise
        self.save_history=all_args.save_history
        
        self.plot_traj=all_args.plot_traj
        self.traj_hold=all_args.traj_hold
        self.plot_sensor_line=all_args.plot_sensor_line
        self.ifi=all_args.ifi
        self.time_c=np.zeros(4,dtype=np.float32)
        self.count=0
        # if self.plot_traj:
        #     self.fig, self.ax = plt.subplots()
        
        # print(np.random.random())
    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
            np.random.seed(1)
        else:
            random.seed(seed)
            np.random.seed(seed)
            self.EnvId=seed
    def generate_dynamic_obstacles_edgelist(self):

        for car in self.cars:
            if car['updated']==0:
                car_edge_list=self.outofbound_edgelist[:]
            else:
                car_edge_list=get_edges(list(Point(car['x'],car['y']).buffer(self.disk_radius,4).exterior.coords))
            for othercar in self.cars:
                if car['car_id']!=othercar['car_id']:
                    inter_ind=self.Env_edges_list_len+16*(car['car_id']%self.num_cars) if car['car_id']<othercar['car_id'] else self.Env_edges_list_len+16*(car['car_id']%self.num_cars)-16
                    othercar['dynamic_obstacles_edges_list'][inter_ind:inter_ind+16]=car_edge_list[:]
                else:
                    continue
        return True

    def generate_obstacles(self, num_obstacles=1):
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                width = np.random.uniform(0.5, 2.0)
                height = np.random.uniform(0.5, 2.0)
                x = np.random.uniform(-self.area_size/2, self.area_size/2 - width)
                y = np.random.uniform(-self.area_size/2, self.area_size/2 - height)
                new_obstacle = Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
                if not any(new_obstacle.intersects(ob) for ob in obstacles):
                    obstacles.append(new_obstacle)
                    break
        return obstacles

    def generate_cars(self):
        cars = []
        for i in range(self.num_cars):
            while True:
                x = np.random.uniform(-self.area_size/2, self.area_size/2)
                y = np.random.uniform(-self.area_size/2, self.area_size/2)
                theta = np.random.uniform(0, 2 * np.pi)
                goal_x = np.random.uniform(-self.area_size/2, self.area_size/2)
                goal_y = np.random.uniform(-self.area_size/2, self.area_size/2)
                new_car = {'x': x, 'y': y, 'theta': theta, 'goal_x': goal_x, 'goal_y': goal_y, 'omega': 0, 'v': 0,'previous_x':0,'previous_y':0,'rho':0,'phi':0,'prev_rho':0,'prev_phi':0,
                            'min_laser': 0, 'prev_min_laser': 0,'min_laser_index': 0, 'prev_min_laser_index': 0,'updated':1,
                        'obs_collision': False, 'car_collision': False, 'reached_goal': False, 'timeout': False,  'other_car':np.zeros(self.other_obs_dim*(self.num_agent-1)),#should be like [flag1,rel_phi1]*other car num
                        'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i }
                        # 'history': [(x, y, theta)],
                #phi, rho is the relative polar coordinate of goal， theta is the orientation of the car
                car_point = Point(x, y)
                if not any(car_point.within(ob) for ob in self.Env_obstacles) and not check_car_collision([new_car['x'],new_car['y']], [[car['x'],car['y']] for car in cars],self.safety_distance):
                    cars.append(new_car)
                    break
        return cars
    def generate_cars_Env(self):
        cars = []
        position=self.SimEnv.positions.copy()
        src_pos=random.sample(position,self.num_cars)
        des_src=[]
        if self.Car_resettype=='diag': #random_target,diag
            for i,pos in enumerate(src_pos):
                x,y=pos
                pos_id=position.index(pos)
                target_id=pos_id+1 if pos_id%2==0 else pos_id-1
                
                goal_x,goal_y = self.SimEnv.positions[target_id][0],self.SimEnv.positions[target_id][1]
                theta = self.get_init_theta(x,y,goal_x,goal_y)
                new_car = {'x': x, 'y': y, 'theta': theta, 'goal_x': goal_x, 'goal_y': goal_y, 'omega': 0, 'v': 0,'previous_x':0,'previous_y':0,'rho':0,'phi':0,'prev_rho':0,'prev_phi':0,
                            'min_laser': 0, 'prev_min_laser': 0,'min_laser_index': 0, 'prev_min_laser_index': 0,'updated':1,
                        'obs_collision': False, 'car_collision': False, 'reached_goal': False, 'timeout': False, 'other_car':np.zeros(self.other_obs_dim*(self.num_agent-1)),#should be like [flag1,rel_phi1]*other car num
                        'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i,'history': []}
                        # ,
                #phi, rho is the relative polar coordinate of goal， theta is the orientation of the car
                cars.append(new_car)
        elif self.Car_resettype=='random_target':
            for i,pos in enumerate(src_pos):
                x,y=pos
                tar_position=position.copy()
                tar_position.remove(pos)
                goal_x,goal_y = random.sample(tar_position,1)[0]
                theta = self.get_init_theta(x,y,goal_x,goal_y)
                new_car = {'x': x, 'y': y, 'theta': theta, 'goal_x': goal_x, 'goal_y': goal_y, 'omega': 0, 'v': 0,'previous_x':0,'previous_y':0,'rho':0,'phi':0,'prev_rho':0,'prev_phi':0,
                            'min_laser': 0, 'prev_min_laser': 0,'min_laser_index': 0, 'prev_min_laser_index': 0,'updated':1,
                        'obs_collision': False, 'car_collision': False, 'reached_goal': False, 'timeout': False,  'other_car':np.zeros(self.other_obs_dim*(self.num_agent-1)),#should be like [flag1,rel_phi1]*other car num
                        'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i,'history': []}
                        # 'history': [(x, y, theta)],
                #phi, rho is the relative polar coordinate of goal， theta is the orientation of the car
                cars.append(new_car)
        elif self.Car_resettype=='diag_fixed':
            src_pos_ids=self.SimEnv.fixed_eval_task[self.EnvId%len(self.SimEnv.fixed_eval_task)]
            for i,pos_id in enumerate(list(src_pos_ids)):
                target_id=pos_id+1 if pos_id%2==0 else pos_id-1
                x,y=self.SimEnv.positions[pos_id][0],self.SimEnv.positions[pos_id][1]
                goal_x,goal_y = self.SimEnv.positions[target_id][0],self.SimEnv.positions[target_id][1]
                theta = self.get_init_theta(x,y,goal_x,goal_y)
                new_car = {'x': x, 'y': y, 'theta': theta, 'goal_x': goal_x, 'goal_y': goal_y, 'omega': 0, 'v': 0,'previous_x':0,'previous_y':0,'rho':0,'phi':0,'prev_rho':0,'prev_phi':0,
                            'min_laser': 0, 'prev_min_laser': 0,'min_laser_index': 0, 'prev_min_laser_index': 0,'updated':1,
                        'obs_collision': False, 'car_collision': False, 'reached_goal': False, 'timeout': False, 'other_car':np.zeros(self.other_obs_dim*(self.num_agent-1)),#should be like [flag1,rel_phi1]*other car num
                        'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i,'history': []}
                        # ,
                #phi, rho is the relative polar coordinate of goal， theta is the orientation of the car
                cars.append(new_car)
        else:
            raise NotImplementedError  
        return cars
    def get_init_theta(self,x,y,goal_x,goal_y):
        if self.fixed_theta:
            return np.arctan2(-y+goal_y,-x+goal_x)
        else:
            return np.random.uniform(0, 2 * np.pi)
    def update_state(self, actions):
        for car, action in zip(self.cars, actions):
            if car['obs_collision'] or car['car_collision'] or car['reached_goal'] or car['timeout']:
                car['updated']=0
                continue
            car['updated']=1
            car['STEP']+=1
            if car['STEP']>=self.max_steps:
                car['timeout']=True
            v, w = action
            car['previous_omega'],car['previous_v']=car['omega'],car['v']
            car['v'],car['omega'] = v,w
            car['previous_x'],car['previous_y']=car['x'],car['y']
            theta0=car['theta']
            
            new_x,new_y = car['x'],car['y']
            for _ in range(4):
                car['theta'] += w * self.delta/4
                new_x = new_x + v * np.cos(car['theta']) * self.delta/4
                new_y = new_y + v * np.sin(car['theta']) * self.delta/4

            car['x'],car['y'] = np.clip(new_x, -self.area_size/2, self.area_size/2),np.clip(new_y, -self.area_size/2, self.area_size/2)
            car['prev_rho'],car['prev_phi']=car['rho'],car['phi']
            car['rho'],car['phi'] =np.sqrt((-car['x']+car['goal_x'])**2 + (-car['y']+car['goal_y'])**2),np.arctan2(-car['y']+car['goal_y'],-car['x']+car['goal_x']) - car['theta']
            
            if check_collision(car['x'], car['y'],self.Env_rectangle_obs,self.Env_circle_obs,self.Env_Stringline,self.safety_distance):
                car['obs_collision'] = True
            elif check_car_collision([car['x'], car['y']], [[c['x'],c['y']] for c in self.cars if c != car],self.safety_distance):
                car['car_collision'] = True
            elif car['rho'] < self.safety_distance:
                car['reached_goal'] = True
  
    def get_observation(self, car):
        if car['updated']==0:
            return np.full(self.obs_dim,0)
        car['prev_min_laser']=car['min_laser']
        car['prev_min_laser_index']=car['min_laser_index']
        if not self.turnoff_sensor:
            # print(car['car_id'],car['dynamic_obstacles_edges_list'],self.total_edge_list_len,self.Env_edges_list_len,self.num_sensors,self.sensor_range,car['theta'],car['x'], car['y'])
            # get_sensor_data_C.
            sensor_data = get_sensor_data_C.get_sensor_data1(car['car_id'],self.Env_edges_list,self.num_sensors,self.sensor_range,car['theta'],car['x'], car['y'],\
                [(ocar['x'],ocar['y'],self.disk_radius) for ocar in self.cars]+ self.Env_circle_obs) 
            if self.add_sensor_noise:
                sensor_data=np.clip(sensor_data+np.random.normal(loc=0, scale=self.sensor_gussian_noise,size=sensor_data.shape),self.disk_radius,self.sensor_range) #add gaussian noise to sensor
            car['min_laser_index'] = np.argmin(sensor_data)
            car['min_laser'] = sensor_data[car['min_laser_index']]
            self.obs_his[car['car_id']%self.num_agent,:-1,:]=self.obs_his[car['car_id']%self.num_agent,1:,:].copy()
            # self.obs_his[car['car_id']%self.num_agent,-1,:]=np.concatenate((sensor_data,[car['rho'],car['phi'],car['omega'],car['v'],car['theta']])).copy()
            self.obs_his[car['car_id']%self.num_agent,-1,:]=sensor_data.copy()
            if car['STEP']==0 and self.his_len>1:
                for i in range(self.his_len-1):
                    self.obs_his[car['car_id']%self.num_agent,i,:]=self.obs_his[car['car_id']%self.num_agent,-1,:].copy()
            observation=self.obs_his[car['car_id']%self.num_agent,:,:].reshape(-1)
            # observation = np.concatenate((observation,[car['omega'],car['v'],car['rho'],car['phi']]))
            observation = np.concatenate((observation,[car['theta'],car['omega'],car['v'],car['x'],car['y'],car['goal_x'],car['goal_y']]))
            
            if self.save_history:
                car['history'].append(np.concatenate((sensor_data,np.array([car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y'],\
                    car['timeout'],car['obs_collision'], car['car_collision'],car['reached_goal'],car['omega'],car['v']]))))
        else:
            observation = np.array([car['theta'],car['omega'],car['v'],car['x'],car['y'],car['goal_x'],car['goal_y']])
            if self.save_history:
                car['history'].append((np.array([car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y'],\
                    car['timeout'],car['obs_collision'], car['car_collision'],car['reached_goal'],car['omega'],car['v']])))
        return observation
    def get_state(self):
        # for each car: flag,x,y, theta,min_laser,min_laser_index
        for i,car in enumerate(self.cars):
            if car['updated']==0:#out
                self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[0]*self.state_dim_peragent
            else:
                self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[car['theta'],car['rho'],car['phi'],car['omega'],car['v'],car['min_laser'],car['min_laser_index']]
        return np.expand_dims(self.state,axis=0).repeat(self.num_agent,axis=0)

    def reset(self):
        self.cars = self.generate_cars_Env()
        self.episode += 1
        observations = [self.get_observation(car) for car in self.cars]
        return observations,self.get_state(),None
        # return observations

    def get_reward(self):
        rewards=np.zeros([self.num_cars,1],dtype=np.float32)
        for i,car in enumerate(self.cars):
            if car['updated']==0:
                continue
            if self.reward_type=='binary':
                if car['reached_goal']:
                    rewards[i][0]=0
                else:
                    rewards[i][0]=-1
            elif self.reward_type=='shaping':
                rewards[i][0]=(car['prev_rho']-car['rho'])*3
                if car['obs_collision'] or car['car_collision']:
                    rewards[i][0]+=-10
                if car['reached_goal']:
                    rewards[i][0]+=100

            else:
                raise NotImplementedError
            # reward_smooth=0#0.01*(car['v']/self.default_speed-abs(car['omega']/self.default_omega))#0#-0.1
            # reward_laser=0
            # reward_rel=-1#(car['prev_rho']-car['rho'])*3
            
            # if car['obs_collision'] or car['car_collision']: # Crashed 
            #     # reward_laser=-10
            #     if self.reward_type=='binary':rewards[i][0]=-10# for GRU_attenion based TD3, single Critic
            #     else:rewards[i]=[ reward_rel , reward_laser,reward_smooth]
            # # comment if in CA mode
            # elif car['reached_goal']:
            #     reward_rel=0#100
            #     if self.reward_type=='pR':rewards[i][0]=80# for GRU_attenion based TD3, single Critic
            #     else:rewards[i]=[reward_rel , reward_laser,reward_smooth]
            # else: 
            #     if self.reward_type=='pR':rewards[i][0]=reward_smooth# for GRU_attenion based TD3, single Critic
            #     else:rewards[i]=[reward_rel, reward_laser,reward_smooth]
        
        # rewards = [1 if car['reached_goal'] else -1 if car['obs_collision'] or car['car_collision'] else 0 for car in self.cars] #when timeout, reward=0
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
        if self.save_history:
            #'updated':car['updated'],'real_done':car['obs_collision'] or car['car_collision'] or car['reached_goal'],
            infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': car['reached_goal'], 'timeout': car['timeout'], 'history': car['history']} for car in self.cars]
        else:
            #'updated':car['updated'],'real_done':car['obs_collision'] or car['car_collision'] or car['reached_goal'], 
            infos=[{'step':car['STEP'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': car['reached_goal'], 'timeout': car['timeout']} for car in self.cars]

        # if any([car['obs_collision'] or car['car_collision'] for car in self.cars]):
        #     return [observations, self.get_state(),rewards, [True for car in self.cars], infos]
        shared_obs=self.get_state()
        # self.time_c[3]+=time.time()-start
        # if self.count%1000==0:
        #     print(f"step time:{self.time_c/np.sum(self.time_c)}")

        return [observations, shared_obs,rewards, [car['obs_collision'] or car['car_collision'] or car['reached_goal'] or car['timeout'] for car in self.cars], infos,None]
        # return [observations, rewards, [car['obs_collision'] or car['car_collision'] or car['reached_goal'] or car['timeout'] for car in self.cars], infos]

    def render(self, trajectory=[], mode='human',gif_name=None):
        # trajectory should be a list of num_agent of [sensor,car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y']] tuples
        fig, ax = plt.subplots()
        self.plot_sensor_line = self.plot_sensor_line if not self.turnoff_sensor else False
        step = 0
        max_step = max([len(traj) for traj in trajectory])
        print(f"max step: {max_step}")
        all_frames = []
        def update(frame):
            ax.clear()
            self.SimEnv.display_environment(ax=ax)
            ax.set_title("Car Trajectories after time step {}".format(frame), fontsize=16)
            ax.set_xlabel("X (m)", fontsize=16)
            ax.set_ylabel("Y (m)", fontsize=16)
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
                    ax.plot(x, y, marker='o')
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
                    ax.plot(goalx, goaly, 'o', markersize=10)
                elif self.plot_traj:
                    state=traj[-1]
                    goalx = state[offset + 3]
                    goaly = state[offset + 4]
                    ax.plot(goalx, goaly, 'o', markersize=10)
                    x_his = [s[offset] for s in traj[:]]
                    y_his = [s[offset + 1] for s in traj[:]]
                    ax.plot(x_his, y_his)
            if mode == 'rgb_array':
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                all_frames.append(image)
        ani = FuncAnimation(fig, update, frames=max_step, repeat=False)
        plt.show()
        if mode == 'rgb_array' and gif_name!=None:
            # Save the frames as a video
            imageio.mimsave(gif_name, all_frames, fps=1/self.ifi)
    def close(self):
        plt.close()