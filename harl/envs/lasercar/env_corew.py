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
def get_sensor_data_single(edges_list,edlen, sensor_range,car_theta,car_x, car_y,angle):
    sensor_angle = car_theta + angle
    min_distance = sensor_range
    contact_edge=-1
    for edge_ind in range(edlen-16):
        x1, y1, x2, y2=edges_list[edge_ind]
        # print(edlen,edge_ind,x1, y1, x2, y2)
    # for x1, y1, x2, y2 in edges_list:
        intersection, ix, iy= ray_intersects_segment(car_x, car_y, sensor_angle, x1, y1, x2, y2)
        if intersection:
            distance = np.sqrt((ix - car_x) ** 2 + (iy - car_y) ** 2)
            if distance<min_distance:
                min_distance=distance
                contact_edge=edge_ind
            # min_distance = min(min_distance, distance)
    
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
def get_sensor_data1(egoid,edges_list,edlen,Env_edges_list_len,num_sensors,sensor_range,car_theta,car_x, car_y):
    sensor_data = np.full(num_sensors, sensor_range)

    angles = np.linspace(-np.pi / 2, np.pi / 2, num_sensors)
    for i in range(num_sensors):
        sensor_data[i],contact_edge = get_sensor_data_single(edges_list,edlen,sensor_range,car_theta,car_x,car_y,angles[i])
    return sensor_data
from gymnasium.utils import seeding

class EnvCorew(gymnasium.Env):
    """
    # 环境中的智能体
    """
    metadata = {"render.modes": ["human", "txt"]}
    def __init__(self,all_args):
        
        self.num_sensors = all_args.num_sensors
        self.Car_resettype=all_args.Car_resettype #random_target,diag
        self.num_cars = all_args.num_agents
        self.num_agent=all_args.num_agents
        self.add_sensor_noise=all_args.add_sensor_noise
        self.fixed_theta=all_args.fixed_theta
        self.other_obs_dim=7
        self.state_dim_peragent=7
        self.obs_dim_ego=self.num_sensors
        # self.obs_dim = self.obs_dim_ego+self.other_obs_dim*(self.num_agent-1)  # 设置智能体的观测维度 # set the observation dimension of agents
        self.his_len=3
        self.obs_dim = self.obs_dim_ego*self.his_len+6
        self.share_observation_dim= self.state_dim_peragent*self.num_agent
        self.obs_his=np.zeros((self.num_agent,self.his_len,self.obs_dim_ego),dtype=np.float32)# his_length=3,0-2:t-2,t-1,t
        self.state=np.zeros(self.share_observation_dim,dtype=np.float32)
        self.action_dim = 1  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional

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
        self.Env_obstacles = self.SimEnv.obstacles
        self.Env_Stringline=self.SimEnv.Stringline
        self.Env_edges_list=self.SimEnv.Env_edges_list
        self.Env_edges_list_len=len(self.Env_edges_list)
        self.total_edge_list_len=self.Env_edges_list_len + self.num_agent*16
        self.outofbound_edgelist=get_edges(list(Point(-10000,-10000).buffer(self.disk_radius,4).exterior.coords))
        # self.default_obstacle_edgelist=List([edge for edge in self.Env_edges_list]+[edge for edge in self.outofbound_edgelist]*self.num_cars)
        self.default_obstacle_edgelist=[edge for edge in self.Env_edges_list]+[edge for edge in self.outofbound_edgelist]*(self.num_cars-1)
        self.cars = self.generate_cars_Env()
        self.generate_dynamic_obstacles_edgelist()
        
        self.default_speed = 1
        self.default_omega = 1
        self.max_steps = 2*self.area_size/(self.delta*self.default_speed) # Maximum number of steps per episode, depending on the environment and task
        
        self.episode = 0
        self.action_space = [spaces.Box(low=np.array([-self.default_omega]), high=np.array([self.default_omega]), dtype=np.float32) for _ in range(self.num_agent)]
        self.observation_space = [spaces.Box(low=0, high=self.sensor_range, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim * self.num_agent,), dtype=np.float32)
            for _ in range(self.num_agent)]

        self.sensor_lines = []
        self.reward_type="binary"#binary,shaping,pR,lp_sn,lp_wosn
        self.share_reward=all_args.share_reward
        self.sensor_gussian_noise=all_args.sensor_gussian_noise
        self.save_history=all_args.save_history
        
        self.plot_traj=all_args.plot_traj
        self.traj_hold=all_args.traj_hold
        self.plot_sensor_line=all_args.plot_sensor_line
        self.ifi=all_args.ifi
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
                        'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i,'dynamic_obstacles_edges_list':self.default_obstacle_edgelist[:] }
                        
                car_point = Point(x, y)
                if not any(car_point.within(ob) for ob in self.Env_obstacles) and not self.check_car_collision(new_car, cars):
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
                        'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i,'dynamic_obstacles_edges_list':self.default_obstacle_edgelist[:],'history': []}
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
                        'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i,'dynamic_obstacles_edges_list':self.default_obstacle_edgelist[:],'history': []}
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
                        'STEP':0,'previous_omega':0,'previous_v':0,'car_id': self.EnvId*self.num_cars+i,'dynamic_obstacles_edges_list':self.default_obstacle_edgelist[:],'history': []}
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
            v = self.default_speed
            w = action
            car['previous_omega']=car['omega']
            car['previous_v']=car['v']
            car['v'] = v
            car['omega'] = w
            car['previous_x']=car['x']
            car['previous_y']=car['y']

            theta0=car['theta']
            
            new_x,new_y = car['x'],car['y']
            for _ in range(4):
                car['theta'] += w * self.delta/4
                new_x = new_x + v * np.cos(car['theta']) * self.delta/4
                new_y = new_y + v * np.sin(car['theta']) * self.delta/4

            # new_x = car['x'] + v/w * (np.sin(car['theta'])-np.sin(theta0))
            # new_y = car['y'] + v/w * (-np.cos(car['theta'])+np.cos(theta0))
            
            new_x = np.clip(new_x, -self.area_size/2, self.area_size/2)
            new_y = np.clip(new_y, -self.area_size/2, self.area_size/2)

            if self.check_collision(new_x, new_y):
                car['obs_collision'] = True
                # print("Collision with obstacle detected. Stopping the car.")
            # elif self.check_car_collision({'x': new_x, 'y': new_y}, [c for c in self.cars if c != car and not c['obs_collision'] and not c['car_collision']]):
            elif self.check_car_collision({'x': new_x, 'y': new_y}, [c for c in self.cars if c != car]):#check all cars still alive in this step
                car['car_collision'] = True
                # print("Collision with another car detected. Stopping the car.")
            else:
                car['x'] = new_x
                car['y'] = new_y
                # car['history'].append((car['x'], car['y'], car['theta']))

                if Point(new_x, new_y).distance(Point(car['goal_x'], car['goal_y'])) < self.safety_distance:
                    car['reached_goal'] = True
            car['prev_rho']=car['rho']
            car['prev_phi']=car['phi']
            car['rho']=np.sqrt((-car['x']+car['goal_x'])**2 + (-car['y']+car['goal_y'])**2)
            car['phi'] = np.arctan2(-car['y']+car['goal_y'],-car['x']+car['goal_x']) - car['theta']
        self.generate_dynamic_obstacles_edgelist()

    def check_collision(self, x, y):
        car_point = Point(x, y)
        for obstacle in self.Env_obstacles:
            if car_point.distance(obstacle) < self.safety_distance:
                return True
        for line in self.Env_Stringline:
            if car_point.distance(line) < self.safety_distance:
                return True
        return False
    def check_car_collision(self, new_car, cars):
        #theoretically, this function should first check two points can see-through, then check the distance
        new_car_point = Point(new_car['x'], new_car['y'])
        for car in cars:
            car_point = Point(car['x'], car['y'])
            if new_car_point.distance(car_point) < 2*self.safety_distance:
                return True
        return False
    
    def get_observation(self, car):
        if car['updated']==0:
            return np.full(self.obs_dim,0)
        # print(car['car_id'],car['dynamic_obstacles_edges_list'],self.total_edge_list_len,self.Env_edges_list_len,self.num_sensors,self.sensor_range,car['theta'],car['x'], car['y'])
        # exit()
        sensor_data = get_sensor_data1(car['car_id'],car['dynamic_obstacles_edges_list'],self.total_edge_list_len,self.Env_edges_list_len,self.num_sensors,self.sensor_range,car['theta'],car['x'], car['y']) 
        if self.add_sensor_noise:
            sensor_data=np.clip(sensor_data+np.random.normal(loc=0, scale=self.sensor_gussian_noise,size=sensor_data.shape),self.disk_radius,self.sensor_range) #add gaussian noise to sensor
        # sensor_data,contact_edge_type = get_sensor_data(car['car_id'],car['dynamic_obstacles_edges_list'],self.total_edge_list_len,self.Env_edges_list_len,self.num_sensors,self.sensor_range,car['theta'],car['x'], car['y'])  
        car['prev_min_laser']=car['min_laser']
        car['prev_min_laser_index']=car['min_laser_index']
        car['min_laser_index'] = np.argmin(sensor_data)
        car['min_laser'] = sensor_data[car['min_laser_index']]
        
        # for othercar in self.cars:
            
        #     if car['car_id']!= othercar['car_id']:
        #         inter_ind=othercar['car_id']%self.num_cars if othercar['car_id']<car['car_id'] else othercar['car_id']%self.num_cars-1
        #         if othercar['updated']==0:#othercar out
        #             car['other_car'][inter_ind*self.other_obs_dim:(inter_ind+1)*self.other_obs_dim]=np.zeros(self.other_obs_dim)
        #             car['other_car'][inter_ind*self.other_obs_dim]=othercar['car_id']
        #         else:
        #             # insight=(contact_edge_type==othercar['car_id']).any()
        #             rel_theta= np.arctan2(-car['y']+othercar['y'],-car['x']+othercar['x']) - car['theta']
        #             rel_dis=np.sqrt((-car['y']+othercar['y'])**2+(-car['x']+othercar['x'])**2)
        #             # car['other_car'][inter_ind*self.other_obs_dim:(inter_ind+1)*self.other_obs_dim]=np.array([othercar['car_id'],2 if insight else 1,rel_theta,rel_dis, othercar['omega'],othercar['v'],othercar['theta']])
        #             car['other_car'][inter_ind*self.other_obs_dim:(inter_ind+1)*self.other_obs_dim]=np.array([othercar['car_id'],2 ,rel_theta,rel_dis, othercar['omega'],othercar['v'],othercar['theta']]) if insight else \
        #                 np.array([othercar['car_id'],0,0,0,0,0,0]) #only show info of agents which insight,a.k.a, P.O
   
        # observation = np.concatenate((sensor_data,contact_edge_type,[car['x'],car['y'],car['rho'],car['phi'],car['omega'],car['v'],car['theta']],car['other_car'][:]))
        # observation = np.concatenate((sensor_data,[car['x'],car['y'],car['rho'],car['phi'],car['omega'],car['v'],car['theta']],car['other_car'][:]))
        # observation = np.concatenate((sensor_data,[car['rho'],car['phi'],car['omega'],car['v'],car['theta']],car['other_car'][:]))
        self.obs_his[car['car_id']%self.num_agent,:-1,:]=self.obs_his[car['car_id']%self.num_agent,1:,:].copy()
        # self.obs_his[car['car_id']%self.num_agent,-1,:]=np.concatenate((sensor_data,[car['rho'],car['phi'],car['omega'],car['v'],car['theta']])).copy()
        self.obs_his[car['car_id']%self.num_agent,-1,:]=sensor_data.copy()
        if car['STEP']==0 and self.his_len>1:
            for i in range(self.his_len-1):
                self.obs_his[car['car_id']%self.num_agent,i,:]=self.obs_his[car['car_id']%self.num_agent,-1,:].copy()
        observation=self.obs_his[car['car_id']%self.num_agent,:,:].reshape(-1)
        # observation = np.concatenate((observation,[car['omega'],car['v'],car['rho'],car['phi']]))
        observation = np.concatenate((observation,[car['theta'],car['omega'],car['x'],car['y'],car['goal_x'],car['goal_y']]))
        if self.save_history:
            car['history'].append(np.concatenate((sensor_data,np.array([car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y'],\
                car['timeout'],car['obs_collision'], car['car_collision'],car['reached_goal'],car['omega'],car['v']]))))
        return observation
    def get_state(self):
        # for each car: flag,x,y, theta,min_laser,min_laser_index
        for i,car in enumerate(self.cars):
            if car['updated']==0:#out
                self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[0]*self.state_dim_peragent
            else:
                self.state[i*self.state_dim_peragent:(i+1)*self.state_dim_peragent]=[car['theta'],car['rho'],car['phi'],car['omega'],car['v'],car['min_laser'],car['min_laser_index']]
        return self.state

    def reset(self):
        self.cars = self.generate_cars_Env()
        self.episode += 1
        observations = [self.get_observation(car) for car in self.cars]
        # return observations,self.get_state()
        return observations

    def get_reward(self):
        rewards=np.zeros([self.num_cars,3],dtype=np.float32)
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
        if self.share_reward:
            return [[np.mean([r.sum() for r in rewards])]]*self.num_agent
        else:
            return [[r.sum()] for r in rewards]
    
    def step(self, actions):
        self.update_state(actions)
        observations = [self.get_observation(car) for car in self.cars]
        # observations_f16=[obs.astype(np.float16) for obs in observations]
        rewards = self.get_reward()
        if self.save_history:
            infos=[{'step':car['STEP'],'updated':car['updated'],'real_done':car['obs_collision'] or car['car_collision'] or car['reached_goal'],'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': car['reached_goal'], 'timeout': car['timeout'], 'history': car['history']} for car in self.cars]
        else:
            infos=[{'step':car['STEP'],'updated':car['updated'],'real_done':car['obs_collision'] or car['car_collision'] or car['reached_goal'], 'obs_collision': car['obs_collision'], 'car_collision': car['car_collision'], 'reached_goal': car['reached_goal'], 'timeout': car['timeout']} for car in self.cars]

        # if any([car['obs_collision'] or car['car_collision'] for car in self.cars]):
        #     return [observations, self.get_state(),rewards, [True for car in self.cars], infos]
        # return [observations, self.get_state(),rewards, [car['obs_collision'] or car['car_collision'] or car['reached_goal'] or car['timeout'] for car in self.cars], infos]
        return [observations, rewards, [car['obs_collision'] or car['car_collision'] or car['reached_goal'] or car['timeout'] for car in self.cars], infos]

    def render(self, trajectory=[], mode='human',gif_name=None,plot_sensor_line=True):
        # trajectory should be a list of num_agent of [sensor,car['x'],car['y'],car['theta'],car['goal_x'],car['goal_y']] tuples
        fig, ax = plt.subplots()
        self.plot_sensor_line=plot_sensor_line
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
            for idx,traj in enumerate(trajectory):
                if frame < len(traj):
                    state = traj[frame]
                    sensor_data = state[0:self.num_sensors]
                    x,y,theta,goalx,goaly = state[self.num_sensors],state[self.num_sensors + 1],state[self.num_sensors + 2],state[self.num_sensors + 3],state[self.num_sensors + 4]
                    timeout,obs_collision, car_collision,reached_goal=state[self.num_sensors + 5],state[self.num_sensors + 6],state[self.num_sensors + 7],state[self.num_sensors + 8]
                    if timeout:print("{} timeout".format(idx))
                    if obs_collision:print("{} obs_collision".format(idx))
                    if car_collision:print("{} car_collision".format(idx))
                    if reached_goal:print("{} reached_goal".format(idx))
                    # print([idx,state[-1],state[-2],goalx,goaly])
                    ax.plot(x, y, marker='o')
                    # Plot trajectory
                    if self.plot_traj:
                        x_his = [s[self.num_sensors] for s in traj[:frame + 1]]
                        y_his = [s[self.num_sensors + 1] for s in traj[:frame + 1]]
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
                elif self.traj_hold:
                    if self.plot_traj:
                        state=traj[-1]
                        goalx = state[self.num_sensors + 3]
                        goaly = state[self.num_sensors + 4]
                        ax.plot(goalx, goaly, 'o', markersize=10)
                        x_his = [s[self.num_sensors] for s in traj[:]]
                        y_his = [s[self.num_sensors + 1] for s in traj[:]]
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