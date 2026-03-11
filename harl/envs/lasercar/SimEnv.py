import numpy as np
from shapely.geometry import Polygon, Point,LineString
from shapely import affinity
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import random
import heapq,os,sys
# os.environ["NETWORKX_BACKEND_PRIORITY"] = 'cugraph'#'cugraph'
import pickle,copy,time
import networkx as nx
from shapely.strtree import STRtree
from concurrent.futures import ThreadPoolExecutor
# nx.config.backend_priority = ["cugraph"]
import torch.multiprocessing as mp
from harl.envs.lasercar.random_scenario import generate_obstacles_and_points
# from random_scenario import generate_obstacles_and_points
quad_segs=4
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
def sigmoid(x):
    indices_pos = np.nonzero(x >= 0)
    indices_neg = np.nonzero(x < 0)
    x[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
    x[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
    return x
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

def line_to_rectangle(line_segment, width):
    """
    Convert a line segment to a rectangle with specified width
    
    Parameters:
    - line_segment: List of two points [(x1, y1), (x2, y2)]
    - width: Width of the rectangle perpendicular to the line segment
    
    Returns:
    - Rectangle as a list of four corner points in counter-clockwise order
    """
    # Extract points
    p1, p2 = line_segment
    
    # Get the direction vector of the line segment
    direction = np.array(p2) - np.array(p1)
    
    # Get the perpendicular vector (rotate 90 degrees)
    # For a 2D vector (x, y), perpendicular is (-y, x) or (y, -x)
    perpendicular = np.array([-direction[1], direction[0]])
    
    # Normalize perpendicular vector and scale to desired width
    if np.linalg.norm(perpendicular) > 0:
        perpendicular = perpendicular / np.linalg.norm(perpendicular) * width
    else:
        perpendicular = np.array([0, width])
    
    # Calculate the four corners of the rectangle
    corners = [
        p1,
        p2,
        tuple(np.array(p2) + perpendicular),
        tuple(np.array(p1) + perpendicular)
    ]
    
    return corners

def gen_poly(position=[0,0],width=0.5,length=1,angle=0):
    #generate a polygon with given position, width, length and angle
    x = position[0]
    y = position[1]
    w = width
    l = length
    a = angle
    poly = Polygon([(x - w / 2, y - l / 2), (x + w / 2, y - l / 2), 
                    (x + w / 2, y + l / 2), (x - w / 2, y + l / 2)])
    return affinity.rotate(poly, a, origin='centroid')
def get_edges(coords):
    edges = []
    for i in range(len(coords) - 1):
        edges.append((*coords[i], *coords[i+1]))
    return edges

def rrt(start, goal, env, max_iter=50000, step_size=0.1):
    """RRT algorithm to find a path from start to goal."""
    tree = {start: None}
    for i in range(max_iter):
        random_point = (random.uniform(-env.area_size/2, env.area_size/2),
                        random.uniform(-env.area_size/2, env.area_size/2))
        
        nearest_point = min(tree.keys(), key=lambda p: np.linalg.norm(np.array(p) - np.array(random_point)))
        direction = np.array(random_point) - np.array(nearest_point)
        direction = direction / np.linalg.norm(direction)
        new_point = tuple(np.array(nearest_point) + direction * step_size)
        
        if not env.is_collision(new_point):
            tree[new_point] = nearest_point
            if np.linalg.norm(np.array(new_point) - np.array(goal)) < step_size:
                tree[goal] = new_point
                return trace_path(tree, goal)
    
    return None

def trace_path(tree, goal):
    """Trace back the path from goal to start."""
    path = [goal]
    while tree[path[-1]] is not None:
        path.append(tree[path[-1]])
    return path[::-1]

def heuristic(a, b):
    """Calculate Euclidean distance heuristic."""
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(start, goal, env):
    """A* algorithm to find the shortest path from start to goal."""
    open_set = []
    heapq.heappush(open_set, (0, start))  # Push the starting point into the priority queue
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        # If we have reached the goal
        if current == goal:
            return reconstruct_path(came_from, current)

        # Explore neighbors in the environment (grid or continuous)
        for dx, dy in [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]:  # You can tweak step size here
            neighbor = (current[0] + dx, current[1] + dy)

            # Skip neighbors that result in collisions
            if env.is_collision(neighbor):
                continue

            # Tentative g score for this neighbor
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            # If this path is better, record it
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def reconstruct_path(came_from, current):
    """Reconstruct the path from the came_from map."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

    
class SimulationEnvironment(object):
    def __init__(self):
        self.robot_radius = 0.2
        self.area_size=0
        self.obstacles=[]
        self.lines=[]
        self.positions=[]
        self.counter=0
        self.time_c=np.zeros(4)
        self.counter1=0
        self.time_c1=np.zeros(4)

    def sample_points(self):
        """Randomly sample points in the environment."""
        samples = []
        while len(samples) < self.num_samples:
            sample = (random.uniform(-self.area_size / 2, self.area_size / 2),
                      random.uniform(-self.area_size / 2, self.area_size / 2))
            if not self.is_collision(sample):
                samples.append(sample)
        return samples
    def build_tree(self):
        # 预处理几何对象
        self.precomputed_buffers = [obs for obs in self.obstacles]
        self.lines_ls = [LineString(line) for line in self.lines]
        # self.merge=self.precomputed_buffers+ self.lines_ls
        # 构建空间索引
        # self.obstacle_tree = STRtree(self.merge,node_capacity=20)
        self.obstacle_tree = STRtree(self.precomputed_buffers,node_capacity=20)
        self.line_tree = STRtree(self.lines_ls,node_capacity=20)        
    def build_roadmap(self,num_samples=100,k=20):
        self.build_tree()

        self.num_samples = num_samples
        self.k = k
        self.graph = nx.Graph()
        self.incremental_nodeset=[]
        """Build the roadmap using random sampling and k-nearest neighbors."""
        # Check if the roadmap already exists
        # file=f"/project/HARL-main/examples/{type(self).__name__}_{num_samples}_{k}_roadmap.pkl"
        file=f"/home/user/yhq/HARL-main/examples/{type(self).__name__}_{num_samples}_{k}_roadmap.pkl"
        # print(file)
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.nodes,self.nodes_array,self.graph,self.cost_matrix = pickle.load(f)
        else:
            self.nodes = self.sample_points()
            self.nodes_array = np.array(self.nodes)

            # Add nodes to the graph
            for sample in self.nodes:
                self.graph.add_node(sample)
            # self.kth_nearest_neighbors_dis = [0]*self.num_samples
            # self.nearest_neighbors=[[] for _ in range(self.num_samples)]
            # Connect nodes using k-nearest neighbors
            for i, sample in enumerate(self.nodes):
                distances = [(self.distance(sample, other), other) for other in self.nodes if other != sample]
                distances.sort()  # Sort by distance
                count=0
                for w, neighbor in distances:
                    if not self.is_collision((sample, neighbor)):
                        self.graph.add_edge(sample, neighbor, weight=w)
                        # self.kth_nearest_neighbors_dis[i]=w
                        # self.nearest_neighbors[i].append((w,neighbor))
                        count+=1
                    if count >= self.k:
                        break
            
            shortest_paths_weighted = nx.all_pairs_dijkstra_path(self.graph, weight='weight')
            self.cost_matrix = np.zeros((self.num_samples, self.num_samples), dtype=np.float32)
            for source,paths in shortest_paths_weighted:
                for target, path in paths.items():
                    src_index = self.nodes.index(source)
                    target_index = self.nodes.index(target)
                    cost=pathlength(path)
                    self.cost_matrix[src_index][target_index]=cost
                    self.cost_matrix[target_index][src_index]=cost
            print("graph conneted:",nx.is_connected(self.graph))
            # self.s_Queue=mp.Queue(4)
            # self.adjw_Queue=mp.Queue(4)
            # for _ in range(4):#
            #     p = mp.Process(target=self.get_adj_w_asy,args=())
            #     p.daemon = True
            #     p.start()

            with open(file, "wb") as f:
                pickle.dump([self.nodes,self.nodes_array,self.graph,self.cost_matrix], f)

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def find_path(self, start_set=None,goal_set=None, start=None, goal=None):
        # Use provided points or default to class attributes
        # if start==None and goal==None:
        #     if isinstance(self.start, list) and isinstance(self.goal, list):
        #         return [nx.shortest_path(self.graph, self.start[i], self.goal[i], weight='weight') for i in range(len(self.start))]
        #     return nx.shortest_path(self.graph, self.start, self.goal, weight='weight')#nx-cugraph,networkx,backend="networkx"
        assert start!=None and goal!=None
        start_point = start 
        goal_point = goal 

        # if len(self.incremental_nodeset)>0:
        #     self.graph.remove_nodes_from(self.incremental_nodeset)
        #     self.incremental_nodeset=[]
        # self.time_c[0]+=time.time()-start_time
        # start_time=time.time()
        # Handle list inputs (multiple start/goal points)
        if isinstance(start_point, list) and isinstance(goal_point, list):
            # For multiple start and goal points
            # self.time_c[1]+=time.time()-start_time
            # start_time=time.time()
                   
            # for point in start_set:
            #     self.s_Queue.put(point,block=True,timeout=200)
            # s_adj_list_tmp=[]
            # s_tmp=[]
            # s_adj_list=[]
            # while len(s_adj_list_tmp)<len(start_set):
            #     s, s_adj = self.adjw_Queue.get(block=True,timeout=200)
            #     s_adj_list_tmp.append(s_adj)
            #     s_tmp.append(s)
            #     s_adj_list.append([])
            # for i,point in enumerate(start_set):
            #     s_adj_list[i]=s_adj_list_tmp[s_tmp.index(point)]
                
            s_adj_list=[self.get_adj_w(point) for point in start_set]
            g_adj_list=[self.get_adj_w(point) for point in goal_set] 
            # print(g_adj_list)               
            # self.time_c[2]+=time.time()-start_time
            # start_time=time.time()
            # path=[nx.shortest_path(self.graph, start_point[i], goal_point[i], weight='weight') for i in range(len(start_point))]
            # path=[self.find_path_with_heuristic( start=start_point[i],start_adj=s_adj_list[i], goal=goal_point[i],goal_adj=g_adj_list[i]) for i in range(len(start_point))]
            path=[self.find_path_with_heuristic( start=start_point[i],start_adj=s_adj_list[start_set.index(start_point[i])], goal=goal_point[i],goal_adj=g_adj_list[goal_set.index(goal_point[i])]) for i in range(len(start_point))]
            # path=[self.find_path_with_heuristic( start=start_point[i],start_adj=s_adj_list[start_set.index(start_point[i])], goal=goal_point[i],goal_adj=[]) for i in range(len(start_point))]


            return path
        else:
            # For single start and goal points
            # Add points to graph if not already there
            if start_point not in self.graph:
                self.add_node_with_connections(start_point)
            if goal_point not in self.graph:
                self.add_node_with_connections(goal_point)
            # Find shortest path
            return nx.shortest_path(self.graph, start_point, goal_point, weight='weight')
    def get_adj_w(self, s):
        if s not in self.nodes:
            s_adj=self.add_node_with_connections(s)
        else:
            s_adj=[]
        return s_adj
    def get_adj_w_asy(self):
        while True:
            s=self.s_Queue.get(block=True,timeout=200)
            if s==None:
                break
            s_adj=self.get_adj_w(s)
            self.adjw_Queue.put((s,s_adj),block=True,timeout=200)
    def find_path_with_heuristic(self, start=None,start_adj=[], goal=None,goal_adj=[]):
        # Use self.cost_matrix to find the path
        if len(start_adj)==0 and len(goal_adj)==0:
            return self.cost_matrix[self.nodes.index(start)][self.nodes.index(goal)]
        elif len(start_adj)>0 and len(goal_adj)>0:
            start_adj_point_index=np.array([self.nodes.index(s[0]) for s in start_adj])
            start_adj_w=[s[1] for s in start_adj]
            goal_adj_point_index=np.array([self.nodes.index(s[0]) for s in goal_adj])
            goal_adj_w=[s[1] for s in goal_adj]
            sub_matrix = self.cost_matrix[np.ix_(start_adj_point_index, goal_adj_point_index)]            
            for i in range(len(start_adj_w)):
                for j in range(len(goal_adj_w)):
                    sub_matrix[i,j]=sub_matrix[i,j]+start_adj_w[i]+goal_adj_w[j]            
            return sub_matrix.min()
        elif len(start_adj)>0 and len(goal_adj)==0:
            start_adj_point_index=np.array([self.nodes.index(s[0]) for s in start_adj])
            start_adj_w=[s[1] for s in start_adj]
            # print(self.nodes[np.argmin(np.linalg.norm(np.array(self.nodes)-np.array(goal)))],goal)
            goal_index = np.array([self.nodes.index(goal)])
            
            sub_matrix = self.cost_matrix[np.ix_(start_adj_point_index, goal_index)]
            for i in range(len(start_adj_w)):
                sub_matrix[i,0]=sub_matrix[i,0]+start_adj_w[i]
            return sub_matrix.min()
        else:
            #len(start_adj)==0 and len(goal_adj)>0:
            # print(len(start_adj),len(goal_adj),start_adj,goal_adj)
            goal_adj_point_index=np.array([self.nodes.index(s[0]) for s in goal_adj])
            goal_adj_w=[s[1] for s in goal_adj]
            start_index = np.array([self.nodes.index(start)])
            sub_matrix = self.cost_matrix[np.ix_(start_index, goal_adj_point_index)]
            for i in range(len(goal_adj_w)):
                sub_matrix[i,0]=sub_matrix[i,0]+goal_adj_w[i]
            return sub_matrix.min()
    def add_node_with_connections(self, point):
        # Add the node to the graph
        # self.counter1+=1   
        # start_time=time.time()
        # self.graph.add_node(point)
        # self.incremental_nodeset.append(point)
        # self.time_c1[0]+=time.time()-start_time
        # start_time=time.time()
        # Calculate distances to all existing nodes
        # distances = [(self.distance(point, other), other) for other in self.nodes]
        distances = np.linalg.norm(np.array(point).reshape(1,2).repeat(self.num_samples,axis=0)-self.nodes_array,axis=1)
        # self.time_c1[1]+=time.time()-start_time
        # start_time=time.time()        
        # self.kth_nearest_neighbors_dis.append(0)
        # self.nearest_neighbors.append([])
        # distances=copy.deepcopy(other_distances)
        # distances.sort()  # Sort by distance
        # self.time_c1[2]+=time.time()-start_time
        # start_time=time.time()        
        # Connect to k nearest neighbors without collisions
        count = 0
        adjecent_nodes = []
        while count < 1:#self.k
            index=np.argmin(distances)
            w=distances[index]
            neighbor=self.nodes[index]
            if not self.is_collision((point, neighbor)):
                # self.graph.add_edge(point, neighbor, weight=w)
                count += 1
                adjecent_nodes.append((neighbor,w))
            distances[index]=np.inf
        # self.time_c1[3]+=time.time()-start_time
        # start_time=time.time()  
        # if self.counter1%100==0:
        #     print("time add_node_with_connections:",self.time_c1/np.sum(self.time_c1))          
        # if count == 0:
        #     print(f"this point is isolated, and {self.is_collision(point)}")
        # assert len(adjecent_nodes)>0, f"this point is isolated"
        return adjecent_nodes
                

    
    def sample_MTSP(self, num_agent=2,num_goal=3):
        assert num_agent>0 and num_goal>0 and num_agent<=num_goal and isinstance(num_agent,int) and isinstance(num_goal,int)
        # generate random points
        points = []
        while len(points)<num_goal+num_agent:
            point=self.sample_point()
            if self.keep_distance(points,point):
                points.append(point)
        return points
    def sample_MTSP_from_graph(self, num_agent=2,num_goal=3):
        assert num_agent>0 and num_goal>0 and num_agent<=num_goal and isinstance(num_agent,int) and isinstance(num_goal,int)
        # generate random points
        points = []
        while len(points)<num_goal+num_agent:
            point=self.nodes[np.random.randint(self.num_samples)]
            if self.keep_distance(points,point):
                points.append(point)
        return points
    def sample_MTSP_from_graph_fix(self, num_agent=2,num_goal=3):
        assert num_agent>0 and num_goal>0 and num_agent<=num_goal and isinstance(num_agent,int) and isinstance(num_goal,int)
        # generate random points
        points = []
        while len(points)<num_goal+num_agent:
            point=self.nodes[self.counter]
            self.counter=(self.counter+1)%self.num_samples
            if self.keep_distance(points,point) and not self._is_point_collision(point,safety_distance=self.robot_radius*2):
                points.append(point)
        return points
    def sample_MTSP_from_graph_withcost(self, num_agent=2,num_goal=3):
        assert num_agent>0 and num_goal>0 and num_agent<=num_goal and isinstance(num_agent,int) and isinstance(num_goal,int)
        # generate random points
        points,idxs = [],[]
        while len(points)<num_goal:
            idx=np.random.randint(self.num_samples)
            point=self.nodes[idx]
            if self.keep_distance(points,point):
                points.append(point)
                idxs.append(idx)
        return points,idxs,self.cost_matrix[np.ix_(idxs, idxs)]
    def keep_distance(self,points,point):
        """
        Check if the point is far enough from all other points
        """
        for p in points:
            if np.linalg.norm(np.array(point)-np.array(p))<self.robot_radius*2:
                return False
        return True
    
    def sample_point(self):
        """
        随机采样一个环境内的点：
         - 点必须位于环境边界内
         - 点不能落在障碍物内部（若有障碍物）且和障碍物保持距离
        """
        while True:
            x = random.uniform(-self.area_size/2, self.area_size/2)
            y = random.uniform(-self.area_size/2, self.area_size/2)  
            if not self.is_collision((x, y)):
                return (x, y)
    
    def is_collision(self, obj):
        """
        Check if the robot or the path between two points collides with any obstacles or lines.
        The input `obj` can be a point (tuple) or a line segment (tuple of two points).
        check if point is in the border
        """
        if isinstance(obj, tuple) and isinstance(obj[0], (int, float)):  # Single point case
            return self._is_point_collision(obj) or not self._is_point_valid(obj)
        elif isinstance(obj, tuple) and isinstance(obj[0], tuple):  # Line segment case
            # return self._is_line_collision(obj)
            return self._is_line_collisionSTR(obj)
        else:
            print('error',obj)
        return False
    def _is_point_valid(self, point):
        # check if point is in the border
        border_poly = Polygon(self.border[0])
        point=Point(point)
        if border_poly.contains(point):
            return True
        else:
            return False
    def _is_point_collision(self, point,safety_distance=None):
        """Check if the robot at `point` collides with any obstacles or lines."""
        if safety_distance is None:
            safety_distance = self.robot_radius
        robot_shape = Point(point).buffer(safety_distance)

        # Check for collision with polygonal obstacles
        for obstacle in self.obstacles:
            if robot_shape.intersects(obstacle):
                return True

        # Check for collision with line segments
        for line in self.lines:
            line_segment = LineString(line)
            if robot_shape.intersects(line_segment):
                return True
        
        return False

    
    def _is_line_collisionSTR(self, line):
        query_line = LineString(line)
        buffer = query_line.buffer(self.robot_radius)
        # candidates_obstacles = self.obstacle_tree.query(buffer)
        # candidates_obstacles=[self.merge[i] for i in candidates_obstacles]
        # AABB粗筛（可选）
        # minx, miny, maxx, maxy = buffer.bounds
        candidates_obstacles = self.obstacle_tree.query(buffer)
        # candidates_obstacles=[self.precomputed_buffers[i] for i in candidates_obstacles]
        # candidates_obstacles=[obstacle for obstacle in candidates_obstacles if obstacle.bounds[0] <= maxx and obstacle.bounds[1] <= maxy and obstacle.bounds[2] >= minx and obstacle.bounds[3] >= miny]
        # print(candidates_obstacles)
        
        # candidates_lines=[obstacle for obstacle in candidates_lines if obstacle.bounds[0] <= maxx and obstacle.bounds[1] <= maxy and obstacle.bounds[2] >= minx and obstacle.bounds[3] >= miny]
        # print(candidates_lines)
        
        # 并行检测障碍物
        # for i in range(len(candidates_obstacles)):
        for obstacle in candidates_obstacles:
            if buffer.intersects(self.precomputed_buffers[obstacle]):
                # print("obstacle collision")
                return True
        candidates_lines = self.line_tree.query(buffer)
        # print(candidates_lines)
        # candidates_lines=[self.lines_ls[i] for i in candidates_lines]
        for line in candidates_lines:
        # for i in range(len(candidates_lines)):
            if buffer.intersects(self.lines_ls[line]):
                # print("line collision")
                return True
        return False
        
    def _is_line_collision(self, line):
        """Check if the line segment between two points collides with any obstacles or lines."""
        line_segment = LineString(line)
        line_segment_buffer = line_segment.buffer(self.robot_radius)
        # Check for collision with polygonal obstacles
        for obstacle in self.obstacles:
            if line_segment_buffer.intersects(obstacle):
                return True
        # Check for collision with line segments (walls, boundaries)
        for env_line in self.lines:
            env_line_segment = LineString(env_line)
            if line_segment_buffer.intersects(env_line_segment):
                return True
        return False

    def display_environment(self,ax=None,show=0,savefig=None):
        if ax==None:
            fig, ax = plt.subplots()
        ax.set_xlim(-self.area_size/2-0.5, self.area_size/2+0.5)
        ax.set_ylim(-self.area_size/2-0.5, self.area_size/2+0.5)
        ax.set_aspect('equal', 'box')
        # Plot obstacles
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, fc='gray', ec='black')
        for coords in self.lines:
            for i in range(len(coords) - 1):
                ax.plot([coords[i][0],coords[i+1][0]], [coords[i][1],coords[i+1][1]], color='black', linewidth=2,linestyle="-")  
        # for i, robot in enumerate(self.positions):
        #     ax.plot(robot[0], robot[1], marker=".",markersize=10, color="#1155cc")
        #     plt.annotate(str(i), (robot[0], robot[1]), xytext=(5, 5), textcoords='offset points',fontsize=16)
        plt.grid(True)
        if savefig: 
            plt.tight_layout()
            plt.savefig(savefig,bbox_inches ='tight',dpi=300)
        if show: plt.show()
        return ax

    def display_environment_MTSP(self,ax=None,show=1,num_agent=2,num_goal=3,savefig=None):
        if ax==None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.set_xlim(-self.area_size/2-0.5, self.area_size/2+0.5)
        ax.set_ylim(-self.area_size/2-0.5, self.area_size/2+0.5)
        ax.set_aspect('equal', 'box')
        # Plot obstacles
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, fc='gray', ec='black')
        for coords in self.lines:
            for i in range(len(coords) - 1):
                ax.plot([coords[i][0],coords[i+1][0]], [coords[i][1],coords[i+1][1]], color='black', linewidth=2,linestyle="-")  
        points=self.sample_MTSP(num_agent=num_agent,num_goal=num_goal)
        for i in range(num_agent):
            ax.plot(points[i][0], points[i][1], marker=".",markersize=5, color="#1155cc")
            # plt.annotate(str(i), (points[i][0], points[i][1]), xytext=(5, 5), textcoords='offset points')
        for i in range(num_agent,num_agent+num_goal):
            ax.plot(points[i][0], points[i][1], marker="*",markersize=4, color="#ff0000")
            # plt.annotate(str(i), (points[i][0], points[i][1]), xytext=(5, 5), textcoords='offset points')
        # plt.grid(True)
        if savefig: 
            plt.tight_layout()
            plt.savefig(savefig)
        if show: plt.show()
        return ax,points

class SimulationEnvironment_maze(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_maze, self).__init__()
        with open('/project/lasercar/envs/maze_edges_points.pkl', 'rb') as f:
            new_edges_list,coords_list = pickle.load(f)
        self.area_size = 20
        self.rectangle_obs=[]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+self.circle_obs
        self.lines=new_edges_list
        self.Stringline=[LineString(line) for line in self.lines]
        self.positions = coords_list
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)
class SimulationEnvironment_circleobs(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_circleobs, self).__init__()
        WallWidth=0.05
        self.area_size = 6
        self.rectangle_obs=[]
        self.circle_obs=[]#list of tuple (x1,y1,r1) representing the center and radius of the circle
        self.obstacles = self.rectangle_obs+[Point(circ[0],circ[0]).buffer(circ[2],quad_segs=quad_segs) for circ in self.circle_obs]
        self.border = [[(-3.0,3.0),(-3.0,-3.0),(3.0,-3.0),(3.0,3.0),(-3.0,3.0)]]
        self.inner_lines = []
        self.lines=self.border+self.inner_lines
        self.Stringline=[LineString(line) for line in self.lines]
        offset=-0.8
        self.positions = [(-2.5, 2.5),(2.5, -2.5),(2.5, 2.5),(-2.5, -2.5),\
            #  (-2.5-offset, 2.5),(2.5+offset, -2.5),(2.5+offset, 2.5),(-2.5-offset, -2.5), \
            #(-2.5, 2.5+offset),(2.5, -2.5-offset),(2.5, 2.5+offset),(-2.5, -2.5-offset), (-2.5-offset, 2.5+offset),(2.5+offset, -2.5-offset),(2.5+offset, 2.5+offset),(-2.5-offset, -2.5-offset)
            ]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)
class SimulationEnvironment_Unmaze(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_Unmaze, self).__init__()
        WallWidth=0.05
        self.area_size = 6
        self.rectangle_obs=[]
        self.circle_obs=[]#list of tuple (x1,y1,r1) representing the center and radius of the circle
        self.obstacles = self.rectangle_obs+[Point(circ[0],circ[0]).buffer(circ[2],quad_segs=quad_segs) for circ in self.circle_obs]
        self.border = [[(-3.0,3.0),(-3.0,-3.0),(3.0,-3.0),(3.0,3.0),(-3.0,3.0)]]
        self.inner_lines = []
        self.lines=self.border+self.inner_lines
        self.Stringline=[LineString(line) for line in self.lines]
        offset=-0.8
        self.positions = [(-2.5, 2.5),(2.5, -2.5),(2.5, 2.5),(-2.5, -2.5),\
            #  (-2.5-offset, 2.5),(2.5+offset, -2.5),(2.5+offset, 2.5),(-2.5-offset, -2.5), \
            #(-2.5, 2.5+offset),(2.5, -2.5-offset),(2.5, 2.5+offset),(-2.5, -2.5-offset), (-2.5-offset, 2.5+offset),(2.5+offset, -2.5-offset),(2.5+offset, 2.5+offset),(-2.5-offset, -2.5-offset)
            ]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)
class SimulationEnvironment_CA(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_CA, self).__init__()
        WallWidth=0.05
        wall_len=3.65
        self.area_size = 9
        self.rectangle_obs=[]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+[Point(circ[0],circ[0]).buffer(circ[2],quad_segs=quad_segs) for circ in self.circle_obs]
        offset=0.8
        self.positions = [(-3.5, 3.5),(3.5, -3.5),(3.5, 3.5),(-3.5, -3.5),\
            # (-2.5-offset, 2.5),(2.5+offset, -2.5),(2.5+offset, 2.5),(-2.5-offset, -2.5),\
            # (-2.5, 2.5+offset),(2.5, -2.5-offset),(2.5, 2.5+offset),(-2.5, -2.5-offset),\
            # (-2.5-offset, 2.5+offset),(2.5+offset, -2.5-offset),(2.5+offset, 2.5+offset),(-2.5-offset, -2.5-offset)
            ]
        self.border = [[(-4.5, 4.5), (-4.5, -4.5), (4.5, -4.5), (4.5, 4.5),(-4.5, 4.5)]]
        self.inner_lines = [[(4.5-wall_len, 1.5), (4.5,1.5)],[(1.5,-(4.5-wall_len)), (1.5,-4.5)],[(-(4.5-wall_len), -1.5), (-4.5,-1.5)],[(-1.5,(4.5-wall_len)), (-1.5,4.5)]]
        self.lines=self.border+self.inner_lines
        self.Stringline=[LineString(line) for line in self.lines]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)


class SimulationEnvironment_CubeCollection(SimulationEnvironment):
    def __init__(self,idx=None):
        super(SimulationEnvironment_CubeCollection, self).__init__()
        WallWidth = 0.05
        wall_len = 3.65
        WIDTH=8.5
        #22
        # sub_scenario=[[1, 5, 8, 10, 11, 16], [0, 2, 7, 8, 15, 16, 18, 20, 23], [0, 1, 2, 5, 10, 11, 13, 15, 21], [0, 2, 3, 5, 7, 9, 10, 13, 18, 19, 20, 23], [3, 4, 6, 8, 9, 10, 12, 14, 15, 18, 21, 22], [0, 2, 5, 6, 8, 11, 13, 17, 18, 19, 20, 22], [0, 1, 2, 4, 5, 7, 10, 11, 12, 13, 14, 15, 17, 20, 21], [0, 1, 2, 3, 5, 6, 7, 10, 11, 15, 16, 17, 20, 21, 22], [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 19, 21], [1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 17, 19, 20, 22], [0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 14, 17, 18, 19, 20, 21, 22, 23], [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23], [0, 1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 21, 23], [0, 1, 2, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23], [0, 1, 2, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 23], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23], [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23], [0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [0, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
        #6
        sub_scenario=[[6], [23,2], [5, 10, 14], [13, 19, 4, 1, 8, 18], [9, 20, 12, 17, 11, 21, 22, 7, 3, 15, 16, 0], [9, 20, 12, 17, 11, 21, 22, 7, 3, 15, 16, 0, 13, 19, 4, 1, 8, 18, 5, 10, 14, 2, 23, 6]]
        self.area_size = 17
        self.circle_obs=[(-2.897500,7.884750,0.5),
            (-7.15,3.6,0.5),
            (-3.39,3.46,0.5),
            (0,4,0.5),
            (2.7,3.39,0.5),
            (4.9,6.6,0.5),
            (-4,1.1,0.5),
            (-1.23,1.87,0.5),
            (1.63,1,0.5),
            (-1,-1.88,0.5),
            (2.2,-2,0.5),
            (-5.49,-3.86,0.5),
            (5.11,-5.83,0.5),
            (-2.23,-7,0.5),]
        self.rectangle_obs=[
            gen_poly([-5.27, 5.89], 0.53, 1.86, -0.53/np.pi*180),
            gen_poly([-6.9, 0.45], 0.66, 1.95, -1.09/np.pi*180),
            gen_poly([-4.5, -1.68], 0.5, 1.61, 0.689/np.pi*180),
            gen_poly([-5.1, -5.96], 0.67, 1.78, -1.53/np.pi*180),
            gen_poly([-1.4, -4.62],  0.71, 1.56, -0.32/np.pi*180),
            gen_poly([1.97, -5.24], 0.34, 1.96, 0.258/np.pi*180),
            gen_poly([6.62, -2.7], 0.46, 1.83, -0.67/np.pi*180),
            gen_poly([5.12, 0.09], 0.54, 1.77, 0.81/np.pi*180),
            gen_poly([5.3, 3.29], 0.62, 2.51, 0.41/np.pi*180),
            gen_poly([1.36, 6.59], 0.61, 1.61, 0.869/np.pi*180),
        ]
        self.obstacles = self.rectangle_obs+[Point(circ[0],circ[1]).buffer(circ[2],quad_segs=quad_segs) for circ in self.circle_obs]
        self.positions = [(1., -5.),(-1., 5.),(-3., 5.),(3., -5.),(-3., 0.),(3., 0.),(-3., -5.),(3., 5.),\
            # (-7,-7),(7,7),  (0,-7),(0,7),  (7,-7),(-7,7), (6.7,-1.5),(-6.7,1.5)
            ]
        self.fixed_eval_task = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7), (0, 1, 5, 6), (0, 1, 5, 7), (0, 1, 6, 7), (0, 2, 3, 4), (0, 2, 3, 5), (0, 2, 3, 6), (0, 2, 3, 7), (0, 2, 4, 5), (0, 2, 4, 6), (0, 2, 4, 7), (0, 2, 5, 6), (0, 2, 5, 7), (0, 2, 6, 7), (0, 3, 4, 5), (0, 3, 4, 6), (0, 3, 4, 7), (0, 3, 5, 6), (0, 3, 5, 7), (0, 3, 6, 7), (0, 4, 5, 6), (0, 4, 5, 7), (0, 4, 6, 7), (0, 5, 6, 7), (1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 3, 6), (1, 2, 3, 7), (1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 4, 7), (1, 2, 5, 6), (1, 2, 5, 7), (1, 2, 6, 7), (1, 3, 4, 5), (1, 3, 4, 6), (1, 3, 4, 7), (1, 3, 5, 6), (1, 3, 5, 7), (1, 3, 6, 7), (1, 4, 5, 6), (1, 4, 5, 7), (1, 4, 6, 7), (1, 5, 6, 7), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 4, 7), (2, 3, 5, 6), (2, 3, 5, 7), (2, 3, 6, 7), (2, 4, 5, 6), (2, 4, 5, 7), (2, 4, 6, 7), (2, 5, 6, 7), (3, 4, 5, 6), (3, 4, 5, 7), (3, 4, 6, 7), (3, 5, 6, 7), (4, 5, 6, 7)]

        if idx!=None:
            self.obstacles_all=self.obstacles
            self.obstacles=[]
            obs_lst=sub_scenario[idx]
            for obs_i in obs_lst:
                self.obstacles.append(self.obstacles_all[obs_i])
        self.border = [[(-WIDTH, WIDTH),(-WIDTH, -WIDTH),(WIDTH, -WIDTH),(WIDTH, WIDTH),(-WIDTH, WIDTH)]]
        self.inner_lines = []
        self.lines=self.border+self.inner_lines
        self.Stringline=[LineString(line) for line in self.lines]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)

# Instantiate and display the environment
# env = SimulationEnvironment_CubeCollection()
# env.display_environment()

class SimulationEnvironment_Warehouse(SimulationEnvironment):
    def __init__(self,idx=None):
        super(SimulationEnvironment_Warehouse, self).__init__()
        WallWidth = 0.15
        wall_len = 3.65
        WIDTH=10.0
        # 46
        # sub_scenario=[[2, 3, 12, 16, 23, 28], [0, 2, 3, 8, 17, 23, 30, 31, 32], [1, 6, 10, 14, 19, 23, 26, 28, 29], [0, 1, 6, 8, 13, 16, 21, 22, 23, 24, 29, 32], [1, 3, 6, 9, 18, 19, 22, 23, 29, 31, 32, 33], [0, 3, 4, 5, 7, 8, 14, 19, 20, 25, 30, 33], [0, 2, 3, 5, 6, 10, 11, 14, 15, 20, 27, 29, 30, 32, 33], [1, 4, 5, 7, 10, 12, 14, 19, 21, 22, 24, 26, 28, 29, 33], [4, 5, 6, 7, 10, 12, 13, 19, 20, 23, 25, 26, 28, 30, 33], [2, 3, 6, 7, 8, 9, 13, 16, 20, 22, 23, 25, 26, 28, 33], [0, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 23, 25, 27, 28, 31, 32, 33], [0, 1, 3, 5, 6, 10, 11, 14, 17, 21, 23, 24, 27, 28, 29, 30, 32, 33], [6, 7, 9, 10, 12, 15, 16, 17, 18, 20, 22, 23, 24, 26, 27, 28, 30, 32], [4, 5, 7, 9, 11, 12, 13, 14, 17, 19, 20, 21, 22, 23, 24, 27, 31, 33], [4, 5, 6, 7, 8, 10, 11, 15, 16, 17, 18, 19, 21, 23, 29, 31, 32, 33], [1, 2, 3, 5, 7, 8, 10, 11, 13, 15, 16, 18, 21, 23, 24, 26, 27, 28, 29, 30, 32], [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 20, 21, 25, 26, 29, 30, 32, 33], [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 32, 33], [0, 2, 3, 4, 6, 10, 12, 13, 14, 16, 17, 18, 19, 22, 23, 24, 25, 27, 28, 29, 30], [1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 17, 21, 24, 25, 26, 27, 28, 30, 31], [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 22, 23, 24, 26, 28, 31, 33], [2, 3, 5, 6, 7, 9, 10, 11, 12, 15, 16, 18, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33], [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 20, 23, 25, 26, 27, 28, 29, 31, 33], [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 19, 23, 25, 26, 27, 29, 30, 31, 32, 33], [0, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 25, 26, 27, 30, 32, 33], [0, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33], [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 24, 25, 28, 29, 30, 31, 33], [1, 2, 3, 6, 7, 11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33], [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33], [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 29, 30, 31, 32, 33], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 27, 28, 29, 30, 31, 32], [0, 1, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33], [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31], [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 30, 31, 32], [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 28, 31, 33], [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33], [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33], [0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33], [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33], [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 30, 31, 32, 33], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]]        
        # 5 #[31], [3], [7],
        sub_scenario=[ [17, 15], [5, 6, 16, 25], [9, 1, 22, 13, 18, 33, 32, 27], [20, 24, 2, 8, 29, 0, 4, 10, 12, 26, 28, 21, 19, 23, 11, 30, 14], [20, 24, 2, 8, 29, 0, 4, 10, 12, 26, 28, 21, 19, 23, 11, 30, 14, 9, 1, 22, 13, 18, 33, 32, 27, 5, 6, 16, 25, 17, 15, 7, 3, 31]]
        self.area_size = 20
        self.rectangle_obs=[
            gen_poly([7, -8.9], 1, 2.1, 0),gen_poly([3.5, -8.9], 1, 2.1, 0),gen_poly([0, -8.9], 1, 2.1, 0),gen_poly([-3.5, -8.9], 1, 2.1, 0),
            gen_poly([-7, -8.9], 1, 2.1, 0),gen_poly([7, 8.9], 1, 2.1, 0),gen_poly([3.5, 8.9], 1, 2.1, 0),gen_poly([0, 8.9], 1, 2.1, 0),
            gen_poly([-3.5, 8.9], 1, 2.1, 0),gen_poly([-7, 8.9], 1, 2.1, 0),gen_poly([7, 4.5], 1, 2.1, 0),gen_poly([7, 2.25], 1, 2.1, 0),
            gen_poly([3.5, 4.5], 1, 2.1, 0),gen_poly([3.5, 2.25], 1, 2.1, 0),gen_poly([-7, 4.5], 1, 2.1, 0), gen_poly([-7, 2.25], 1, 2.1, 0),
            gen_poly([-3.5, 4.5], 1, 2.1, 0),gen_poly([-3.5, 2.25], 1, 2.1, 0),gen_poly([7, -4.5], 1, 2.1, 0),gen_poly([7, -2.25], 1, 2.1, 0),
            gen_poly([3.5, -4.5], 1, 2.1, 0),gen_poly([3.5, -2.25], 1, 2.1, 0), gen_poly([-7, -4.5], 1, 2.1, 0),gen_poly([-7, -2.25], 1, 2.1, 0),
            gen_poly([-3.5, -4.5], 1, 2.1, 0),gen_poly([-3.5, -2.25], 1, 2.1, 0),                    
        ]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+[Point(circ[0],circ[0]).buffer(circ[2],quad_segs=quad_segs) for circ in self.circle_obs]
        self.inner_lines=[[(-1.8, 3.45+2.5),(-1.8, 3.45-2.5)],[(1.8, 3.45+2.5),(1.8, 3.45-2.5)],[(-1.8, -3.45+2.5),(-1.8, -3.45-2.5)],\
                [(1.8, -3.45+2.5),(1.8, -3.45-2.5)],[(4.65-5.6/2, 3.4),(4.65+5.6/2, 3.4)],[(-4.65-5.6/2, 3.4),(-4.65+5.6/2, 3.4)],\
                [(4.65-5.6/2, -3.4),(4.65+5.6/2, -3.4)],[(-4.65-5.6/2, -3.4),(-4.65+5.6/2, -3.4)]]
        if idx!=None:
            self.obstacles_all=self.obstacles
            self.obstacles=[]
            innerlines_all=self.inner_lines
            self.inner_lines=[]
            obs_lst=sub_scenario[idx]
            for obs_i in obs_lst:
                if obs_i<len(self.obstacles_all):
                    self.obstacles.append(self.obstacles_all[obs_i])
                else:
                    self.inner_lines.append(innerlines_all[obs_i-len(self.obstacles_all)])

        self.positions = [
           (8.5, -7),(-8.5, 7),(-8.5, -7),(8.5, 7),(8.5, 0),(-8.5, 0),(0, 7),(0, -7),\
            # (-8.5,3.65),(8.5,-3.65),(-8.5,-3.65),(8.5,3.65),(-4.64,7),(4.64,-7),(-4.64,-7),(4.64,7)
        ]
        self.fixed_eval_task = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7), (0, 1, 5, 6), (0, 1, 5, 7), (0, 1, 6, 7), (0, 2, 3, 4), (0, 2, 3, 5), (0, 2, 3, 6), (0, 2, 3, 7), (0, 2, 4, 5), (0, 2, 4, 6), (0, 2, 4, 7), (0, 2, 5, 6), (0, 2, 5, 7), (0, 2, 6, 7), (0, 3, 4, 5), (0, 3, 4, 6), (0, 3, 4, 7), (0, 3, 5, 6), (0, 3, 5, 7), (0, 3, 6, 7), (0, 4, 5, 6), (0, 4, 5, 7), (0, 4, 6, 7), (0, 5, 6, 7), (1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 3, 6), (1, 2, 3, 7), (1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 4, 7), (1, 2, 5, 6), (1, 2, 5, 7), (1, 2, 6, 7), (1, 3, 4, 5), (1, 3, 4, 6), (1, 3, 4, 7), (1, 3, 5, 6), (1, 3, 5, 7), (1, 3, 6, 7), (1, 4, 5, 6), (1, 4, 5, 7), (1, 4, 6, 7), (1, 5, 6, 7), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 4, 7), (2, 3, 5, 6), (2, 3, 5, 7), (2, 3, 6, 7), (2, 4, 5, 6), (2, 4, 5, 7), (2, 4, 6, 7), (2, 5, 6, 7), (3, 4, 5, 6), (3, 4, 5, 7), (3, 4, 6, 7), (3, 5, 6, 7), (4, 5, 6, 7)]

        self.border = [[(-WIDTH, WIDTH),(-WIDTH, -WIDTH),(WIDTH, -WIDTH),(WIDTH, WIDTH),(-WIDTH, WIDTH)]]
        # self.inner_lines = []
        self.lines=self.border+self.inner_lines
        self.Stringline=[LineString(line) for line in self.lines]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)


class SimulationEnvironment_Scenario1(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_Scenario1, self).__init__()
        self.area_size = 15
        self.rectangle_obs=[]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+self.circle_obs

        # self.scenario_lines=[[(18, 158), (158, 158), (158, 18), (238, 22), (238, 161), (378, 162), (377, 238), (238, 238), (237, 378), (158, 381), (158, 241), (18, 241),(18, 158)]]
        # self.positions = [(-7.00,11.50),(-18.00,9.50),(-7.00,9.50),(-18.00,11.50),(-12.50,17.00),(-12.50,4.00)]
        # self.lines=[[ ((coords[1]-500)/25,-(coords[0]-400)/20) for coords in line] for line in self.scenario_lines]
        # self.lines=[[(coor[0]+13,coor[1]-11) for coor in line] for line in self.lines]
        # self.positions=[(coor[0]+13,coor[1]-11) for coor in self.positions]
        
        Length=15
        Width=1
        Distance=1
        self.border=[[(Width/2,-Length/2),(Width/2,-Width/2),(Length/2,-Width/2),(Length/2,Width/2),(Width/2,Width/2),(Width/2,Length/2),(-Width/2,Length/2),(-Width/2,Width/2),(-Length/2,Width/2),(-Length/2,-Width/2),(-Width/2,-Width/2),(-Width/2,-Length/2),(Width/2,-Length/2)]]
        self.inner_lines=[]
        self.lines=self.border+self.inner_lines
        self.positions=[(0,Length/2-Distance),(0,-Length/2+Distance),(Length/2-Distance,0),(-Length/2+Distance,0)]
        self.fixed_eval_task = [(0, 1, 2, 3)]
        self.Stringline=[LineString(line) for line in self.lines]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)
class SimulationEnvironment_Scenario2(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_Scenario2, self).__init__()
        self.area_size = 16
        self.rectangle_obs=[]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+self.circle_obs
        # self.positions=[]
        # self.scenario_lines=[[(138, 501), (138, 422), (58, 418),(58, 661),(137, 658),(138, 578),(298, 582),(298, 661),(377, 658),(378, 422),(298, 418), (297, 498),(138, 501)]]#  
        # positions = [(-2.00,16.00),(0.00,16.00),(3.00,16.00),(5.00,16.00)]
        # self.lines=[[ ((coords[1]-500)/25,-(coords[0]-400)/20) for coords in line] for line in self.scenario_lines]
        # # centralize
        # self.lines=[[(coor[0]-1.5,coor[1]-9) for coor in line] for line in self.lines]
        # positions=[(coor[0]-1.5,coor[1]-9) for coor in positions]
        # for po in positions:
        #     self.positions.append(po)
        #     self.positions.append((po[0],-po[1]))

        Length,Height=15,8
        Width1=2
        Width2=0.6
        self.border=[[(Length/2,-Height/2-Width1),(Length/2,-Height/2),(Width2/2,-Height/2),(Width2/2,Height/2),(Length/2,Height/2),(Length/2,Height/2+Width1),(-Length/2,Height/2+Width1),(-Length/2,Height/2),(-Width2/2,Height/2),(-Width2/2,-Height/2),(-Length/2,-Height/2),(-Length/2,-Height/2-Width1),(Length/2,-Height/2-Width1)]]
        self.inner_lines=[]
        self.lines=self.border+self.inner_lines
        self.positions=[(-6,-Height/2-Width1/2),(-6,Height/2+Width1/2),(-2,-Height/2-Width1/2),(-2,Height/2+Width1/2),\
            (2,-Height/2-Width1/2),(2,Height/2+Width1/2),(6,-Height/2-Width1/2),(6,Height/2+Width1/2)]#[(0,x) for x in np.linspace(-Height/2-Width1/2,Height/2+Width1/2,7)]
        self.fixed_eval_task = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7), (0, 1, 5, 6), (0, 1, 5, 7), (0, 1, 6, 7), (0, 2, 3, 4), (0, 2, 3, 5), (0, 2, 3, 6), (0, 2, 3, 7), (0, 2, 4, 5), (0, 2, 4, 6), (0, 2, 4, 7), (0, 2, 5, 6), (0, 2, 5, 7), (0, 2, 6, 7), (0, 3, 4, 5), (0, 3, 4, 6), (0, 3, 4, 7), (0, 3, 5, 6), (0, 3, 5, 7), (0, 3, 6, 7), (0, 4, 5, 6), (0, 4, 5, 7), (0, 4, 6, 7), (0, 5, 6, 7), (1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 3, 6), (1, 2, 3, 7), (1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 4, 7), (1, 2, 5, 6), (1, 2, 5, 7), (1, 2, 6, 7), (1, 3, 4, 5), (1, 3, 4, 6), (1, 3, 4, 7), (1, 3, 5, 6), (1, 3, 5, 7), (1, 3, 6, 7), (1, 4, 5, 6), (1, 4, 5, 7), (1, 4, 6, 7), (1, 5, 6, 7), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 4, 7), (2, 3, 5, 6), (2, 3, 5, 7), (2, 3, 6, 7), (2, 4, 5, 6), (2, 4, 5, 7), (2, 4, 6, 7), (2, 5, 6, 7), (3, 4, 5, 6), (3, 4, 5, 7), (3, 4, 6, 7), (3, 5, 6, 7), (4, 5, 6, 7)]
        

        self.Stringline=[LineString(line) for line in self.lines]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)
def checkintersection(line,env):
    Line=LineString(line)
    for obs in env.obstacles:
        if obs.intersects(Line):
            return True
    for line1 in env.Stringline:
        if line1.intersects(Line):
            return True
    return False
class SimulationEnvironment_Scenario3(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_Scenario3, self).__init__()
        self.area_size = 15.0
        self.rectangle_obs=[]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+self.circle_obs

        # self.scenario_lines=[[(78, 698),(378,702),(377, 978),(78, 978)],[(258,702),(258, 821)],[(257, 978),(258, 858)]]
        # self.positions = [(10.00,4.00),(12.00,4.00),(14.00,4.00),(16.00,4.00),(18.00,4.00)]
        # self.lines=[[ ((coords[1]-500)/25,-(coords[0]-400)/20) for coords in line] for line in self.scenario_lines]
        # # centralize
        # self.lines=[[(coor[0]-13,coor[1]-9) for coor in line] for line in self.lines]
        # positions=[(coor[0]-13,coor[1]-9) for coor in self.positions]
        # self.positions=[]
        # for po in positions:
        #     self.positions.append(po)
        #     self.positions.append((po[0],-po[1]))
        L,W1,W2=self.area_size,2.0,1.0
        self.border=[[(-L/2,L/2),(-L/2,-L/2),(L/2,-L/2),(L/2,L/2),(-L/2,L/2)]]
        self.inner_lines=[[(-L/2,0.0),(-W2/2,0.0)],[(W2/2,0.0),(L/2,0.0)]]
        self.lines=self.border+self.inner_lines
        # self.lines=[[(-L/2,0),(-W2/2,0)],[(W2/2,0),(L/2,0)]]
        positions_num=4
        for i in range(positions_num):
            self.positions.append((L*(i+1)/(positions_num+1)-L/2,-L/2+W1/2)) 
            self.positions.append((L*(i+1)/(positions_num+1)-L/2,L/2-W1/2))

        self.Stringline=[LineString(line) for line in self.lines]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)

class SimulationEnvironment_Scenario4(SimulationEnvironment):
    def __init__(self,idx=0):
        super(SimulationEnvironment_Scenario4, self).__init__()
        self.area_size = 20
        WIDTH=10.0
        self.rectangle_obs=[]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+self.circle_obs
        self.border=[[(-WIDTH, WIDTH),(-WIDTH, -WIDTH),(WIDTH, -WIDTH),(WIDTH, WIDTH),(-WIDTH, WIDTH)]]
        self.inner_lines=[]
        self.lines=self.border+self.inner_lines
        self.Stringline=[LineString(line) for line in self.lines]
        radius=np.linspace(0.16,0.95,5)[idx] #from minimus 0.15376 to 0.9, 0.95
        num=8
        for theta in np.linspace(0,np.pi,num+1)[:-1]:
            self.positions.append((radius*WIDTH*np.cos(theta),radius*WIDTH*np.sin(theta)))
            self.positions.append((-radius*WIDTH*np.cos(theta),-radius*WIDTH*np.sin(theta)))    
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)

class SimulationEnvironment_Scenario5(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_Scenario5, self).__init__()
        self.area_size = 12
        self.rectangle_obs=[]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+self.circle_obs
        self.scenario_lines=[[(418, 398), (418, 681), (477, 678), (478, 398),(418, 398)]]
        self.positions = [(-2.5,-2.5),(5.5,-2.5),(3.5,-2.5),(-0.5,-2.5)]
        self.lines=[[ ((coords[1]-500)/25,-(coords[0]-400)/20) for coords in line] for line in self.scenario_lines]
        # centralize
        self.border=[[(coor[0]-2,coor[1]+2) for coor in line] for line in self.lines]
        self.inner_lines=[]
        self.lines=self.border+self.inner_lines
        self.Stringline=[LineString(line) for line in self.lines]
        self.positions=[(coor[0]-2,coor[1]+2) for coor in self.positions]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)

class SimulationEnvironment_Scenario6(SimulationEnvironment):
    def __init__(self):
        super(SimulationEnvironment_Scenario6, self).__init__()
        self.area_size = 15
        self.rectangle_obs=[]
        self.circle_obs=[]
        self.obstacles = self.rectangle_obs+self.circle_obs
        # self.scenario_lines=[[(477, 678), (478, 398), (780, 402),(780, 678),(477, 678)],[(518, 438), (518, 461)],[(518, 638), (518, 661)],[(558, 402), (558, 421)],[(558, 598), (558, 621)],\
        # [(578, 438), (578, 461)], [(578, 558), (578, 581)], [(598, 498), (598, 541)], [(638, 438), (638, 461)], [(638, 578), (638, 621)], [(678, 518), (678, 541)], [(698, 438), (698, 461)],\
        # [ (698, 578), (698, 601)]]
        # self.positions = [(-2.5,-18.5),(-0.5,-18.5),(1.5,-18.5),(3.5,-18.5),(5.5,-18.5)]
        # self.lines=[[ ((coords[1]-500)/25,-(coords[0]-400)/20) for coords in line] for line in self.scenario_lines]
        # # centralize
        # self.lines=[[(coor[0]-1,coor[1]+11.5) for coor in line] for line in self.lines]
        # positions=[(coor[0]-1,coor[1]+11.5) for coor in self.positions]
        # self.positions=[]
        # for po in positions:
        #     self.positions.append(po)
        #     self.positions.append((po[0],-po[1]))
        
        L, Wall_num,short_num,W2=self.area_size,7,8,1.8
        W1,short_L=(L-(Wall_num-1)*W2)/2,L/short_num
        positions_num=4
        self.border=[[(-L/2,L/2),(-L/2,-L/2),(L/2,-L/2),(L/2,L/2),(-L/2,L/2)]]
        self.inner_lines=[]
        
        
        for j in range(Wall_num):
            for i in range(short_num//2):
                self.inner_lines.append([(-L/2+2*i*short_L+(j%2)*short_L,-L/2+W1+W2*j),(-L/2+(2*i+1)*short_L+(j%2)*short_L,-L/2+W1+W2*j)])
        self.lines=self.border+self.inner_lines
        self.positions=[]
        for i in range(positions_num):
            self.positions.append((L*(i+1)/(positions_num+1)-L/2,-L/2+W1/2)) 
            self.positions.append((L*(i+1)/(positions_num+1)-L/2,L/2-W1/2))
        self.fixed_eval_task = [(0, 1, 2, 3)]
        self.Stringline=[LineString(line) for line in self.lines]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)


class SimulationEnvironment_Scenario7(SimulationEnvironment):
    def __init__(self,idx=None,num_obstacles=15,obstacle_size = 0.6,irregularity_factor=0):
        super(SimulationEnvironment_Scenario7, self).__init__()
        self.area_size = 20 # Define the area size (e.g., a 20x20 meter area)
        WIDTH=10.0
        num_points = 8  # Number of points to generate
        obstacle_size = obstacle_size  # Size of each obstacle (e.g., 2 meters square), max radius
        min_distance = 0.6  # Minimum distance between points and obstacles (0.6 meters)
        num_vertices=8

        for x in np.linspace(-9,9,num_points):
            self.positions.append((x,9.5))
            self.positions.append((x,-9.5))
        # Generate obstacles and points
        # with open('test.pkl','rb') as f:
        #     self.obstacles=pickle.load(f)
        if idx!=None:
            # Load obstacles 
            with open('scenario7_25.pkl','rb') as f:
                self.obstacles=pickle.load(f)
                self.obstacles=self.obstacles[idx]
        else:
            self.obstacles, points = generate_obstacles_and_points(self.positions,self.area_size, num_obstacles, num_points, obstacle_size, num_vertices,irregularity_factor,min_distance)
        # print(points)
        # self.positions = [(p.x,p.y) for p in points]
        self.border=[[(-WIDTH, WIDTH),(-WIDTH, -WIDTH),(WIDTH, -WIDTH),(WIDTH, WIDTH),(-WIDTH, WIDTH)]]
        self.inner_lines=[]
        self.lines=self.border+self.inner_lines
        self.Stringline=[LineString(line) for line in self.lines]
        self.Env_edges_list=[]
        for obstacle in self.obstacles:
            self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
        for line in self.lines:
            self.Env_edges_list+=get_edges(line)
        # with open('test.pkl','wb') as f:
        #     pickle.dump(self.obstacles,f)
# class SimulationEnvironment_Scenario7_fix(SimulationEnvironment):
#     def __init__(self,idx):
#         super(SimulationEnvironment_Scenario7_fix, self).__init__()
#         self.area_size = 20 # Define the area size (e.g., a 20x20 meter area)
#         WIDTH=10.0
#         num_points = 8  # Number of points to generate
#         for x in np.linspace(-9,9,num_points):
#             self.positions.append((x,9.5))
#             self.positions.append((x,-9.5))
#         # Load obstacles 
#         with open('scenario7_25.pkl','rb') as f:
#             self.obstacles=pickle.load(f)
#             self.obstacles=self.obstacles[idx]

#         self.lines=[[(-WIDTH, WIDTH),(-WIDTH, -WIDTH),(WIDTH, -WIDTH),(WIDTH, WIDTH),(-WIDTH, WIDTH)]]

#         self.Stringline=[LineString(line) for line in self.lines]
#         self.Env_edges_list=[]
#         for obstacle in self.obstacles:
#             self.Env_edges_list+=get_edges(list(obstacle.exterior.coords))
#         for line in self.lines:
#             self.Env_edges_list+=get_edges(line)
 
def gen_fixed_scenario7():
    all_obstacles=[]
    for obstacle_size in np.linspace(1,3.5,5):
        for irregularity_factor in np.linspace(0,1,5):
            env = SimulationEnvironment_Scenario7(num_obstacles=10,obstacle_size=obstacle_size,irregularity_factor=irregularity_factor)
            # env.display_environment()
            all_obstacles.append(env.obstacles)
    # with open('scenario7_25.pkl','wb') as f:
    #     pickle.dump(all_obstacles,f)    

import numpy as np
import random

class RRTStar:
    def __init__(self, start, goal, env, max_iter=50000, step_size=0.7, radius=20):
        self.start = start
        self.goal = goal
        self.env = env
        self.max_iter = max_iter
        self.step_size = step_size
        self.radius = radius
        self.tree = {start: None}
        self.cost = {start: 0}
        self.path = None

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def sample_point(self):
        """Randomly sample a point in the environment."""
        return (random.uniform(-self.env.area_size / 2, self.env.area_size / 2),
                random.uniform(-self.env.area_size / 2, self.env.area_size / 2))

    def nearest(self, point):
        """Find the nearest node in the tree."""
        return min(self.tree.keys(), key=lambda n: self.distance(n, point))

    def steer(self, from_node, to_node):
        """Move from from_node towards to_node by step_size."""
        direction = np.array(to_node) - np.array(from_node)
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            return to_node
        direction = direction / distance
        new_node = tuple(np.array(from_node) + direction * self.step_size)
        return new_node

    def is_path_valid(self, from_node, to_node):
        """Check if the path between from_node and to_node is collision-free."""
        # Check if the line segment between from_node and to_node intersects with any obstacle
        return not self.env.is_collision((from_node, to_node))

    def find_nearby(self, point):
        """Find all nodes within the given radius from the point."""
        return [n for n in self.tree.keys() if self.distance(n, point) < self.radius]

    def rewire(self, new_node):
        """Rewire nearby nodes to ensure the path is optimal."""
        nearby_nodes = self.find_nearby(new_node)
        for node in nearby_nodes:
            # We need to check if the path between new_node and node is valid
            if self.is_path_valid(new_node, node):
                new_cost = self.cost[new_node] + self.distance(new_node, node)
                if new_cost < self.cost[node]:
                    self.tree[node] = new_node
                    self.cost[node] = new_cost

    def build_rrt_star(self):
        """Build RRT*."""
        for i in range(self.max_iter):
            random_point = self.sample_point()
            nearest_node = self.nearest(random_point)
            new_node = self.steer(nearest_node, random_point)

            # Check if the new node is valid and the path to it is collision-free
            if not self.env.is_collision(new_node) and self.is_path_valid(nearest_node, new_node):
                # Add the new node to the tree
                self.tree[new_node] = nearest_node
                self.cost[new_node] = self.cost[nearest_node] + self.distance(nearest_node, new_node)
                
                # Rewire nearby nodes
                self.rewire(new_node)

                # Check if goal is reached
                if self.distance(new_node, self.goal) < self.step_size:
                    # Make sure the path to the goal is valid
                    if self.is_path_valid(new_node, self.goal):
                        self.tree[self.goal] = new_node
                        self.cost[self.goal] = self.cost[new_node] + self.distance(new_node, self.goal)
                        self.path = self.reconstruct_path()
                        return self.path
            #         else:
            #             print("Goal reached but path to goal is not valid.")
            # else:
            #     print("New node is in collision or path to it is not valid.")
            # Print progress every 1000 iterations
            if i % 100 == 0:
                print(f"RRT* iteration: {i}/{self.max_iter}")
                
        print("Failed to find a path within the maximum number of iterations.")
        return None

    def reconstruct_path(self):
        """Reconstruct the optimal path from start to goal."""
        path = [self.goal]
        current = self.goal
        while current in self.tree and self.tree[current] is not None:
            current = self.tree[current]
            path.append(current)
        return path[::-1]

    def informed_rrt_star(self):
        """Build Informed RRT*."""
        c_best = float('inf')
        c_min = self.distance(self.start, self.goal)
        x_center = np.array([(self.start[0] + self.goal[0]) / 2.0, (self.start[1] + self.goal[1]) / 2.0])
        a1 = np.array([(self.goal[0] - self.start[0]) / c_min, (self.goal[1] - self.start[1]) / c_min])
        etheta = np.arctan2(a1[1], a1[0])
        rotation_matrix = np.array([[np.cos(etheta), -np.sin(etheta)], [np.sin(etheta), np.cos(etheta)]])
        
        for i in range(self.max_iter):
            if self.path:
                c_best = self.cost[self.goal]
                l = np.diag([c_best / 2.0, np.sqrt(c_best**2 - c_min**2) / 2.0])
                random_point = self.sample_informed(x_center, l, rotation_matrix)
            else:
                random_point = self.sample_point()
            
            nearest_node = self.nearest(random_point)
            new_node = self.steer(nearest_node, random_point)

            # Check if the new node is valid and the path to it is collision-free
            if not self.env.is_collision(new_node) and self.is_path_valid(nearest_node, new_node):
                # Add the new node to the tree
                self.tree[new_node] = nearest_node
                self.cost[new_node] = self.cost[nearest_node] + self.distance(nearest_node, new_node)
                
                # Rewire nearby nodes
                self.rewire(new_node)

                # Check if goal is reached
                if self.distance(new_node, self.goal) < self.step_size:
                    # Make sure the path to the goal is valid
                    if self.is_path_valid(new_node, self.goal):
                        self.tree[self.goal] = new_node
                        self.cost[self.goal] = self.cost[new_node] + self.distance(new_node, self.goal)
                        self.path = self.reconstruct_path()
                        return self.path
            
            # Print progress every 1000 iterations
            if i % 100 == 0:
                print(f"Informed RRT* iteration: {i}/{self.max_iter}")
                
        print("Failed to find a path within the maximum number of iterations.")
        return None

    def sample_informed(self, x_center, l, rotation_matrix):
        """Sample a point within the informed ellipsoid."""
        r = np.dot(rotation_matrix, np.dot(l, self.sample_unit_ball()))
        return tuple(x_center + r)

    def sample_unit_ball(self):
        """Sample a point within a unit ball."""
        while True:
            point = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(point) <= 1:
                return point
def pathlength(path):
    #path=[(x1,y1),(x2,y2),,,]
    length=0
    for i in range(len(path)-1):
        length+=np.linalg.norm(np.array(path[i])-np.array(path[i+1]))
    return length
def exist_path(env):
    # Start and goal positions
    # start = env.positions[2]
    # goal = env.positions[3]
    start = [env.positions[i*2] for i in range(len(env.positions)//2)]
    goal = [env.positions[I*2+1] for I in range(len(env.positions)//2)]
    path_rrt, path_astar, rrt_star_path, prm_path=None,None,None,None
    # # # Run RRT to find a path
    # path_rrt = rrt(start, goal, env)
    # # Run A* to find a path
    # path_astar = a_star(start, goal, env)
    # Run RRT* to find a path
    # rrt_star_planner = RRTStar(start[0], goal[0], env,max_iter=50000,step_size=0.5,radius=10)
    rrt_star_planner = RRTStar(env.nodes[2], env.nodes[3], env,max_iter=50000,step_size=0.3,radius=10)
    rrt_star_path = rrt_star_planner.build_rrt_star()
    # rrt_star_planner = RRTStar(start[0], goal[0], env,max_iter=50000,step_size=0.5,radius=10)
    # rrt_star_path_informed = rrt_star_planner.informed_rrt_star()
    print( pathlength(rrt_star_path),env.cost_matrix[2,3])

    # Run PRM to find a path
    # prm_planner = PRM(start, goal, env, num_samples=1000)
    # prm_planner.build_roadmap()
    # prm_path = prm_planner.find_path()
    # print(2*env.area_size, [pathlength(prm_path1) for prm_path1 in prm_path])

    ax=env.display_environment(show=0)
    # ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if path_rrt or path_astar or rrt_star_path or prm_path:
        if path_rrt:
            path_x, path_y = zip(*path_rrt)
            ax.plot(path_x, path_y, 'r-', label="RRT Path")
        if path_astar:
            path_x, path_y = zip(*path_astar)
            ax.plot(path_x, path_y, 'g-', label="Astar Path")     
        if rrt_star_path:
            path_x, path_y = zip(*rrt_star_path)
            ax.plot(path_x, path_y, 'c-', label="RRTstar Path") 
        if prm_path:
            for prm_path1 in prm_path:
                path_x, path_y = zip(*prm_path1)
                ax.plot(path_x, path_y, 'm-', label="PRM Path")
            # path_x, path_y = zip(*prm_path)
            # ax.plot(path_x, path_y, 'm-', label="PRM Path")   
        for start1, goal1 in zip(start,goal):
            ax.plot(*start1, 'go', label="Start")
            ax.plot(*goal1, 'bo', label="Goal")    
        # ax.plot(*start, 'go', label="Start")
        # ax.plot(*goal, 'bo', label="Goal")
        ax.legend()
    else:
        print("No path found.")

    # plt.show()
    plt.savefig("fig/Environment_Warehouse_path.png")

def safe_initstate_check():
    # this function use to ensure that in each env, all the point is safe to start
    # keep a safe distance to obstcale, lines and other position
    env_list=[SimulationEnvironment_Unmaze,SimulationEnvironment_CA, SimulationEnvironment_CubeCollection,SimulationEnvironment_Warehouse,SimulationEnvironment_maze,SimulationEnvironment_Scenario1,SimulationEnvironment_Scenario2,SimulationEnvironment_Scenario3,SimulationEnvironment_Scenario5,SimulationEnvironment_Scenario6]
    for env_class in env_list:
        print(env_class)
        Unmaze=env_class()
        print(2*Unmaze.area_size/(0.1*1))
        # for po in Unmaze.positions:
        #     p=Point(po)
        #     for line in Unmaze.Stringline:
        #         if p.distance(line)<0.3:print(f"{po},{line},too close! {p.distance(line)}")
        #     for ob in Unmaze.obstacles:
        #         if p.distance(ob)<0.3:print(f"{po},{ob},too close! {p.distance(ob)}")
        #     positions=Unmaze.positions.copy()
        #     positions.remove(po)
        #     for other_po in positions:
        #         opo=Point(other_po)
        #         if p.distance(opo)<0.6:print(f"{po},{other_po},too close! {p.distance(opo)}")
def combination_obs():
    from itertools import combinations
    import random
    lst = [i for i in range(24)]
    combinations_lst = []
    obs_num=[6,9,12,15,18,21,24]
    scen_num=[1,2,3,4,5,6,1]
    # obs_num=[6,9,12,15,18,21,24,27,30,34]
    # scen_num=[1,2,3,4,5,6,7,8,9,1]
    print(sum(scen_num))
    for i, r in enumerate(obs_num):
        c=0
        while c<scen_num[i]:
            newl=sorted(random.sample(lst,r))
            if newl not in combinations_lst:
                combinations_lst.append(newl) 
                c+=1

    for combination in combinations_lst:
        print(list(combination),len(combination))
    print(combinations_lst)
def bidivide_obs():
    lst = [i for i in range(34)]
    combinations_lst=[]
    combinations_lst.append(lst)
    random.shuffle(lst)
    L=len(lst)
    while L>=2:
        l_lst=lst[0:L//2]
        r_lst=lst[L//2:]
        combinations_lst.append(l_lst)
        lst=r_lst
        L=len(lst)
    combinations_lst.append(lst)
    combinations_lst.reverse()
    print(combinations_lst)
    # for lst in combinations_lst:
    #     print(lst)
    return combinations_lst

# from itertools import combinations
# print(list(combinations([0,1,2,3,4,5,6,7], 4)))

# for i in range(5):
    # env = SimulationEnvironment_Scenario7(idx=1,num_obstacles=10,obstacle_size=0.5,irregularity_factor=1)
    # env = SimulationEnvironment_Scenario7_fix(24)
# env = SimulationEnvironment_Scenario2()
# env = SimulationEnvironment_Scenario6()
# env = SimulationEnvironment_CubeCollection()
# env = SimulationEnvironment_Warehouse()
# env.build_roadmap(num_samples=1000,k=20)
    # print(len(env.obstacles))
# ax=env.display_environment(savefig="fig/Environment_CubeCollection_position.png")#show=0,savefig="../fig/Scenario4_{}.png".format(i)
# exist_path(env)
# gen_fixed_scenario7()
# safe_initstate_check()
# dangerous position,SimulationEnvironment_CubeCollection
# combination_obs()
# bidivide_obs()
