import numpy as np
import yaml
import pybullet_data
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Node:
    """定义节点类，代表每一个搜索点。
    Attributes:
        point (np.array): 节点在空间中的坐标。
        cost (float): 从起点到当前节点的总路径成本。
        parent (Node): 当前节点的父节点，用于追溯路径。
    """
    def __init__(self, point, cost=0, parent=None):
        self.point = point  # 节点坐标
        self.cost = cost  # 到达该节点的成本
        self.parent = parent  # 父节点
    def __repr__(self):
        return f"Node(point={self.point})"
    def __str__(self):
        return f"Node at {self.point}"

def distance(point1, point2):
    """计算两点之间的欧几里得距离。
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def nearest(nodes, q_rand):
    """在现有节点中找到离随机点q_rand最近的节点。
    """
    # 在min中每次调用lambda对nodes中的每个节点进行计算，返回与q_rand最近的节点
    return min(nodes, key=lambda node: distance(node.point, q_rand))

def steer(q_near, q_rand, step_size=1.0):
    """从q_near朝向q_rand方向生成一个新的节点，但距离不超过step_size。
    """
    direction = np.array(q_rand) - np.array(q_near.point)
    length = np.linalg.norm(direction)
    direction = direction / length  
    length = min(step_size, length)  
    return Node(q_near.point + direction * length)  

def is_collision_free(node, obstacles):
    """检查给定的节点是否与任何障碍物发生碰撞。
    """
    for (ox, oy, size) in obstacles:
        if np.linalg.norm([node.point[0] - ox, node.point[1] - oy]) <= 0.8*size:
            return False  # 如果节点在障碍物内，返回False
    return True  # 节点无碰撞，返回True

def find_path(nodes, start, goal, goal_threshold=0.5):
    """寻找从起点到终点的路径。
    """
    # 寻找离终点最近的节点作为结束点
    goal_node = min([node for node in nodes if distance(node.point, goal) < goal_threshold], key=lambda n: n.cost, default=None)
    path = []
    if goal_node is None:
        return path
    while goal_node is not None:
        path.append(tuple(goal_node.point))  # 将节点添加到路径
        goal_node = goal_node.parent  # 回溯父节点
    return path[::-1]  # 反转路径

def rrt_star(start, goal, obstacles, num_iterations=1000, search_radius=1):
    """执行RRT*算法以找到起点到终点的路径。
    """
    nodes = [Node(start)]  # 初始化节点列表，包含起点
    for _ in range(num_iterations):
        q_rand = np.random.uniform(0, 10, 2)  # 随机生成点
        q_near = nearest(nodes, q_rand)  # 找到最近的节点
        q_new = steer(q_near, q_rand)  # 生成新节点
        if is_collision_free(q_new, obstacles):  # ensure no collision
            neighbors = [node for node in nodes if distance(node.point, q_new.point) < search_radius and is_collision_free(node, obstacles)]
            q_new.parent = min(neighbors, key=lambda node: node.cost + distance(node.point, q_new.point), default=q_near) if neighbors else q_near
            q_new.cost = q_new.parent.cost + distance(q_new.parent.point, q_new.point)
            nodes.append(q_new)  # add new nodes

    path = find_path(nodes, start, goal)  # find the path
    # plot the path
    #plt.figure(figsize=(5, 5)) 
    
    fig, ax = plt.subplots(figsize=(5, 5))
    for node in nodes:
        if node.parent:
            plt.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], 'b-')  # 以蓝色绘制搜索树
    for i in range(len(path) - 1):
        plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], 'r-')  # 以红色绘制最优路径
    plt.plot(start[0], start[1], 'go')  
    plt.plot(goal[0], goal[1], 'mo')  
    print(path)
    
    for (x, y, size) in obstacles:
        lower_left_x = x - size / 2
        lower_left_y = y - size / 2
        square = patches.Rectangle((lower_left_x, lower_left_y), size, size, 
                                linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(square)

    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.grid()
    plt.title('Obstacles as Filled Squares')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().set_aspect('equal', adjustable='box') 
    
    '''
    for (ox, oy, radius) in obstacles:
        circle = plt.Circle((ox, oy), radius, color='k', fill=True)
        plt.gca().add_patch(circle) 
    '''
    plt.show()

def create_boundaries(length, width):
    """
        create rectangular boundaries with length and width

        Args:

        length: integer

        width: integer
    """
    for i in range(length):
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, -1, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, width, 0.5])
    for i in range(width):
        p.loadURDF("./final_challenge/assets/cube.urdf", [-1, i, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [length, i, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, -1, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, -1, 0.5])

def create_env(yaml_file):
    """
    Creates and loads assets only related to the environment such as boundaries and obstacles.
    Robots are not created in this function (check `create_turtlebot_actor`).
    """
    with open(yaml_file, 'r') as f:
        try:
            env_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e) 
            
    # Create env boundaries
    dimensions = env_params["map"]["dimensions"]
    create_boundaries(dimensions[0], dimensions[1])

    # Create env obstacles
    for obstacle in env_params["map"]["obstacles"]:
        p.loadURDF("./final_challenge/assets/cube.urdf", [obstacle[0], obstacle[1], 0.5])
    return env_params



start = (9, 9)
goal = (6, 0)
size = 1

obstacles = [(-5, -3, 2),(2, 5, 0.5),(-6, 5, 1), (6, -7, 1.5), (7, 7, 1)]
obstacles = [
        [2, 2], [3, 2], [4, 2], [5, 2], [6, 2],
        [0, 4], [1, 4], [2, 4], [3, 4], [4, 4],
        [7, 4], [8, 4], [9, 4], [3, 6], [4, 6],
        [5, 6], [6, 6], [7, 6], [0, 8], [1, 8],
        [2, 8], [5, 8], [6, 8], [7, 8], [8, 8],
        [9, 8], [0, 0], [10, 0], [10, 1], [10, 2],
        [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], 
        [10, 8], [10, 9], [10, 10], [9, 10], [8, 10], 
        [7, 10], [6, 10], [5, 10], [4, 10], [3, 10], 
        [2, 10], [1, 10], [0, 10], [0, 9], [0, 8], 
        [0, 7], [0, 6], [0, 5], [0, 4], [0, 3], 
        [0, 2], [0, 1]]

obstacles = np.array(obstacles)
new_column = np.full((obstacles.shape[0], 1), 1)
obstacles = np.hstack((obstacles, new_column))

nodes = rrt_star(start, goal, obstacles)  
