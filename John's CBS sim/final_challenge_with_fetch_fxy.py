import pybullet as p
import time
import pybullet_data
import yaml
from cbs import cbs
import math
import threading


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


def create_agents(yaml_file):
    """
    Creates and loads turtlebot agents.

    Returns list of agent IDs and dictionary of agent IDs mapped to each agent's goal.
    """
    agent_box_ids = []
    box_id_to_goal = {}
    agent_name_to_box_id = {}
    with open(yaml_file, 'r') as f:
        try:
            agent_yaml_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e)
    start_orientation = p.getQuaternionFromEuler([0,0,0])
    for agent in agent_yaml_params["agents"]:
        start_position = (agent["start"][0], agent["start"][1], 0)
        box_id = p.loadURDF("data/turtlebot.urdf", start_position, start_orientation, globalScaling=1.2)
        agent_box_ids.append(box_id)
        box_id_to_goal[box_id] = agent["goal"]
        agent_name_to_box_id[agent["name"]] = box_id
    return agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params


def read_cbs_output(file):
    """
        Read file from output.yaml, store path list.

        Args:

        output_yaml_file: output file from cbs.

        Returns:

        schedule: path to goal position for each robot.
    """
    with open(file, 'r') as f:
        try:
            params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return params["schedule"]


def checkPosWithBias(Pos, goal, bias):
    """
        Check if pos is at goal with bias

        Args:

        Pos: Position to be checked, [x, y]

        goal: goal position, [x, y]

        bias: bias allowed

        Returns:

        True if pos is at goal, False otherwise
    """
    if(Pos[0] < goal[0] + bias and Pos[0] > goal[0] - bias and Pos[1] < goal[1] + bias and Pos[1] > goal[1] - bias):
        return True
    else:
        return False

x_coords1 = []
y_coords1 = []
x_coords2 = []
y_coords2 = []

def navigation(agent, goal, schedule):
    """
        Set velocity for robots to follow the path in the schedule.

        Args:

        agents: array containing the IDs for each agent

        schedule: dictionary with agent IDs as keys and the list of waypoints to the goal as values

        index: index of the current position in the path.

        Returns:

        Leftwheel and rightwheel velocity.
    """
    basePos = p.getBasePositionAndOrientation(agent)
    index = 0
    dis_th = 0.3
    while(not checkPosWithBias(basePos[0], goal, dis_th)):
        basePos = p.getBasePositionAndOrientation(agent)
        next = [schedule[index]["x"], schedule[index]["y"]]
        if(checkPosWithBias(basePos[0], next, dis_th)):
            index = index + 1
        if(index == len(schedule)):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break
        x = basePos[0][0]
        y = basePos[0][1]

        if agent ==71:
            x_coords1.append(x)  # Record x-coordinate
            y_coords1.append(y)  # Record y-coordinate
        else:
            x_coords2.append(x)  # Record x-coordinate
            y_coords2.append(y)  # Record y-coordinate


        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule[index]["y"] - y), (schedule[index]["x"] - x))

        if(Orientation < 0):
            Orientation = Orientation + 2 * math.pi
        if(goal_direction < 0):
            goal_direction = goal_direction + 2 * math.pi
        theta = goal_direction - Orientation

        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi

        current = [x, y]
        distance = math.dist(current, next)
        k1, k2, A = 30, 13, 20
        linear = k1  * distance * math.cos(theta) +30
        angular = k2 * theta

        if angular > 0.7:
            linear =0


        linear = max(-70,min(70,linear))
        angular = max(-10,min(10,angular))

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        # time.sleep(0.001)
    print(agent, "here")


def run(agents, goals, schedule):
    """
        Set up loop to publish leftwheel and rightwheel velocity for each robot to reach goal position.

        Args:

        agents: array containing the boxID for each agent

        schedule: dictionary with boxID as key and path to the goal as list for each robot.

        goals: dictionary with boxID as the key and the corresponding goal positions as values
    """
    threads = []
    for agent in agents:
        t = threading.Thread(target=navigation, args=(agent, goals[agent], schedule[agent]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def draw_grid(length, width, grid_size=1.0):
    """
    使用调试可视化工具绘制格子，并标记坐标

    length: 格子长度（行数）
    width: 格子宽度（列数）
    grid_size: 每个格子的尺寸（默认为1.0）
    """
    for i in range(length + 1):  # 多一行，以形成格子边界
        for j in range(width + 1):  # 多一列，以形成格子边界
            x = i * grid_size
            y = j * grid_size

            # 绘制垂直线（x方向）
            p.addUserDebugLine([x, 0, 0], [x, width * grid_size, 0], lineColorRGB=[1, 1, 1], lineWidth=1)
            # 绘制水平线（y方向）
            p.addUserDebugLine([0, y, 0], [length * grid_size, y, 0], lineColorRGB=[1, 1, 1], lineWidth=1)

            # 绘制坐标文本
            p.addUserDebugText(f"({x}, {y})", [x, y, 0.2], textSize=1.0, lifeTime=0)


def merge_yaml_files(file_p1, file_p2, output_file):
    # 读取第一个YAML文件
    with open(file_p1, 'r') as f1:
        data_p1 = yaml.safe_load(f1)

    # 读取第二个YAML文件
    with open(file_p2, 'r') as f2:
        data_p2 = yaml.safe_load(f2)

    # 合并两个文件的 schedule 部分
    if 'schedule' in data_p1 and 'schedule' in data_p2:
        # 针对每个 agent 处理时间步
        for key in data_p2['schedule']:
            if key in data_p1['schedule']:
                # 获取 P1 中该 agent 的最大时间步
                max_t_p1 = max([entry['t'] for entry in data_p1['schedule'][key]])

                # 遍历 P2 中该 agent 的 schedule，将其时间步的 t 值调整为在 P1 之后
                for entry in data_p2['schedule'][key]:
                    entry['t'] += max_t_p1 + 1

                # 拼接 P2 的 schedule 到 P1 的 schedule
                data_p1['schedule'][key].extend(data_p2['schedule'][key])
            else:
                # 如果 P1 中没有该 agent, 直接添加
                data_p1['schedule'][key] = data_p2['schedule'][key]

        # 如果 P2 中有 P1 没有的 agent，直接添加
        for key in data_p2['schedule']:
            if key not in data_p1['schedule']:
                data_p1['schedule'][key] = data_p2['schedule'][key]

    # 将合并后的数据写入新的YAML文件
    with open(output_file, 'w') as output:
        yaml.dump(data_p1, output, default_flow_style=False, allow_unicode=True)


# physics_client = p.connect(p.GUI, options='--width=1920 --height=1080 --mp4=multi_3.mp4 --mp4fps=30')
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# Disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

plane_id = p.loadURDF("plane.urdf")

# draw_grid(length=10, width=10, grid_size=1.0)

global env_loaded
env_loaded = False

# Create environment
env_params = create_env("./final_challenge/env.yaml")

# Create turtlebots

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                     cameraTargetPosition=[4.5, 4.5, 4])



agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params = create_agents("./final_challenge/actors1.yaml")
cbs.run(dimensions=env_params["map"]["dimensions"], obstacles=env_params["map"]["obstacles"], agents=agent_yaml_params["agents"], out_file="./final_challenge/cbs_output_fxy_p1.yaml")

with open("./final_challenge/actors2.yaml", 'r') as f:
    try:
        agent_yaml_params = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        print(e)

start_orientation = p.getQuaternionFromEuler([0,0,0])
for index, agent in enumerate(agent_yaml_params["agents"]):
    start_position = (agent["start"][0], agent["start"][1], 0)
    box_id_to_goal[agent_name_to_box_id[index+1]] = agent["goal"]

cbs.run(dimensions=env_params["map"]["dimensions"], obstacles=env_params["map"]["obstacles"], agents=agent_yaml_params["agents"], out_file="./final_challenge/cbs_output_fxy_p2.yaml")
merge_yaml_files('./final_challenge/cbs_output_fxy_p1.yaml', './final_challenge/cbs_output_fxy_p2.yaml', './final_challenge/cbs_output_fxy.yaml')



cbs_schedule = read_cbs_output("./final_challenge/cbs_output_fxy.yaml")
# Replace agent name with box id in cbs_schedule
box_id_to_schedule = {}
for name, value in cbs_schedule.items():
    box_id_to_schedule[agent_name_to_box_id[name]] = value

run(agent_box_ids, box_id_to_goal, box_id_to_schedule)
time.sleep(2)

schedule = box_id_to_schedule[71]

# 提取 'x' 和 'y' 值
x_values = [item['x'] for item in schedule]
y_values = [item['y'] for item in schedule]

schedule = box_id_to_schedule[72]

# 提取 'x' 和 'y' 值
x_values2 = [item['x'] for item in schedule]
y_values2 = [item['y'] for item in schedule]

import matplotlib.pyplot as plt


plt.figure(figsize=(11, 11))
plt.plot(x_coords1[1:], y_coords1[1:], color='blue', label="Robot1 Path")  # Robot1 实际路径为蓝色
plt.plot(x_values, y_values, color='cyan', linestyle='--', label="Robot1 Schedule")  # Robot1 计划路径为青色虚线
plt.scatter([box_id_to_goal[71][0]], [box_id_to_goal[71][1]], color='red', label="Goal 1")  # Robot1 的目标点为红色
plt.scatter(9,9,color='red', label="Start 1")
plt.scatter(9,6,color='red', label="Fetch 1")

plt.plot(x_coords2[1:], y_coords2[1:], color='orange', label="Robot2 Path")  # Robot2 实际路径为橙色
plt.plot(x_values2, y_values2, color='green', linestyle='--', label="Robot2 Schedule")  # Robot2 计划路径为绿色虚线
plt.scatter([box_id_to_goal[72][0]], [box_id_to_goal[72][1]], color='purple', label="Goal 2")  # Robot2 的目标点为紫色
plt.scatter(0,9,color='purple', label="Start 2")
plt.scatter(0,6,color='purple', label="Fetch 2")

# 设置坐标范围
plt.xlim(-1, 10)  # X 坐标范围
plt.ylim(-1, 10)  # Y 坐标范围

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("CBS_Dijkstra result VS Robot trace")
plt.legend()
plt.grid()
plt.show()