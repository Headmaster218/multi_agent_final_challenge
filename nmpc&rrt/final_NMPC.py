import pybullet as p
import time
import pybullet_data
import yaml
import math
import threading
import numpy as np
import matplotlib.pyplot as plt

from create_obstacles import create_obstacles
from nmpc import compute_velocity, update_state, compute_xref

SIM_TIME = 20.
TIMESTEP = 0.3
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
VMAX = 2.0  
GOAL_TOLERANCE = 0.4  

HORIZON_LENGTH = int(4)
NMPC_TIMESTEP = 0.3

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
        box_id = p.loadURDF("data/turtlebot.urdf", start_position, start_orientation, globalScaling=1)
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
    dis_th = 0.4

    integral_theta, prev_theta_error = 0, 0
    k1, kI1, kD1 = 20, 0.1, 4
    k2, kI2, kD2 = 20, 0.05, 2
    dt = 0.01

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
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule[index]["y"] - y), (schedule[index]["x"] - x))
        '''
        print("Agent", agent, " orientation: ", Orientation)
        print("Agent", agent, " goal_direction: ", goal_direction)
        '''
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
       

        theta_error = theta
        integral_theta += theta_error * dt
        derivative_theta = (theta_error - prev_theta_error) / dt

        linear = k1 * math.cos(theta_error) + kI1 * math.cos(integral_theta) + kD1 * math.cos(derivative_theta)
        angular = k2 * theta_error + kI2 * integral_theta + kD2 * derivative_theta

        prev_theta_error = theta_error

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        # time.sleep(0.001)

    print(agent, "here")



def nmpc_navigation_with_path(agent, goal):
    """
    使用 NMPC 生成从起点到目标的路径，并返回路径点
    """
    
    basePos = p.getBasePositionAndOrientation(agent)
    x = basePos[0][0]
    y = basePos[0][1]
    robot_state = np.array([x, y]) 
    #goal = np.array(goal)
    goal = np.array([4,9])
    path = [list(robot_state)]  

    obstacles = create_obstacles()
    obstacles = np.array(obstacles)

    while not checkPosWithBias(robot_state, goal, GOAL_TOLERANCE):
        
        xref = compute_xref(robot_state, goal, HORIZON_LENGTH, TIMESTEP)
        
        velocity, _ = compute_velocity(robot_state, obstacles, xref)

        robot_state = update_state(robot_state, velocity, TIMESTEP)

        path.append(list(robot_state))

        linear_velocity = velocity[0]
        angular_velocity = velocity[1]
        print("linear_velocity: ", linear_velocity, "angular_velocity: ", angular_velocity)
        rightWheelVelocity = linear_velocity + angular_velocity
        leftWheelVelocity = linear_velocity - angular_velocity
        rightWheelVelocity = 20
        leftWheelVelocity = 20
        #print("robot_state: ", robot_state)
        #print("xref: ", xref)
        #print("linear_velocity: ", linear_velocity, "angular_velocity: ", angular_velocity)
        
        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=50)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=50)

    return path  # 返回路径点

def run(agents, goals):
    """
        Set up loop to publish leftwheel and rightwheel velocity for each robot to reach goal position.

        Args:

        agents: array containing the boxID for each agent

        schedule: dictionary with boxID as key and path to the goal as list for each robot.

        goals: dictionary with boxID as the key and the corresponding goal positions as values
    """
    threads = []
    # multi-agent
    
    for agent in agents:
        t = threading.Thread(target=nmpc_navigation_with_path, args=(agent, goals[agent]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


# physics_client = p.connect(p.GUI, options='--width=1920 --height=1080 --mp4=multi_3.mp4 --mp4fps=30')
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# Disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

plane_id = p.loadURDF("plane.urdf")

global env_loaded
env_loaded = False

# Create environment
env_params = create_env("./final_challenge/env.yaml")

# Create turtlebots
agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params = create_agents("./final_challenge/actors.yaml")
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                     cameraTargetPosition=[4.5, 4.5, 4])


# 初始化路径字典
all_paths = {}

print("-----------------------------------")
print("all_paths: ", agent_box_ids)
print("-----------------------------------")

# Run robot based on cbs output
#run(agent_box_ids, box_id_to_goal, obstacles=env_params["map"]["obstacles"]) 
run(agent_box_ids, box_id_to_goal) 
time.sleep(2)

