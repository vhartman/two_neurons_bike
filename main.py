import numpy as np

import matplotlib.pyplot as plt

import pybullet as p
import math
import time
import pybullet_data

class bikesim:
    def __init__(self):
        p.connect(p.GUI)
        #p.connect(p.DIRECT)

        #p.setPhysicsEngineParameter(numSubSteps=0)
        p.setGravity(0,0,-10)
        p.setRealTimeSimulation(0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane=p.loadURDF("plane.urdf",[0,0,0], useFixedBase=False)
        p.changeDynamics(plane,-1, mass=0,lateralFriction=10, linearDamping=0, angularDamping=0)

        self.bike = p.loadURDF("bicycle/bike.urdf",[0,0,1.0], [0,0,0,-1], useFixedBase=False)

        #timestep = 1./100000.
        self.timestep = 0.001

        self.initial_speed = 5
        self.pedalling = False

    def reset(self):
        leaning_noise = (np.random.normal(0, 0.5)) * 0.
        p.resetBasePositionAndOrientation(bodyUniqueId=self.bike, 
                                          posObj=[0,0,1.0], 
                                          ornObj=p.getQuaternionFromEuler([
                                              np.pi/2 + leaning_noise, 0, 0]))

        p.resetBaseVelocity(self.bike,[self.initial_speed,0,0],[0,0,0])

        steering_noise = (np.random.normal(0, 0.5)) * 0.
        p.resetJointState(self.bike,0,np.pi*2 + steering_noise)
        p.resetJointState(self.bike,1,0)
        p.resetJointState(self.bike,2,0)

        p.setJointMotorControl2(self.bike,0,p.VELOCITY_CONTROL,targetVelocity=0,force=0.0)
        p.setJointMotorControl2(self.bike,1,p.VELOCITY_CONTROL,targetVelocity=0, force=0)

        if self.pedalling:
            p.setJointMotorControl2(self.bike,2,p.VELOCITY_CONTROL,targetVelocity=5, force=100)
        else:
            p.setJointMotorControl2(self.bike,2,p.VELOCITY_CONTROL,targetVelocity=5, force=0)

        p.changeDynamics(self.bike,1,lateralFriction=1,linearDamping=0.00, angularDamping=0)
        p.changeDynamics(self.bike,2,lateralFriction=1,linearDamping=0.00, angularDamping=0)

    def get_frontwheel(self):
        pts = p.getClosestPoints(bodyA=1, bodyB=0, linkIndexA=1, linkIndexB=-1, distance=0.5)

        if len(pts) > 0:
            return pts[0][5][:2]

        return None

    def show(self):
        #p.getCameraImage(320,200)#,renderer=p.ER_BULLET_HARDWARE_OPENGL )
        self.center_camera()

    def visualize(self):
        gui = 1

        positions = []

        p.getCameraImage(320,200)#,renderer=p.ER_BULLET_HARDWARE_OPENGL )
        #p.resetDebugVisualizerCamera(30,90,-5,[0, 0, 1]);

        while (1):
            observation = self.get_observation()

            #print(heading, desired_heading, leaning, desired_lean, torque)
            desired_heading = -1
            #torque = nn_controller(observation, desired_heading)
            torque = None

            _, fell_over = self.step(torque)

            if fell_over:
                break

            if self.timestep != 0:
                time.sleep(self.timestep)

            # extract front wheel position
            fw_pos = self.get_frontwheel()
            if fw_pos is not None:
                positions.append(fw_pos)

            if (gui):
                self.center_camera()

        plt.plot(*np.array(positions).T, color='black', alpha=0.3)

    def center_camera(self):
        distance=5
        yaw = 0
        humanPos, humanOrn = p.getBasePositionAndOrientation(self.bike)
        humanBaseVel = p.getBaseVelocity(self.bike)

        camInfo = p.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        distance=camInfo[10]
        yaw = camInfo[8]
        pitch=camInfo[9]
        targetPos = [0.95*curTargetPos[0]+0.05*humanPos[0],0.95*curTargetPos[1]+0.05*humanPos[1],curTargetPos[2]]
        p.resetDebugVisualizerCamera(distance,yaw,pitch,targetPos);


    def get_observation(self):
        pos, orientation = p.getBasePositionAndOrientation(self.bike)
        angles = p.getEulerFromQuaternion(orientation)

        _, ang_vel = p.getBaseVelocity(self.bike)

        heading = angles[2]
        leaning = angles[0] - np.pi/2

        heading_dot = ang_vel[2]
        #leaning_dot = ang_vel[0] - ang_vel[1]

        back_axis = p.getLinkState(self.bike, 2)
        front_axis = p.getLinkState(self.bike, 1)
        steering_axis = p.getLinkState(self.bike, 0)

        tmp = p.getMatrixFromQuaternion(orientation)
        mat = np.array(tmp).reshape((3, 3))

        x = mat[:, 0]
        leaning_dot = np.dot(x, np.array([ang_vel[0], ang_vel[1], 0]))

        observation = (pos[0], pos[1], pos[2], heading, heading_dot, leaning, leaning_dot)
        return observation

    def step(self, action=None):

        if action is not None:
            action = np.clip(action, -100, 100)
            p.setJointMotorControl2(self.bike,0,p.TORQUE_CONTROL,force=action)

        p.stepSimulation()

        obs = self.get_observation()

        # the bike fell over, or the bike went flying
        fallen_over = False
        if obs[2] < 0.35 or obs[2] > 50:
            fallen_over = True

        return obs, fallen_over

def compute_desired_angle(pd, p):
    rel_pos = np.array(p) - np.array(pd)
    return np.arctan2(rel_pos[1], rel_pos[0])

def get_square_path(w=15):
    square_path = [[w, 0], [w, w], [0, w], [0, 0]]
    return square_path

def get_circle_path(r=20, N=20):
    circle_path = [[r*np.sin(3.1415 * 2*i / N), r*np.cos(3.1415 * 2*i/N)-r] for i in range(N)]
    return circle_path

def run_controller_to_follow_path(sim, path, parameters=None):
    sim.pedalling = True
    sim.reset()
    sim.timestep = 0.000

    current_pt_index = 0
    current_pt = path[current_pt_index]

    max_iters = 15000

    reward = 0
    iters = 0
    positions = []
    while(1):
        observation = sim.get_observation()
        positions.append(observation[:2])

        desired_heading = compute_desired_angle(observation[:2], current_pt)

        reward += 0.1 * -(observation[3] - desired_heading)**2 + 1
        torque = nn_controller(observation, desired_heading, parameters)

        _, fell_over = sim.step(torque)
        sim.show()

        if sim.timestep != 0:
            time.sleep(sim.timestep)

        if fell_over or observation[2] > 1.5:
            reward -= 10000
            break

        if np.linalg.norm(np.array(observation[:2]) - np.array(current_pt)) < 2:
            if current_pt_index == len(path)-1:
                reward += 5000
                break

            current_pt_index += 1
            current_pt = path[current_pt_index]

        iters += 1

        if iters >= max_iters:
            break

    reward += current_pt_index * 1000

    #print(iters)

    return reward, positions

def angle_diff(a1, a2):
    tmp = a1-a2

    if abs(tmp) < np.pi:
        return tmp
    elif tmp > np.pi:
        return -(2 * np.pi - tmp)
    else:
        return (2 * np.pi + tmp)


def nn_controller(observation, desired_heading=-1, parameters=None):
    x, y, z, heading, heading_dot, leaning, leaning_dot = observation

    if parameters is None:
        c1 = -1
        c2 = 100
        c3 = 100
    else:
        c1 = parameters[0]
        c2 = parameters[1]
        c3 = parameters[2]

    heading_diff = angle_diff(desired_heading, heading)
    desired_lean = c1 * (heading_diff)
    desired_lean = 1 / (1 + np.exp(-desired_lean)) - 0.5

    torque = c2 * (desired_lean - leaning) - c3 * leaning_dot

    return torque

def tune_nn_controller():
    sim = bikesim()

    best_param = None
    best_reward = None

    for _ in range(10000):
        parameters = np.random.uniform(low=0, high=100, size=3)
        parameters[0] = parameters[0] * -1 / 20
        r_circle, _ = run_controller_to_follow_path(sim, get_circle_path(), parameters)
        r_square, _ = run_controller_to_follow_path(sim, get_square_path(), parameters)

        r = r_circle + r_square

        if best_reward is None or r > best_reward:
            best_param = parameters
            best_reward = r

            print(r, best_param)

def rl_controller():
    pass

def make_pushing_plot():
    sim = bikesim()
    sim.initial_speed = 3

    plt.figure(figsize=(10,5))
    for _ in range(1000):
        sim.reset()
        sim.visualize()

    plt.xlim([0, 35])
    plt.ylim([-12, 12])

    ax = plt.gca()
    [t.set_color('white') for t in ax.xaxis.get_ticklines()]
    [t.set_color('white') for t in ax.yaxis.get_ticklines()]
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    plt.savefig(f'frontwheel_trace.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

def test_heading_computation():
    desired_heading = compute_desired_angle([0, 0], [5, 0])
    print(desired_heading)

    desired_heading = compute_desired_angle([5, 0], [5, 5])
    print(desired_heading)

    desired_heading = compute_desired_angle([5, 5], [0, 5])
    print(desired_heading)

    desired_heading = compute_desired_angle([0, 5], [0, 0])
    print(desired_heading)

    print("circle:")
    r = 20
    N = 20
    circle_path = [[r*np.sin(3.1415 * 2*i / N), r*np.cos(3.1415 * 2*i/N)-r] for i in range(N)]

    for i in range(N-1):
        desired_heading = compute_desired_angle(circle_path[i], circle_path[i+1])
        print(desired_heading)

def test_angle_diff():
    print(angle_diff(3, -3))
    print(angle_diff(-3, 3))

def compare_and_plot_controllers():
    # found parameters
    found_parameters = [-0.95345113, 22.95930387, 27.7614634]
    hand_tuned_parameters = [-1, 100, 100]
    #hand_tuned_parameters_v2 = [-0.5, 200, 50]

    sim = bikesim()

    circle_path = get_circle_path()
    r_c_f, found_circle_path = run_controller_to_follow_path(sim, circle_path, found_parameters)
    r_c_ht, ht_circle_path = run_controller_to_follow_path(sim, circle_path, hand_tuned_parameters)

    plt.plot(*np.array(circle_path).T, color='black', ls='--')
    plt.plot(*np.array(ht_circle_path).T, color='tab:orange', label='hand tuned')
    plt.plot(*np.array(found_circle_path).T, color='tab:blue', label='found')
    plt.axis('equal')
    plt.legend()
    plt.savefig(f'circle_path.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

    square_path = get_square_path()
    r_s_f, found_sq_path = run_controller_to_follow_path(sim, square_path, found_parameters)
    r_s_ht, ht_sq_path = run_controller_to_follow_path(sim, square_path, hand_tuned_parameters)

    plt.plot(*np.array(square_path).T, color='black', ls='--')
    plt.plot(*np.array(ht_sq_path).T, color='tab:orange', label='hand tuned')
    plt.plot(*np.array(found_sq_path).T, color='tab:blue', label='found')
    plt.axis('equal')
    plt.legend()
    plt.savefig(f'square_path.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

    print("found:", r_c_f + r_s_f)
    print("ht:", r_c_ht + r_s_ht)

def main():
    #make_pushing_plot()

    #test_heading_computation()
    #test_angle_diff()

    # found parameters
    #sim = bikesim()
    #rc, _ = run_controller_to_follow_path(sim, get_circle_path(), parameters, True)
    #rs = run_controller_to_follow_path(sim, get_square_path(), parameters, True)
    #print(rc + rs)

    #tune_nn_controller()
    compare_and_plot_controllers()

main()
