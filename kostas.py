import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
# from environment import *
from copy import deepcopy
import time


MarkerSize = 3
MarkerEdgeWidth = 3
plot_flag = 1
acceleration = 30


# Kostas = plt.imread('./Kostas Research Center.png')
kostas = np.array(plt.imread('./Kostas Research Center 2.png'))
# corners = [[370, 640, 542, 276], [215, 289, 681, 610]]
corners = np.array([[111, 408, 300, 7],[43, 127, 517, 435]])
downsampling = 6
corners = (corners/downsampling).astype(int)
# kostas.resize(int(kostas.shape[0]/downsampling), int(kostas.shape[1]/downsampling))
kostas = cv2.resize(kostas, (round(kostas.shape[1]/downsampling), round(kostas.shape[0]/downsampling)))

# general_mission_parameters
class GeneralMissionParameters:
    def __init__(self,accomplished=False, distance_thres=0, position_detected=[], speed=0, num_simple_actions=0):
        """
        distance_thres: Threshold of the distance to consider the drone is in a specific point
        """
        self.accomplished = accomplished # The Flag for accomplish a mission
        self.distance_thres = distance_thres # Threshold of the distance to consider the drone is in a specific point
        self.position_detected = position_detected
        self.speed = speed
        self.num_simple_actions = num_simple_actions


# generate person parameters
class Person:
    def __init__(self, position=np.array([]), orientation=0, speed=0, max_person_speed=0, corners=np.array([])):
        """
        position: [x,y] (pixels)
        orientation: Where the person is walking
        speed: magnitude of speed (pixels/s)
                v_x = speed*cos(orientation-90 (rad))
                v_y = speed*sin(orientation-90 (rad))
        """

        self.corners = corners
        in_cage = 0
        while in_cage==0:
            self.position = np.array([np.random.randint(min(self.corners[0]), max(self.corners[0])),
                                      np.random.randint(min(self.corners[0]), max(self.corners[1]))])
            in_cage = self.check_boundaries()
        self.orientation = orientation
        self.speed = speed
        self.max_person_speed = max_person_speed

    def check_boundaries(self):
        """
        Checks if a drone is within the range of the cage of KRI
        """
        # Slope of the boundaries
        m1 = (self.corners[1][1] - self.corners[1][0]) / (self.corners[0][1] - self.corners[0][0])
        m2 = (self.corners[1][1] - self.corners[1][2]) / (self.corners[0][1] - self.corners[0][2])
        m3 = (self.corners[1][3] - self.corners[1][2]) / (self.corners[0][3] - self.corners[0][2])
        m4 = (self.corners[1][3] - self.corners[1][0]) / (self.corners[0][3] - self.corners[0][0])

        # Control if is insade the cage. The equation is control=m(x-a)
        control1 = m1 * (self.position[0] - self.corners[0][0]) + self.corners[1][0]  # Y must be above the line
        control2 = m2 * (self.position[0] - self.corners[0][2]) + self.corners[1][2]  # Y must be below the line
        control3 = m3 * (self.position[0] - self.corners[0][2]) + self.corners[1][2]  # Y must be below the line
        control4 = m4 * (self.position[0] - self.corners[0][0]) + self.corners[1][0]  # Y must be above the line

        ck1 = np.sign(self.position[1] - control1) + 1  # -1 converts to 0, 1 converts to 2
        ck2 = -np.sign(self.position[1] - control2) + 1
        ck3 = -np.sign(self.position[1] - control3) + 1
        ck4 = np.sign(self.position[1] - control4) + 1

        return ck1 and ck2 and ck3 and ck4

    def plot_person(self):
        """
        plot the person in the map
        """
        plt.plot(self.position[0], self.position[1],
                 'wo', markersize=3, markeredgewidth=3, fillstyle='none')

    def plot_velocity(self):
        """
        plot person's velocity
        """
        plt.quiver(self.position[0], self.position[1],
                   self.speed * np.cos(np.deg2rad(self.orientation - 90)),
                   self.speed * np.sin(np.deg2rad(self.orientation - 90)),
                   color='b', units='dots', scale=0.5, width=3)

    def random_walk(self):
        self.orientation = np.mod(self.orientation + 5 * np.random.normal(),
                                  360)  # Same orientation plus a random from N(0,100)
        self.position = self.position + np.array([self.speed * np.cos(np.deg2rad(self.orientation - 90)),
                                                  self.speed * np.sin(np.deg2rad(self.orientation - 90))])
        self.speed = max(0, min(self.speed + 5 * np.random.normal(), self.max_person_speed))


class Drone:
    class mode:
        def __init__(self, previous='FreeFly', actual='FreeFly',
                     parameters_destination=np.array([]), detection=0):
            """
            % mode:
            % Off --> The drone is off
            % Disarm --> The drone is disarmed
            % Arm --> The drone is armed but not flying
            % FreeFly --> the drone is flying
            % Others, to built
            """
            self.previous = previous
            self.actual = actual
            self.parameters_destination = np.array(parameters_destination)
            self.detection = 0

    def __init__(self, index=0, status_net=True, mode=None,
                 home=np.array([]), position=np.array([]), orientation=0, direction = 0,
                 speed=0, vision=np.array([]), vision_on=True, corners=np.array([])):
        """
        index: drone's index
        status_net: True --> active to the net, False --> not active to the net
        mode: same in the mode class
        home: [x,y] (pixel) Home position (for Return to Launch option)
        position: [x,y] (pixel)
        orientation: Degrees respect to North (clock-wise). 0: North, 90: East, 180: South, 270: West
        speed: magnitude of speed (pixels/s) v_x = speed*cosd(orientation-90), v_y = speed*sind(orientation-90)
        vision:
        vision_on: Set the camera on at the beginning
        """
        self.index = index
        self.status_net = status_net
        self.mode = self.mode()
        self.corners = corners
        in_cage = 0.0
        while in_cage==0.0:
            self.home = np.array([np.random.randint(min(self.corners[0]), max(self.corners[0])),
                         np.random.randint(min(self.corners[0]), max(self.corners[1]))])
            in_cage = self.check_boundaries(is_home=True)
        # self.home = np.array(home)
        self.position = self.home
        # self.orientation = orientation
        self.orientation = np.random.randint(360)
        self.speed = speed
        self.vision = np.array(vision)
        self.vision_on = vision_on

    def plot_drone_home(self):
        """
        plot the drone in the map
        """
        plt.plot(self.home[0], self.home[1],
                 'md', markersize=3, markeredgewidth=3, fillstyle='none')

    def check_boundaries(self, is_home=False):
        """
        Checks if a drone is within the range of the cage of KRI
        """
        # Slope of the boundaries
        m1 = (self.corners[1][1] - self.corners[1][0]) / (self.corners[0][1] - self.corners[0][0])
        m2 = (self.corners[1][1] - self.corners[1][2]) / (self.corners[0][1] - self.corners[0][2])
        m3 = (self.corners[1][3] - self.corners[1][2]) / (self.corners[0][3] - self.corners[0][2])
        m4 = (self.corners[1][3] - self.corners[1][0]) / (self.corners[0][3] - self.corners[0][0])

        if not is_home:
            pos = [self.position[0], self.position[1]]
        else:
            pos = [self.home[0], self.home[1]]

        # Control if is insade the cage. The equation is control=m(x-a)
        control1 = m1 * (pos[0] - self.corners[0][0]) + self.corners[1][0]  # Y must be above the line
        control2 = m2 * (pos[0] - self.corners[0][2]) + self.corners[1][2]  # Y must be below the line
        control3 = m3 * (pos[0] - self.corners[0][2]) + self.corners[1][2]  # Y must be below the line
        control4 = m4 * (pos[0] - self.corners[0][0]) + self.corners[1][0]  # Y must be above the line

        ck1 = np.sign(pos[1] - control1) + 1  # -1 converts to 0, 1 converts to 2
        ck2 = -np.sign(pos[1] - control2) + 1
        ck3 = -np.sign(pos[1] - control3) + 1
        ck4 = np.sign(pos[1] - control4) + 1

        return ck1 and ck2 and ck3 and ck4

    def plot_status(self):
        """
        indicate drone's status with different shape and color
        """
        if self.mode.actual == 'Off':
            plt.plot(self.position[0], self.position[1],
                     'ks', markersize=MarkerSize, fillstyle='none')
        elif self.mode.actual == 'Disarm':
            if self.status_net:
                plt.plot(self.position[0], self.position[1],
                         'gs', markersize=MarkerSize,
                         markeredgewidth=MarkerEdgeWidth, fillstyle='none')
            else:
                plt.plot(self.position[0], self.position[1],
                         'rs', markersize=MarkerSize,
                         markeredgewidth=MarkerEdgeWidth, fillstyle='none')
        elif self.mode.actual == 'Arm':
            if self.status_net:
                plt.plot(self.position[0], self.position[1],
                         'g+', markersize=MarkerSize,
                         markeredgewidth=MarkerEdgeWidth, fillstyle='none')
            else:
                plt.plot(self.position[0], self.position[1],
                         'r+', markersize=MarkerSize,
                         markeredgewidth=MarkerEdgeWidth, fillstyle='none')
        else:
            if self.status_net:
                plt.plot(self.position[0], self.position[1],
                         'gx', markersize=MarkerSize,
                         markeredgewidth=MarkerEdgeWidth, fillstyle='none')
            else:
                plt.plot(self.position[0], self.position[1],
                         'rx', markersize=MarkerSize,
                         markeredgewidth=MarkerEdgeWidth, fillstyle='none')

    def plot_velocity(self):
        """
        plot drone's velocity
        """
        plt.quiver(self.position[0], self.position[1],
                   self.speed * np.cos(np.deg2rad(self.orientation - 90)),
                   self.speed * np.sin(np.deg2rad(self.orientation - 90)),
                   color='b', units='dots', scale=0.5, width=3)

    def plot_vision(self, dense=1):
        """
        plot the angular vision of a drone
        """
        for i in range(0, len(self.vision), dense):
            if sum(self.vision[i]) != 0:
                tmp = np.nonzero(self.vision[i])
                line_range = [tmp[0][0], tmp[0][-1]]
                plt.plot(line_range, [i, i], 'y', LineWidth=4, alpha=0.5)

    def detect_persion(self, person):
        detected = 0
        pos_detected = []
        for person_idx in range(0, len(person)):
            y_idx, x_idx = np.nonzero(self.vision)
            if (round(person[person_idx].position[0]) in x_idx) and (round(person[person_idx].position[1]) in y_idx):
                detected += 1
                pos_detected.append([person[person_idx].position[0],
                                     person[person_idx].position[1]])

        return detected, pos_detected

    def goto(self, general_mission_parameters):
        near = 0
        drone_out = self
        if len(self.mode.parameters_destination) > 0:
            drone_out.orientation = np.rad2deg(np.arctan2(self.mode.parameters_destination[1] - self.position[1],
                                                          self.mode.parameters_destination[0] - self.position[0])) + 90
            if np.linalg.norm(self.position - self.mode.parameters_destination) < self.speed:
                drone_out.speed = np.linalg.norm(self.position - self.mode.parameters_destination)
            if np.linalg.norm(
                    self.position - self.mode.parameters_destination) < general_mission_parameters.distance_thres:
                near = 1
        return drone_out, near

    def action(self, mission_parameters):
            if self.mode.actual == 'Ignore':  # Ignore the detection and continue with the previous status
                # self = self
                pass
            elif self.mode.actual == 'RTL':
                # Update the attributes of the drone based on the destination position.
                # Indicate if the drone is near the destination
                self, near = self.goto(mission_parameters)
                #             drone_out[drone_idx] = drone_in[drone_idx]
                # if the drone is near to the home position, land
                if near:
                    self.speed = 0
                    self.mode.actual = 'Arm'
                    print("Drone {} landed and armed".format(self.index))
            elif self.mode.actual == 'GoToPerson':  # Send the drones to the position of the person detected
                # Update the attributes of the drone based on the destination position.
                # Indicate if the drone is near the destination
                self, near = self.goto(mission_parameters)
                # If the drone is near to the destination position, loiter
                if near:
                    self.speed = 0
                    self.mode.actual = 'Loiter'
                    print("Drone {} is loitering".format(self.index))
                    self.mode.parameters_destination = self.position
            elif self.mode.actual is 'Loiter':  # Keep the drone flying at its current position
                self, near = self.goto(mission_parameters)
                self.speed = 0
            # else:
            #     self = self


# Function

def mission_update(num_drones, drone_in, mission, drone_idx, mission_parameters, p_package_lost, reward):
    if mission == 'Ignore':  # Ignore the detection and continue with the previous status
        drone_out = deepcopy(drone_in)

    elif mission == 'RTL':  # Send the drones back to their launched point
        drone_out = deepcopy(drone_in)
        # If the drone that detects the person is not on the net,
        # do not transmit any information to the remaining drones
        if not drone_in[drone_idx].status_net:
            drone_out[drone_idx].mode.actual = 'RTL'
            print("Drone {} is returning to launch".format(drone_idx))
            # Update the parameters of the mission. In this case, the destination position is the home position.
            drone_out[drone_idx].mode.parameters_destination = drone_in[drone_idx].home
            drone_out[drone_idx].vision_on = False  # Set the camera off when returning to launch
        else:
            reward = reward - 1  # If the drone is in the net, it transmits a package that reduces 1 point the reward
            for idx in range(0, min(num_drones, len(drone_in))):
                if idx == drone_idx:  # The drone that detects the person updates its mission
                    drone_out[idx].mode.actual = 'RTL'
                    print("Drone {} is returning to launch".format(idx))
                    # Update the parameters of the mission. In this case, the destination position is the home position.
                    drone_out[idx].mode.parameters_destination = drone_in[idx].home
                    drone_out[idx].vision_on = False  # Set the camera off when returning to launch
                else:
                    if drone_in[idx].status_net:
                        send_package = ((np.sign(np.random.rand(1) - p_package_lost) + 1) / 2)[0]
                        if send_package == 1:
                            drone_out[idx].mode.actual = 'RTL'
                            print("Drone {} is returning to launch".format(idx))
                            # Update the parameters of the mission. In this case, the
                            # destination position is the home position.
                            drone_out[idx].mode.parameters_destination = drone_in[idx].home
                            drone_out[idx].vision_on = False
                        else:
                            print("Package sent from drone {} to drone {} was lost"
                                  .format(drone_idx, idx))
    elif mission == 'GoToPerson':
        drone_out = deepcopy(drone_in)  # First, all the structure of the drone is copied
        # If the drone that detects the person is not on the net,
        # do not transmit any information to the remaining drones
        if not drone_in[drone_idx].status_net:
            drone_out[drone_idx].mode.actual = 'GoToPerson'
            print("Drone {} is going to position of person detected"
                  .format(drone_idx))
            drone_out[drone_idx].mode.parameters_destination = mission_parameters.position_people[0]
            drone_out[drone_idx].vision_on = False  # Set the camera off when returning to launch
        else:
            reward -= 1
            for idx in range(0, min(num_drones, len(drone_in))):
                if idx == drone_idx:
                    drone_out[idx].mode.actual = 'GoToPerson'
                    print("Drone {} is going to position of person detected".format(idx))
                    drone_out[idx].mode.parameters_destination = mission_parameters.position_people[0]
                    drone_out[idx].vision_on = False  # Set the camera off when returning to launch
                else:
                    if drone_in[idx].status_net:
                        send_package = ((np.sign(np.random.rand(1) - p_package_lost) + 1) / 2)[0]
                        if send_package == 1:
                            drone_out[idx].mode.actual = 'GoToPerson'
                            print("Drone {} is going to position of person detected"
                                  .format(idx))
                            drone_out[idx].mode.parameters_destination = mission_parameters.position_people[0]
                            drone_out[idx].vision_on = False  # Set the camera off when returning to launch
                        else:
                            print("Package sent frone drone {} to drone {} was lost"
                                  .format(drone_idx, idx))
    else:
        drone_out = deepcopy(drone_in)
    return drone_out, reward


# Definition of the mission
# 'Ignore'        --> continue as before
# 'RTL'           --> return to launch
# 'GoToPerson'    --> the drones go to the position where the person was located
# 'Random_action' --> Executes a rondom simple action
#                     from moving forward, backward, right, left, rotate left, rotate right
mission = 'Random_action'

general_mission_parameters =\
    GeneralMissionParameters(accomplished=False, # The mission has not been accomplished at the beginning
                             distance_thres=5,
                             speed=(20/3)/downsampling, # Default speed for the drones, equivalent to 1m/s
                             num_simple_actions=6 # Number of simple actions for the 'Random_action' mode
                             )
print('Mission when locating a person: ' + mission)
reward = 0 # Cost and reward of the mission


# Probability parameters
p_disconnection = 0.0  # Probability the drone disconnects the net
p_misdetection = 0.1    # Probability of not identifying a person when a person is on range
p_package_lost = 0.05   # Probability of lossing a package of information among the drones
p_camera_off = 0.0      # Probability of turning off the camera and not searching


# Creation of the drone structures
num_drones = 6                                    # Maximun number of drones
x_pos = range(kostas.shape[0])                    # Dimensions of the picture
y_pos = range(kostas.shape[1])
X_pos, Y_pos = np.meshgrid(x_pos, y_pos)
radius_vision = (10*20/3)/downsampling            # Radius for vision (pixels)
angular_vision = 60                               # Degrees of vision (<180)
std_drone = 0.1                                   # Standard deviation for the movement of the drone


# Drone status setup

# create a list contain all drone
drone = []
# drone 0
if mission is not 'Random_action':
    mission_actual = 'FreeFly'
else: mission_actual = mission
drone.append(Drone(index=0, status_net=True,
                   mode=Drone.mode(previous='FreeFly',actual=mission_actual, parameters_destination=0),
                   speed=general_mission_parameters.speed,
                   vision=np.zeros(shape=(len(X_pos), len(X_pos[0]))),
                   vision_on=True, corners=corners))
# drone 1
drone.append(Drone(index=1, status_net=True,
                   mode=Drone.mode(previous='FreeFly',actual=mission_actual, parameters_destination=0),
                   speed=general_mission_parameters.speed,
                   vision=np.zeros(shape=(len(X_pos), len(X_pos[0]))),
                   vision_on=True, corners=corners))
# drone 2
drone.append(Drone(index=2, status_net=True,
                   mode=Drone.mode(previous='FreeFly',actual=mission_actual, parameters_destination=0),
                   speed=general_mission_parameters.speed,
                   vision=np.zeros(shape=(len(X_pos), len(X_pos[0]))),
                   vision_on=True, corners=corners))
# drone 3
drone.append(Drone(index=3, status_net=True,
                   mode=Drone.mode(previous='FreeFly',actual=mission_actual, parameters_destination=0),
                   speed=general_mission_parameters.speed,
                   vision=np.zeros(shape=(len(X_pos), len(X_pos[0]))),
                   vision_on=True, corners=corners))
# drone 4
drone.append(Drone(index=4, status_net=True,
                   mode=Drone.mode(previous='FreeFly',actual=mission_actual, parameters_destination=0),
                   speed=general_mission_parameters.speed,
                   vision=np.zeros(shape=(len(X_pos), len(X_pos[0]))),
                   vision_on=True, corners=corners))
# drone 5
drone.append(Drone(index=5, status_net=True,
                   mode=Drone.mode(previous='FreeFly',actual=mission_actual, parameters_destination=0),
                   speed=general_mission_parameters.speed,
                   vision=np.zeros(shape=(len(X_pos), len(X_pos[0]))),
                   vision_on=True, corners=corners))


# Creation poeple
num_people = 3
max_person_speed = 20/3
std_person = 0

# create a list of person
person = []

# Person 0
person.append(Person(orientation=0, speed=0, max_person_speed=max_person_speed, corners=corners))
# Person 1
person.append(Person(orientation=0, speed=0, max_person_speed=max_person_speed, corners=corners))
# Person 2
person.append(Person(orientation=0, speed=0, max_person_speed=max_person_speed, corners=corners))

# fig, ax = plt.subplots(figsize=(10,10))
print("Simulation STARTS")
times = 1

while times < 2:
    # plt.figure(figsize=(10, 10))
    plt.imshow(kostas, origin='lower')
    for i in range(4):
        plt.plot([corners[0][-1 + i], corners[0][i]],
                 [corners[1][-1 + i], corners[1][i]],
                 c='k', LineWidth=1)

    # Plotting Drone properties
    for drone_idx in range(min(num_drones, len(drone))):
        drone[drone_idx].plot_drone_home()
        # Check if the drone is inside the cage
        boundary_check = drone[drone_idx].check_boundaries()
        if not boundary_check:  # If it is not in the cage, status_net goes to 0 and gets stopped
            if not drone[drone_idx].mode.actual == 'Off':
                if time == 1:
                    print("Drone {} is out of the KRI cage!"
                          .format(drone_idx))
                else:
                    print("Drone {} crashed against the net"
                          .format(drone_idx))
                # If the drone is out of the range, disconnet from the net,
                # stop it and turn of the camera
                drone[drone_idx].status_net = False
                drone[drone_idx].mode.actual = 'Off'
                drone[drone_idx].speed = 0
                drone[drone_idx].vision = 0 * drone[drone_idx].vision
                drone[drone_idx].vision_on = False
            plt.plot(drone[drone_idx].position[0],
                     drone[drone_idx].position[1],
                     'ks', markersize=6, fillstyle='none')
        else:
            # Color for the status of the drone
            drone[drone_idx].plot_status()

        # Speed
        drone[drone_idx].plot_velocity()

        # Angular vision
        drone[drone_idx].vision = np.zeros(shape=(len(X_pos), len(X_pos[0])))
        if drone[drone_idx].vision_on:
            drone[drone_idx].vision[(X_pos - drone[drone_idx].position[0]) ** 2
                                    + (Y_pos - drone[drone_idx].position[1]) ** 2
                                    < radius_vision ** 2] = ((np.sign(np.random.rand(1) - p_camera_off) + 1) / 2)[0]
            if (180 <= np.mod(drone[drone_idx].orientation - angular_vision / 2, 360)) and (
                    np.mod(drone[drone_idx].orientation - angular_vision / 2, 360) < 360):
                drone[drone_idx].vision[
                    Y_pos > np.tan(np.deg2rad(90 + drone[drone_idx].orientation - angular_vision / 2)) * (
                                X_pos - drone[drone_idx].position[0]) + drone[drone_idx].position[1]] = 0
            else:
                drone[drone_idx].vision[
                    Y_pos <= np.tan(np.deg2rad(90 + drone[drone_idx].orientation - angular_vision / 2)) * (
                                X_pos - drone[drone_idx].position[0]) + drone[drone_idx].position[1]] = 0
            if (0 <= np.mod(drone[drone_idx].orientation + angular_vision / 2, 360)) and (
                    np.mod(drone[drone_idx].orientation + angular_vision / 2, 360) < 180):
                drone[drone_idx].vision[
                    Y_pos > np.tan(np.deg2rad(90 + drone[drone_idx].orientation + angular_vision / 2)) * (
                                X_pos - drone[drone_idx].position[0]) + drone[drone_idx].position[1]] = 0
            else:
                drone[drone_idx].vision[
                    Y_pos <= np.tan(np.deg2rad(90 + drone[drone_idx].orientation + angular_vision / 2)) * (
                                X_pos - drone[drone_idx].position[0]) + drone[drone_idx].position[1]] = 0
            drone[drone_idx].plot_vision()
        # else:
        #     print("vision off")
            if drone[drone_idx].vision_on:
                reward -= 1
    # Plotting people
    for person_idx in range(min(num_people, len(person))):
        if not person[person_idx].check_boundaries():
            person[person_idx].orientation = person[
                                                 person_idx].orientation + 180
            # If the person hits the borders, turns 180 degrees
        person[person_idx].plot_person()
        person[person_idx].plot_velocity()

    # Detection
    for drone_idx in range(min(num_drones, len(drone))):
        Detected_objects, position_people = drone[drone_idx].detect_persion(person)
        if Detected_objects > 0:
            general_mission_parameters.position_people = position_people
            num_ppl_detected = sum((np.sign(np.random.rand(1, Detected_objects) - p_misdetection) + 1) / 2)[0]
            print("Drone {} detected {} people out of {} objects detected"
                  .format(drone_idx, int(num_ppl_detected), Detected_objects))
            if num_ppl_detected is not 0:
                drone, reward = mission_update(num_drones,
                                               drone,
                                               mission,
                                               drone_idx,
                                               general_mission_parameters,
                                               p_package_lost,
                                               reward)
                reward += 100

    # Action
    # drone = action(num_drones, drone, general_mission_parameters)
    for drone_idx in range(0, min(num_drones, len(drone))):
        drone[drone_idx].action(general_mission_parameters)

    # Drone Updates with random variables
    for drone_idx in range(0, min(num_drones, len(drone))):
        if drone[drone_idx].mode.actual != 'Off':  # Update only if the drone is not off
            # It disconnects with a Bernoulli(p_disconnection)
            drone[drone_idx].status_net = bool((np.sign(np.random.rand(1) - p_disconnection) + 1) / 2)
            if not ((drone[drone_idx].mode.actual == 'Disarm') or (
                    drone[drone_idx].mode.actual == 'Arm')):  # If the drone is flying
                # Same orientation plus a random from N(0,1)
                drone[drone_idx].orientation = drone[drone_idx].orientation + np.random.normal()
                # If the drone changed the flying mode, do not move while planning the new mode
                if drone[drone_idx].mode.previous == drone[drone_idx].mode.actual:
                    # New position = previous position + speed/s x 1s
                    drone[drone_idx].position = drone[drone_idx].position + np.array(
                        [drone[drone_idx].speed * np.cos(np.deg2rad(drone[drone_idx].orientation - 90)),
                         drone[drone_idx].speed * np.sin(np.deg2rad(drone[drone_idx].orientation - 90))])
                    reward = reward - 1
                drone[drone_idx].speed = drone[drone_idx].speed + np.random.normal(0,1)
        drone[drone_idx].mode.previous = drone[drone_idx].mode.actual

    # People updates with random variables
    for person_idx in range(0, min(num_people, len(person))):
        person[person_idx].random_walk()
    plt.title("Reward = {}".format(reward))
    plt.show()
    times += 1
    time.sleep(0.5)


print('End')