import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
import cv2
import pandas as pd
import argparse
import os.path


class Simulation:
    class Environment:
        def __init__(self, background, corners, downsampling=1, acceleration=30, plot_flag=True, info_flag=True, max_time=900):
            fig = np.array(plt.imread(background))
            self.background = cv2.resize(fig, (round(fig.shape[1] / downsampling), round(fig.shape[0] / downsampling)))
            self.corners = (corners / downsampling).astype(float)
            self.x_pos = range(self.background.shape[1])  # Dimensions of the picture
            self.y_pos = range(self.background.shape[0])
            self.max_x_pos = self.background.shape[1]
            self.max_y_pos = self.background.shape[0]
            self.X_pos, self.Y_pos = np.meshgrid(self.x_pos, self.y_pos)
            self.downsampling = downsampling
            self.acceleration = acceleration
            self.plot_flag = plot_flag
            self.info_flag = info_flag
            self.max_time = max_time

    # general_mission_parameters
    class GeneralMissionParameters:
        def __init__(self, name, drone_placement_pattern=0, action_id=None, isDebug=False,
                     accomplished=False, distance_thres=0, position_detected=[],
                     position_people=np.array([]), speed=0.0, num_simple_actions=0, num_drones=0, num_people=0):
            """
            distance_thres: Threshold of the distance to consider the drone is in a specific point
            Home position for the drone
            placed_pattern:
                0 --> Random position within the cage
                1 --> Distributed over one edge
                2 --> Distributed over all edges
                3 --> Starting from one corner
                4 --> Starting from all corner
            """

            self.name = name  # Name of mission
            self.drone_placement_pattern = drone_placement_pattern
            if name == "Random_action":
                self.mission_actual = name  # Random Generation Flag
            else:
                self.mission_actual = "FreeFly"
            self.action_id = action_id  # Action ID (a num_drone * 1 vector)
            self.isDebug = isDebug  # Debug Flag
            self.accomplished = accomplished  # The Flag for accomplish a mission
            self.distance_thres = distance_thres  # Threshold of the distance to consider drone is in a specific point
            self.position_detected = position_detected
            self.position_people = position_people
            self.speed = speed
            self.num_simple_actions = num_simple_actions
            self.num_drones = num_drones
            self.num_people = num_people

    # Cost and Reward of the mission
    class Reward:
        def __init__(self, total=0, person_detected=900, cost_movement=0.1,
                    cost_camera_use=0, cost_communications=0, cost_crash=100):
        #def __init__(self, total=0, person_detected=900, cost_movement=1,
        #             cost_camera_use=0.5, cost_communications=0.1, cost_crash=200):
            self.total = total
            self.person_detected = person_detected
            self.cost_movement = cost_movement
            self.cost_camera_use = cost_camera_use
            self.cost_communications = cost_communications
            self.cost_crash = cost_crash

    # generate person parameters
    class Person:
        def __init__(self, index=0, position=np.array([]), orientation=0,
                     speed=0.0, max_person_speed=0.0, corners=np.array([]), std_person=0):
            """
            position: [x,y] (pixels)
            orientation: Where the person is walking
            speed: magnitude of speed (pixels/s)
                    v_x = speed*cos(orientation-90 (rad))
                    v_y = speed*sin(orientation-90 (rad))
            """
            self.index = index
            self.corners = corners
            # generate in cage random position
            in_cage = 0
            while in_cage == 0:
                #self.position = np.array([np.random.randint(min(self.corners[0]), max(self.corners[0])),
                #                          np.random.randint(min(self.corners[0]), max(self.corners[1]))])
                self.position = np.array([np.random.randint(min(self.corners[0]), max(self.corners[0])),
                                          np.random.randint(min(self.corners[0]) + 1/4 * max(self.corners[1]), max(self.corners[1]))])
                in_cage = self.check_boundaries()
            self.orientation = orientation
            self.speed = speed
            self.max_person_speed = max_person_speed
            self.std_person = std_person

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
                     'bo', markersize=3, markeredgewidth=3, fillstyle='none')

        def plot_velocity(self):
            """
            plot person's velocity
            """
            plt.quiver(self.position[0], self.position[1],
                       self.speed * np.cos(np.deg2rad(self.orientation - 90)),
                       self.speed * np.sin(np.deg2rad(self.orientation - 90)),
                       color='b', units='dots', scale=0.5, width=3)

        def random_walk(self):
            # Same orientation plus a random from N(0,100)
            self.orientation = np.mod(self.orientation + self.std_person * np.random.normal(), 360)
            # New position = previous position + speed/s * 1s
            self.position = self.position + np.array([self.speed * np.cos(np.deg2rad(self.orientation - 90)),
                                                      self.speed * np.sin(np.deg2rad(self.orientation - 90))])
            # Same speed plus a random from N(0,25)
            self.speed = max(0, min(self.speed + self.std_person * np.random.normal(), self.max_person_speed))

    class Drone:
        class Mode:
            def __init__(self, previous='FreeFly', actual='FreeFly',
                         parameters_destination=np.array([]), parameters_detection=0):
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
                self.parameters_detection = parameters_detection

        def __init__(self,placed_pattern=0, dowmsampling=6, index=0, status_net=True, mode=Mode(),
                     home=np.array([]), orientation=0,
                     speed=0.0, vision=np.array([]), vision_on=True, corners=np.array([]),
                     radius_vision=0.0, angular_vision=0.0, std_drone=0.0,
                     p_disconnection=0.0, p_misdetection=0.1, p_package_lost=0.05, p_camera_off=0.0):
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
            self.mode = mode
            self.corners = corners
            self.home_position(placed_pattern, dowmsampling)
            self.position = self.home
            # self.orientation = orientation
            # self.orientation = np.random.randint(360)
            self.direction = self.orientation
            self.speed = speed
            self.vision = np.array(vision)
            self.vision_on = vision_on

            self.radius_vision = radius_vision  # Radius for vision (pixels)
            self.angular_vision = angular_vision  # Degrees of vision (<180)
            self.std_drone = std_drone  # Standard deviation for the movement of the drone

            # Probability parameters
            self.p_disconnection = p_disconnection  # Probability the drone disconnects the net
            self.p_misdetection = p_misdetection  # Probability of not identifying a person when a person is on range
            self.p_package_lost = p_package_lost  # Probability of lossing a package of information among the drones
            self.p_camera_off = p_camera_off  # Probability of turning off the camera and not searching

        def home_position(self,placed_pattern, downsampling):
            """
            Home position for the drone
            :param placed_pattern:
                0 --> Random position within the cage
                1 --> Distributed over one edge
                2 --> Distributed over all edges
                3 --> Starting from one corner
                4 --> Starting from all corner
            :param downsampling: down-sampling value for home margin
            :return:
            """
            home_margin = 6.66/downsampling
            if placed_pattern == 0:
                in_cage = 0.0
                while in_cage == 0.0:
                    self.home = np.array([np.random.randint(min(self.corners[0]), max(self.corners[0])),
                                          np.random.randint(min(self.corners[0]), max(self.corners[1]))])
                    in_cage = self.check_boundaries(is_home=True)
                    self.orientation = np.random.randint(360)
            elif placed_pattern == 1:  # Distributed over one edge (bottom edge)
                aux = np.random.randint(round(self.corners[0][1]-self.corners[0][0] - 2 * home_margin)) +\
                      self.corners[0][0] + home_margin
                self.home = np.array([aux,
                                      (self.corners[1][1]-self.corners[1][0])/(self.corners[0][1] - self.corners[0][0])*
                                      (aux-self.corners[0][0])+self.corners[1][0]+home_margin])
                self.orientation = 180
            elif placed_pattern == 2:  # Distributed over all edges randomly
                edge = np.random.randint(4)
                if edge == 0:  # bottom
                    aux = np.random.randint(round(self.corners[0][1] - self.corners[0][0] - 2 * home_margin)) +\
                          self.corners[0][0] + home_margin
                    self.home = np.array([aux,
                                          (self.corners[1][1] - self.corners[1][0]) / (
                                                      self.corners[0][1] - self.corners[0][0]) *
                                          (aux - self.corners[0][0]) + self.corners[1][0] + home_margin])
                    self.orientation = 180
                elif edge == 1:  # right
                    aux = np.random.randint(round(self.corners[0][1] - self.corners[0][2] - 2 * home_margin)) + \
                          self.corners[0][2] + home_margin
                    self.home = np.array([aux,
                                          (self.corners[1][1] - self.corners[1][2]) / (
                                                      self.corners[0][1] - self.corners[0][2]) *
                                          (aux - self.corners[0][2] + home_margin) + self.corners[1][2] ])
                    self.orientation = 270
                elif edge == 2:  # top
                    aux = np.random.randint(round(self.corners[0][2] - self.corners[0][3] - 2 * home_margin)) + \
                          self.corners[0][3] + home_margin
                    self.home = np.array([aux,
                                          (self.corners[1][2] - self.corners[1][3]) / (
                                                      self.corners[0][2] - self.corners[0][3]) *
                                          (aux - self.corners[0][3]) + self.corners[1][3] - home_margin])
                    self.orientation = 0
                elif edge == 3:  # left
                    aux = np.random.randint(round(self.corners[0][0] - self.corners[0][3] - 2 * home_margin)) + \
                          self.corners[0][3] + home_margin
                    self.home = np.array([aux,
                                          (self.corners[0][1] - self.corners[1][3]) / (
                                                  self.corners[0][0] - self.corners[0][3]) *
                                          (aux - self.corners[0][3] - home_margin) + self.corners[1][3]])
                    self.orientation = 90
            elif placed_pattern == 3:  # Starting from one corner (left bottom)
                # self.home = np.array([self.corners[0][0] + home_margin,
                #                       self.corners[1][0] + 1.5 * home_margin])
                self.home = np.array([self.corners[0][0] + 3 * home_margin,
                                      self.corners[1][0] + 3 * home_margin])
                self.orientation = 135
            elif placed_pattern == 4:  # Starting from all corners randomly
                corner = np.random.randint(4)

                if corner == 0:  # Bottom left
                    self.home = np.array([self.corners[0][0] + home_margin,
                                          self.corners[1][0] + 1.5 * home_margin])
                    self.orientation = 135
                elif corner == 1:  # Bottom right
                    self.home = np.array([self.corners[0][1] - 1.5 * home_margin,
                                          self.corners[1][1] + home_margin])
                    self.orientation = 225
                elif corner == 2:  # Top right
                    self.home = np.array([self.corners[0][2] - home_margin,
                                          self.corners[1][2] - 1.5 * home_margin])
                    self.orientation = 315
                elif corner == 3:
                    self.home = np.array([self.corners[0][3] + 1.5 *  home_margin,
                                          self.corners[1][3] - home_margin])
                    self.orientation = 45

        def plot_drone_home(self):
            """
            plot the drone in the map
            """
            plt.plot(self.home[0], self.home[1],
                     'md', markersize=1.5, markeredgewidth=1, fillstyle='none')

        def check_boundaries(self, is_home=False):
            """
            Checks if a drone is within the range of the cage of KRI
            :param is_home: whether the checking is for initialization
            :return: Flag to indicate whether is in the box
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
                         'ks', markersize=6, fillstyle='none')
            elif self.mode.actual == 'Disarm':
                if self.status_net:
                    plt.plot(self.position[0], self.position[1],
                             'gs', markersize=6,
                             markeredgewidth=1, fillstyle='none')
                else:
                    plt.plot(self.position[0], self.position[1],
                             'rs', markersize=6,
                             markeredgewidth=1, fillstyle='none')
            elif self.mode.actual == 'Arm':
                if self.status_net:
                    plt.plot(self.position[0], self.position[1],
                             'g+', markersize=6,
                             markeredgewidth=1, fillstyle='none')
                else:
                    plt.plot(self.position[0], self.position[1],
                             'r+', markersize=6,
                             markeredgewidth=1, fillstyle='none')
            else:
                if self.status_net:
                    plt.plot(self.position[0], self.position[1],
                             'gx', markersize=6,
                             markeredgewidth=1, fillstyle='none')
                else:
                    plt.plot(self.position[0], self.position[1],
                             'rx', markersize=6,
                             markeredgewidth=1, fillstyle='none')

        def plot_velocity(self):
            """
            plot drone's velocity
            """
            plt.quiver(self.position[0], self.position[1],
                       self.speed * np.cos(np.deg2rad(self.direction - 90)),
                       self.speed * np.sin(np.deg2rad(self.direction - 90)),
                       color='b', units='dots', scale=0.5, width=3)

        def plot_vision(self, dense=1):
            """
            plot the angular vision of a drone
            """
            for i in range(0, len(self.vision), dense):
                if sum(self.vision[i]) != 0:
                    tmp = np.nonzero(self.vision[i])
                    line_range = [tmp[0][-1], tmp[0][0]]
                    plt.plot(line_range, [i, i], 'y', LineWidth=4, alpha=0.5)

        def detect_person(self, person):
            """
            :param person: person class with people position
            :return:
                detected: number of objects have been detected
                pos_detected: the position of object has been detected
            """
            detected = 0
            pos_detected = []
            for person_idx in range(0, len(person)):
                y_idx, x_idx = np.nonzero(self.vision)
                if (round(person[person_idx].position[0]) in x_idx) and (
                        round(person[person_idx].position[1]) in y_idx):
                    detected += 1
                    pos_detected.append((person[person_idx].position[0],
                                         person[person_idx].position[1]))

            return detected, pos_detected

        def goto(self, general_mission_parameters):
            near = 0
            self.direction = np.rad2deg(np.arctan2(self.mode.parameters_destination[1] - self.position[1],
                                                   self.mode.parameters_destination[0] - self.position[0])) + 90
            if np.linalg.norm(self.position - self.mode.parameters_destination) < self.speed:
                self.speed = np.linalg.norm(self.position - self.mode.parameters_destination)
            if np.linalg.norm(
                    self.position - self.mode.parameters_destination) < general_mission_parameters.distance_thres:
                near = 1
            return near

        def action(self, mission_parameters):
            if self.mode.actual == 'Ignore':  # Ignore the detection and continue with the previous status
                # self = self
                pass
            elif self.mode.actual == 'RTL':
                # Update the attributes of the drone based on the destination position.
                # Indicate if the drone is near the destination
                near = self.goto(mission_parameters)
                #             drone_out[drone_idx] = drone_in[drone_idx]
                # if the drone is near to the home position, land
                if near:
                    self.speed = 0
                    self.mode.actual = 'Arm'
                    if self.environment.info_flag:
                        print("Drone {} landed and armed".format(self.index))
            elif self.mode.actual == 'GoToPerson':  # Send the drones to the position of the person detected
                # Update the attributes of the drone based on the destination position.
                # Indicate if the drone is near the destination
                near = self.goto(mission_parameters)  # If the drone is near to the destination position, loiter
                if near:
                    self.speed = 0
                    self.mode.actual = 'Loiter'
                    if self.environment.info_flag:
                        print("Drone {} is loitering".format(self.index))
                    self.mode.parameters_destination = self.position
            elif self.mode.actual is 'Loiter':  # Keep the drone flying at its current position
                near = self.goto(mission_parameters)
                self.speed = 0

            # Define the basic random actions: front, back, right, left, rotation +90, -90, 180
            elif self.mode.actual is 'Random_action':
                #if mission_parameters.isDebug:
                #    drone_action_id = np.random.randint(mission_parameters.num_simple_actions)
                #else:
                #    drone_action_id = mission_parameters.action_id[self.index]
                drone_action_id = np.random.randint(mission_parameters.num_simple_actions)
                self.simple_action(drone_action_id, mission_parameters)
            elif self.mode.actual is 'FreeFly':
                drone_action_id = mission_parameters.action_id[self.index]
                self.simple_action(drone_action_id, mission_parameters)
            else:
                pass

        def simple_action(self, action_id, mission_parameters):
            # action_id = 0 --> Move 1 meter/s to the north
            # action_id = 1 --> Move 1 meter/s to the south
            # action_id = 2 --> Move 1 meter/s to the east
            # action_id = 3 --> Move 1 meter/s to the west
            # action_id = 4 --> Rotate 30 degrees clockwise
            # action_id = 5 --> Rotate 30 degrees counter clockwise
            # action_id = 6 --> Rotate 180 degrees clockwise
            if action_id == 0:
                self.direction = 0
                self.speed = mission_parameters.speed
            elif action_id == 1:
                self.direction = 180
                self.speed = mission_parameters.speed
            elif action_id == 2:
                self.direction = 90
                self.speed = mission_parameters.speed
            elif action_id == 3:
                self.direction = 270
                self.speed = mission_parameters.speed
            elif action_id == 4:
                self.orientation = self.orientation + 30
                self.speed = 0
            elif action_id == 5:
                self.orientation = self.orientation - 30
                self.speed = 0
            elif action_id == 6:
                self.orientation = self.orientation + 180
                self.speed = 0

    # Function
    def mission_update(self, drone_idx):
        # Ignore the detection and continue with the previous status
        if self.general_mission_parameters.name == 'Ignore':
            # drone_out = deepcopy(drone_in)
            pass
        elif self.general_mission_parameters.name == 'RTL':  # Send the drones back to their launched point
            # drone_out = deepcopy(drone_in)
            # If the drone that detects the person is not on the net,
            # do not transmit any information to the remaining drones
            if not self.drones[drone_idx].status_net:
                self.drones[drone_idx].mode.actual = 'RTL'
                if self.environment.info_flag:
                    print("Drone {} is returning to launch".format(drone_idx))
                # Update the parameters of the mission. In this case, the destination position is the home position.
                self.drones[drone_idx].mode.parameters_destination = self.drones[drone_idx].home
                self.drones[drone_idx].vision_on = False  # Set the camera off when returning to launch
            else:
                # If the drone is in the net, it transmits a package that reduces 1 point the reward
                self.reward.total -= self.reward.cost_communications
                self.drones[drone_idx].reward -= self.reward.cost_communications
                for idx in range(0, min(self.general_mission_parameters.num_drones, len(self.drones))):
                    if idx == drone_idx:  # The drone that detects the person updates its mission
                        self.drones[idx].mode.actual = 'RTL'
                        if self.environment.info_flag:
                            print("Drone {} is returning to launch".format(idx))
                        # Update the parameters of the mission.
                        # In this case, the destination position is the home position.
                        self.drones[idx].mode.parameters_destination = self.drones[idx].home
                        self.drones[idx].vision_on = False  # Set the camera off when returning to launch
                    else:
                        if self.drones[idx].status_net:
                            send_package = ((np.sign(np.random.rand(1) - self.drones[idx].p_package_lost) + 1) / 2)[0]
                            if send_package == 1:
                                self.drones[idx].mode.actual = 'RTL'
                                if self.environment.info_flag:
                                    print("Drone {} is returning to launch".format(idx))
                                # Update the parameters of the mission. In this case, the
                                # destination position is the home position.
                                self.drones[idx].mode.parameters_destination = self.drones[idx].home
                                self.drones[idx].vision_on = False
                            else:
                                if self.environment.info_flag:
                                    print("Package sent from drone {} to drone {} was lost"
                                        .format(drone_idx, idx))
        elif self.general_mission_parameters.name == 'GoToPerson':
            self.reward.total += self.reward.person_detected
            self.drones[drone_idx].reward += self.reward.person_detected

            # drone_out = deepcopy(drone_in)  # First, all the structure of the drone is copied
            # If the drone that detects the person is not on the net,
            # do not transmit any information to the remaining drones
            if not self.drones[drone_idx].status_net:
                self.drones[drone_idx].mode.actual = 'GoToPerson'
                if self.environment.info_flag:
                    print("Drone {} is going to position of person detected"
                        .format(drone_idx))
                self.drones[drone_idx].mode.parameters_destination = self.general_mission_parameters.position_people[0]
                self.drones[drone_idx].vision_on = False  # Set the camera off when returning to launch
            else:
                self.reward.total -= self.reward.cost_communications
                self.drones[drone_idx].reward -= self.reward.cost_communications
                for idx in range(0, min(self.general_mission_parameters.num_drones, len(self.drones))):
                    if idx == drone_idx:
                        self.drones[idx].mode.actual = 'GoToPerson'
                        if self.environment.info_flag:
                            print("Drone {} is going to position of person detected".format(idx))
                        self.drones[idx].mode.parameters_destination = \
                            self.general_mission_parameters.position_people[0]
                        self.drones[idx].vision_on = False  # Set the camera off when returning to launch
                    else:
                        if self.drones[idx].status_net:
                            send_package = ((np.sign(np.random.rand(1) - self.drones[idx].p_package_lost) + 1) / 2)[0]
                            if send_package == 1:
                                self.drones[idx].mode.actual = 'GoToPerson'
                                if self.environment.info_flag:
                                    print("Drone {} is going to position of person detected"
                                        .format(idx))
                                self.drones[idx].mode.parameters_destination = \
                                    self.general_mission_parameters.position_people[0]
                                self.drones[idx].vision_on = False  # Set the camera off when returning to launch
                            else:
                                if self.environment.info_flag:
                                    print("Package sent frone drone {} to drone {} was lost"
                                        .format(drone_idx, idx))
        elif self.general_mission_parameters.name == "Random_action" or self.general_mission_parameters.name == 'FreeFly':
            self.drones[drone_idx].mode.parameters_detection = 0
            for idx_ppl in range(len(self.general_mission_parameters.position_people)):
                if self.general_mission_parameters.position_people[idx_ppl] not in \
                        self.general_mission_parameters.position_detected:
                    if self.environment.info_flag:
                        print("One person was detected at position: ({},{}), for a total of {} people detected."
                              .format(self.general_mission_parameters.position_people[idx_ppl][0],
                                      self.general_mission_parameters.position_people[idx_ppl][1],
                                      len(self.general_mission_parameters.position_detected)))
                    self.general_mission_parameters.position_detected.\
                        append(self.general_mission_parameters.position_people[idx_ppl])
                    self.reward.total += self.reward.person_detected
                    self.drones[drone_idx].reward += self.reward.person_detected
                    self.drones[drone_idx].mode.parameters_detection += 1
                    if len(self.general_mission_parameters.position_detected) == \
                            self.general_mission_parameters.num_people:
                        if self.environment.info_flag:
                            print("All {} people have been detected. \nMission accomplished!".format(
                                self.general_mission_parameters.num_people))
                        self.general_mission_parameters.accomplished = True
        else:
            pass

    # Drone status setup
    def generate_drones(self):
        # Create the drone structures
        drones = [self.Drone(dowmsampling=self.environment.downsampling, index=i, status_net=True,
                                placed_pattern=self.general_mission_parameters.drone_placement_pattern,
                                mode=self.Drone.Mode(previous='FreeFly',
                                                     actual=self.general_mission_parameters.mission_actual,
                                                     parameters_destination=np.array([])),
                                speed=self.general_mission_parameters.speed,
                                vision=np.zeros(shape=(self.environment.X_pos.shape[0],
                                                       self.environment.X_pos.shape[1])),
                                radius_vision=(10*20/3)/self.environment.downsampling,  # Radius for vision (pixels)
                                angular_vision=60,  # Degrees of vision (<180)
                                std_drone=0.1,  # Standard deviation for the movement of the drone
                                vision_on=True, corners=self.environment.corners)

            for i in range(self.general_mission_parameters.num_drones)
        ]

        # drone = list()
        # # drone 0
        # drone.append(self.Drone(dowmsampling=self.environment.downsampling, index=0, status_net=True,
        #                         placed_pattern=self.general_mission_parameters.drone_placement_pattern,
        #                         mode=self.Drone.Mode(previous='FreeFly',
        #                                              actual=self.general_mission_parameters.mission_actual,
        #                                              parameters_destination=np.array([])),
        #                         speed=self.general_mission_parameters.speed,
        #                         vision=np.zeros(shape=(self.environment.X_pos.shape[0],
        #                                                self.environment.X_pos.shape[1])),
        #                         radius_vision=(10*20/3)/self.environment.downsampling,  # Radius for vision (pixels)
        #                         angular_vision=60,  # Degrees of vision (<180)
        #                         std_drone=0.1,  # Standard deviation for the movement of the drone
        #                         vision_on=True, corners=self.environment.corners))
        # # drone 1
        # drone.append(self.Drone(dowmsampling=self.environment.downsampling, index=1, status_net=True,
        #                         placed_pattern=self.general_mission_parameters.drone_placement_pattern,
        #                         mode=self.Drone.Mode(previous='FreeFly',
        #                                              actual=self.general_mission_parameters.mission_actual,
        #                                              parameters_destination=np.array([])),
        #                         speed=self.general_mission_parameters.speed,
        #                         vision=np.zeros(shape=(self.environment.X_pos.shape[0],
        #                                                self.environment.X_pos.shape[1])),
        #                         radius_vision=(10*20/3)/self.environment.downsampling,  # Radius for vision (pixels)
        #                         angular_vision=60,  # Degrees of vision (<180)
        #                         std_drone=0.1,  # Standard deviation for the movement of the drone
        #                         vision_on=True, corners=self.environment.corners))
        # # drone 2
        # drone.append(self.Drone(dowmsampling=self.environment.downsampling, index=2, status_net=True,
        #                         placed_pattern=self.general_mission_parameters.drone_placement_pattern,
        #                         mode=self.Drone.Mode(previous='FreeFly',
        #                                              actual=self.general_mission_parameters.mission_actual,
        #                                              parameters_destination=np.array([])),
        #                         speed=self.general_mission_parameters.speed,
        #                         vision=np.zeros(shape=(self.environment.X_pos.shape[0],
        #                                                self.environment.X_pos.shape[1])),
        #                         radius_vision=(10*20/3)/self.environment.downsampling,  # Radius for vision (pixels)
        #                         angular_vision=60,  # Degrees of vision (<180)
        #                         std_drone=0.1,  # Standard deviation for the movement of the drone
        #                         vision_on=True, corners=self.environment.corners))
        # # drone 3
        # drone.append(self.Drone(dowmsampling=self.environment.downsampling, index=3, status_net=True,
        #                         placed_pattern=self.general_mission_parameters.drone_placement_pattern,
        #                         mode=self.Drone.Mode(previous='FreeFly',
        #                                              actual=self.general_mission_parameters.mission_actual,
        #                                              parameters_destination=np.array([])),
        #                         speed=self.general_mission_parameters.speed,
        #                         vision=np.zeros(shape=(self.environment.X_pos.shape[0],
        #                                                self.environment.X_pos.shape[1])),
        #                         radius_vision=(10*20/3)/self.environment.downsampling,  # Radius for vision (pixels)
        #                         angular_vision=60,  # Degrees of vision (<180)
        #                         std_drone=0.1,  # Standard deviation for the movement of the drone
        #                         vision_on=True, corners=self.environment.corners))
        # # drone 4
        # drone.append(self.Drone(dowmsampling=self.environment.downsampling, index=4, status_net=True,
        #                         placed_pattern=self.general_mission_parameters.drone_placement_pattern,
        #                         mode=self.Drone.Mode(previous='FreeFly',
        #                                              actual=self.general_mission_parameters.mission_actual,
        #                                              parameters_destination=np.array([])),
        #                         speed=self.general_mission_parameters.speed,
        #                         vision=np.zeros(shape=(self.environment.X_pos.shape[0],
        #                                                self.environment.X_pos.shape[1])),
        #                         radius_vision=(10*20/3)/self.environment.downsampling,  # Radius for vision (pixels)
        #                         angular_vision=60,  # Degrees of vision (<180)
        #                         std_drone=0.1,  # Standard deviation for the movement of the drone
        #                         vision_on=True, corners=self.environment.corners))
        # # drone 5
        # drone.append(self.Drone(dowmsampling=self.environment.downsampling, index=5, status_net=True,
        #                         placed_pattern=self.general_mission_parameters.drone_placement_pattern,
        #                         mode=self.Drone.Mode(previous='FreeFly',
        #                                              actual=self.general_mission_parameters.mission_actual,
        #                                              parameters_destination=np.array([])),
        #                         speed=self.general_mission_parameters.speed,
        #                         vision=np.zeros(shape=(self.environment.X_pos.shape[0],
        #                                                self.environment.X_pos.shape[1])),
        #                         radius_vision=(10*20/3)/self.environment.downsampling,  # Radius for vision (pixels)
        #                         angular_vision=60,  # Degrees of vision (<180)
        #                         std_drone=0.1,  # Standard deviation for the movement of the drone
        #                         vision_on=True, corners=self.environment.corners))
        return drones

    def generate_people(self, max_person_speed=20/3):
        # Creation poeple
        person = list()
        # Person 0
        person.append(self.Person(orientation=0, speed=0, max_person_speed=max_person_speed,
                                  corners=self.environment.corners, std_person=0))
        # Person 1
        person.append(self.Person(orientation=0, speed=0, max_person_speed=max_person_speed,
                                  corners=self.environment.corners, std_person=0))
        # Person 2
        person.append(self.Person(orientation=0, speed=0, max_person_speed=max_person_speed,
                                  corners=self.environment.corners, std_person=0))
        return person

    def __init__(self, num_drones=6, plot_flag=True, info_flag=True, downsampling=6, acceleration=8, max_time=900, drone_placement_pattern=0):
        """
        :param plot_flag:
        :param downsampling:
        :param acceleration:
        :param max_time:
        :param drone_placement_pattern:
                0 --> Random position within the cage
                1 --> Distributed over one edge
                2 --> Distributed over all edges
                3 --> Starting from one corner
                4 --> Starting from all corner
        """
        my_path = os.path.abspath(os.path.dirname(__file__))

        self.environment = self.Environment(os.path.join(my_path, "./Kostas Research Center 2.png"),
                                            corners=np.array([[111, 408, 300, 7], [43, 127, 517, 435]]),
                                            downsampling=downsampling,              # downsampling parameter
                                            acceleration=acceleration,              # acceleration parameter
                                            plot_flag=plot_flag,                    # plot flag
                                            info_flag=info_flag,                    # Info flag
                                            max_time=max_time)                      # max running time
        self.general_mission_parameters = \
            self.GeneralMissionParameters(name='FreeFly',
                                          drone_placement_pattern=drone_placement_pattern,
                                          isDebug=False,
                                          accomplished=False,  # The mission has not been accomplished at the beginning
                                          distance_thres=5,
                                          speed=(20/3)/self.environment.downsampling,  # Default speed for the drones,
                                          # equivalent to 1m/s
                                          #speed=(5/3)/self.environment.downsampling,
                                          num_simple_actions=6,  # Number of simple actions for the 'Random_action' mode
                                          num_people=3,
                                          num_drones=num_drones)
        self.drones = self.generate_drones()
        self.person = self.generate_people()
        self.reward = self.Reward()
        self.data_per_step = list()
        self.time_start = time()
        self.time_step = 0

        if plot_flag:
            self.fig, self.ax = plt.subplots(figsize=(30, 15))

    def get_initial_observations(self):
        observations = []
        for drone_idx in range(min(self.general_mission_parameters.num_drones, len(self.drones))):
            observations.append((self.drones[drone_idx].index,
                                 self.drones[drone_idx].mode.actual,
                                 self.drones[drone_idx].status_net,
                                 (self.drones[drone_idx].position[0], self.drones[drone_idx].position[1]),
                                 np.mod(self.drones[drone_idx].direction, 360),
                                 np.mod(self.drones[drone_idx].orientation, 360),
                                 self.drones[drone_idx].speed,
                                 []))
        return observations

    def step(self, action_id=None):
        old_total_reward = self.reward.total
        self.general_mission_parameters.action_id = action_id

        if self.environment.plot_flag:
            self.ax.clear()
            plt.imshow(self.environment.background, origin='lower')
            for i in range(4):
                plt.plot([self.environment.corners[0][-1 + i], self.environment.corners[0][i]],
                         [self.environment.corners[1][-1 + i], self.environment.corners[1][i]],
                         c='k', LineWidth=1)

        # Update drone properties
        for drone_idx in range(min(self.general_mission_parameters.num_drones, len(self.drones))):
            self.drones[drone_idx].reward = 0  # Define an individual reward for each drone
            if self.environment.plot_flag:
                self.drones[drone_idx].plot_drone_home()
            # Check if the drone is inside the cage
            boundary_check = self.drones[drone_idx].check_boundaries()
            if not boundary_check:  # If it is not in the cage, status_net goes to 0 and gets stopped
                if not self.drones[drone_idx].mode.actual == 'Off':
                    if len(self.data_per_step) == 0:
                        if self.environment.info_flag:
                            print("Drone {} is out of the KRI cage!"
                                .format(drone_idx))
                    else:
                        self.reward.total -= self.reward.cost_crash
                        self.drones[drone_idx].reward -= self.reward.cost_crash
                        if self.environment.info_flag:
                            print("Drone {} crashed against the net"
                                .format(drone_idx))
                    # If the drone is out of the range, disconnet from the net,
                    # stop it and turn of the camera
                    self.drones[drone_idx].status_net = False
                    self.drones[drone_idx].mode.actual = 'Off'
                    self.drones[drone_idx].speed = 0
                    self.drones[drone_idx].vision = 0 * self.drones[drone_idx].vision
                    self.drones[drone_idx].vision_on = False
                if self.environment.plot_flag:
                    plt.plot(self.drones[drone_idx].position[0], self.drones[drone_idx].position[1],
                             'ks', markersize=6, fillstyle='none')
            elif self.environment.plot_flag:
                # Color for the status of the drone
                self.drones[drone_idx].plot_status()

            # Speed
            if self.environment.plot_flag:
                self.drones[drone_idx].plot_velocity()

            # Angular vision
            self.drones[drone_idx].vision = np.zeros(shape=(self.environment.X_pos.shape[0],
                                                            self.environment.X_pos.shape[1]))
            if self.drones[drone_idx].vision_on:  # If the camera is on, but it can be off with some probability
                self.drones[drone_idx].vision[(self.environment.X_pos - self.drones[drone_idx].position[0]) ** 2
                                              + (self.environment.Y_pos - self.drones[drone_idx].position[1]) ** 2
                                              < self.drones[drone_idx].radius_vision ** 2] =\
                    float(((np.sign(np.random.rand(1) - self.drones[drone_idx].p_camera_off) + 1) / 2))
                if (180 <= np.mod(self.drones[drone_idx].orientation -
                                  self.drones[drone_idx].angular_vision / 2, 360)) and \
                        (np.mod(self.drones[drone_idx].orientation -
                                self.drones[drone_idx].angular_vision / 2, 360) < 360):
                    self.drones[drone_idx].vision[
                        self.environment.Y_pos > np.tan(np.deg2rad(90 + self.drones[drone_idx].orientation -
                                                                   self.drones[drone_idx].angular_vision / 2)) * (
                                self.environment.X_pos - self.drones[drone_idx].position[0]) +
                        self.drones[drone_idx].position[1]] = 0
                else:
                    self.drones[drone_idx].vision[
                        self.environment.Y_pos < np.tan(np.deg2rad(90 + self.drones[drone_idx].orientation -
                                                                   self.drones[drone_idx].angular_vision / 2)) * (
                                    self.environment.X_pos - self.drones[drone_idx].position[0]) +
                        self.drones[drone_idx].position[1]] = 0
                if (0 <= np.mod(self.drones[drone_idx].orientation +
                                self.drones[drone_idx].angular_vision / 2, 360)) and \
                        (np.mod(self.drones[drone_idx].orientation +
                                self.drones[drone_idx].angular_vision / 2, 360) < 180):
                    self.drones[drone_idx].vision[
                        self.environment.Y_pos > np.tan(np.deg2rad(90 + self.drones[drone_idx].orientation +
                                                                   self.drones[drone_idx].angular_vision / 2)) * (
                                    self.environment.X_pos - self.drones[drone_idx].position[0]) +
                        self.drones[drone_idx].position[1]] = 0
                else:
                    self.drones[drone_idx].vision[
                        self.environment.Y_pos < np.tan(np.deg2rad(90 + self.drones[drone_idx].orientation +
                                                                   self.drones[drone_idx].angular_vision / 2)) * (
                                    self.environment.X_pos - self.drones[drone_idx].position[0]) +
                        self.drones[drone_idx].position[1]] = 0
                if self.environment.plot_flag:
                    self.drones[drone_idx].plot_vision()
                if sum(sum(self.drones[drone_idx].vision)) != 0:
                    self.reward.total -= self.reward.cost_camera_use
                    self.drones[drone_idx].reward -= self.reward.cost_camera_use

        # Plotting people
        for person_idx in range(min(self.general_mission_parameters.num_people, len(self.person))):
            if not self.person[person_idx].check_boundaries():
                # If the person hits the borders, turns 180 degrees
                self.person[person_idx].orientation = self.person[person_idx].orientation + 180
            if self.environment.plot_flag:
                self.person[person_idx].plot_person()
                self.person[person_idx].plot_velocity()

        # Detection
        self.data_per_step.append([])
        for drone_idx in range(min(self.general_mission_parameters.num_drones, len(self.drones))):
            detected_objects, position_people = self.drones[drone_idx].detect_person(self.person)
            if detected_objects > 0:
                self.general_mission_parameters.position_people = position_people
                num_ppl_detected = sum((np.sign(np.random.rand(1, detected_objects) -
                                                self.drones[drone_idx].p_misdetection) + 1) / 2)[0]
                if self.environment.info_flag:
                    print("Drone {} detected {} people out of {} objects detected"
                        .format(drone_idx, int(num_ppl_detected), detected_objects))
                if num_ppl_detected > 0:
                    self.mission_update(drone_idx)
            self.data_per_step[-1].append((self.drones[drone_idx].index,
                                           self.drones[drone_idx].mode.actual,
                                           self.drones[drone_idx].status_net,
                                           (self.drones[drone_idx].position[0], self.drones[drone_idx].position[1]),
                                           np.mod(self.drones[drone_idx].direction, 360),
                                           np.mod(self.drones[drone_idx].orientation, 360),
                                           self.drones[drone_idx].speed,
                                           position_people))
        self.data_per_step[-1].append(self.reward.total)

        # Action
        for drone_idx in range(0, min(self.general_mission_parameters.num_drones, len(self.drones))):
            self.drones[drone_idx].action(self.general_mission_parameters)

        # Drone Updates with random variables
        for drone_idx in range(0, min(self.general_mission_parameters.num_drones, len(self.drones))):
            if self.drones[drone_idx].mode.actual != 'Off':  # Update only if the drone is not off
                # It disconnects with a Bernoulli(p_disconnection)
                self.drones[drone_idx].status_net = bool((np.sign(np.random.rand(1) -
                                                                  self.drones[drone_idx].p_disconnection) + 1) / 2)
                if not ((self.drones[drone_idx].mode.actual == 'Disarm') or (
                        self.drones[drone_idx].mode.actual == 'Arm')):  # If the drone is flying
                    # Same orientation plus a random from N(0,1)
                    self.drones[drone_idx].orientation += self.drones[drone_idx].std_drone * np.random.normal()
                    # Same direction plus a random from N(0,1)
                    self.drones[drone_idx].direction += self.drones[drone_idx].std_drone * np.random.normal()
                    # If the drone changed the flying mode, do not move while planning the new mode
                    if self.drones[drone_idx].mode.previous == self.drones[drone_idx].mode.actual:
                        # New position = previous position + speed/s x 1s
                        self.drones[drone_idx].position = self.drones[drone_idx].position + np.array(
                            [self.drones[drone_idx].speed * np.cos(np.deg2rad(self.drones[drone_idx].direction - 90)),
                             self.drones[drone_idx].speed * np.sin(np.deg2rad(self.drones[drone_idx].direction - 90))])
                        self.reward.total -= self.reward.cost_movement
                        self.drones[drone_idx].reward -= self.reward.cost_movement
                    self.drones[drone_idx].speed = self.drones[drone_idx].speed + \
                                                   self.drones[drone_idx].std_drone * np.random.normal(0, 1)
            self.drones[drone_idx].mode.previous = self.drones[drone_idx].mode.actual

        # People updates with random variables
        for person_idx in range(0, min(self.general_mission_parameters.num_people, len(self.person))):
            self.person[person_idx].random_walk()

        # Check if mission is done or all the drones have crashed
        all_drones_off = len([drone for drone in self.drones if drone.mode.actual == 'Off']) == len(self.drones)
        is_done = self.general_mission_parameters.accomplished or all_drones_off or \
                    self.time_step >= self.environment.max_time

        team_reward = self.reward.total - old_total_reward
        self.time_step += 1

        if self.environment.plot_flag:
            plt.title("Time step {}, Step Reward = {}".format(self.time_step, team_reward))
            plt.xlabel("Total Reward {}".format(self.reward.total))
            self.fig.canvas.draw()
            if self.time_step == 1:
                plt.pause(10.0)
            else:
                plt.pause(0.2)
            # plt.pause(max(1 / self.environment.acceleration - (time() - self.time_start), 0.1))

            if is_done:
                plt.close()

        rewards = [drone.reward for drone in self.drones]
        return self.data_per_step[-1][0:-1], rewards, is_done

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Kostas Simulator",
                                    usage='use "%(prog)s --help" for more information',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parse.add_argument('--num_drones', type=int, default=3, help="number of drones")
    parse.add_argument('--max_time',type=int, default=900, help="max running time")
    parse.add_argument('--plot_flag',type=bool, default=True, help="plotting flag")
    parse.add_argument('--info_flag', type=bool, default=True, help="info flag")
    parse.add_argument('--acceleration',type=int, default=30, help="acceleration value")
    parse.add_argument('--drone_placement_pattern',type=int, default=1,
                       help="""drone_placement_pattern:  ##\n\
                       ##0 --> Random position within the cage\n\
                       ##1 --> Distributed over one edge\n\
                       ##2 --> Distributed over all edges\n\
                       ##3 --> Starting from one corner\n\
                       ##4 --> Starting from all corners""")
    args = parse.parse_args()

    # Simutation begin
    """
    drone_placement_pattern:
        0 --> Random position within the cage
        1 --> Distributed over one edge
        2 --> Distributed over all edges
        3 --> Starting from one corner
        4 --> Starting from all corner
    """

    simulation = Simulation(num_drones=args.num_drones,
                            plot_flag=args.plot_flag,
                            info_flag=args.info_flag,
                            max_time=args.max_time,
                            drone_placement_pattern=args.drone_placement_pattern)
    print("SIMULATION STARTS")
    t = time()
    #times = 1
    print("Position of Targets")
    for person_idx in range(min(simulation.general_mission_parameters.num_people, len(simulation.person))):
        print("({})".format(simulation.person[person_idx].position))
    print('Mission when locating a person: ' + simulation.general_mission_parameters.name)
    # if simulation.environment.plot_flag:
    #     fig, ax = plt.subplots()
    #     fig.show()
    while simulation.time_step < simulation.environment.max_time and not simulation.general_mission_parameters.accomplished:
        # if simulation.environment.plot_flag:
        #     ax.clear()
        # time_start = time()
        """
        the ob(observations) is an list of tuple, which refer the observation of current time, in the format of
         (
         index,                --> index of the drone
         actual mode, 
         status_net, 
         drone position(x, y), --> tuple 
         direction, 
         orientation, 
         position_people       --> list of tuple contains people that have been detected by the drone currently
         )
         The tuple above is stand for one drone's ob, thus we have 6(the number of drone) tuples to indicate the drone's
         OB.
         the re(reward) is the step reward 
        """
        ob, re, done_flag = simulation.step(action_id=None)
        # if simulation.environment.plot_flag:
        #     plt.title("Time stamp {}, Step Reward = {}".format(times, re))
        #     plt.xlabel("Total Reward {}".format(simulation.reward.total))
        #     fig.canvas.draw()
        #     plt.pause(max(1 / simulation.environment.acceleration - (time() - time_start), 0.1))
        #times += 1
    if simulation.time_step >= simulation.environment.max_time:
        print("Drones run out of battery")

    print("Total Reward is: {}\nSIMULATION ENDS in {} seconds".format(simulation.reward.total, round(time() - t, 2)))
    #if simulation.general_mission_parameters.accomplished:
    #    print("The total reward is {}".format(simulation.reward.total))

    file = pd.DataFrame(simulation.data_per_step)
    file.to_csv('./all_data.csv', sep=',', index=False)
    print('All data has been saved in all_data.csv\nEnd')
    # Keep the plot when simulation finished
    if simulation.environment.plot_flag:
        plt.show(block=True)
