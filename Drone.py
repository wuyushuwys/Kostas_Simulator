import numpy as np
import matplotlib.pyplot as plt


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
            % Others, to be built
            """
            self.previous = previous
            self.actual = actual
            self.parameters_destination = np.array(parameters_destination)
            self.parameters_detection = parameters_detection

    def __init__(self, placed_pattern=0, dowmsampling=6, index=0, status_net=True, mode=Mode(),
                 home=np.array([]), orientation=0,
                 speed=0.0, vision=np.array([]), vision_on=True, corners=np.array([]),
                 radius_vision=0.0, angular_vision=0.0, std_drone=0.0,
                 p_disconnection=0.0, p_misdetection=0, p_package_lost=0.05, p_camera_off=0.0):
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
        distance: Array of disrances [Bottom, Right, Top, Left ]
        """
        self.index = index
        self.status_net = status_net
        self.mode = mode
        self.corners = corners
        # Slope of the boundaries
        self.m1 = (self.corners[1][1] - self.corners[1][0]) / (self.corners[0][1] - self.corners[0][0])  # Bottom
        self.m2 = (self.corners[1][1] - self.corners[1][2]) / (self.corners[0][1] - self.corners[0][2])  # Right
        self.m3 = (self.corners[1][3] - self.corners[1][2]) / (self.corners[0][3] - self.corners[0][2])  # Top
        self.m4 = (self.corners[1][3] - self.corners[1][0]) / (self.corners[0][3] - self.corners[0][0])  # Left
        self.k_array = [self.m1, self.m2, self.m3, self.m4]
        if len(home)==0:
            self.home_position(placed_pattern, dowmsampling)
        else:
            self.home = np.array(home)
        self.position = self.home
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
        self.distance = np.array([])
        self.get_distance()

    def home_position(self, placed_pattern, downsampling):
        """
        Home position for the drone
        :param placed_pattern:
            0 --> Random position within the cage
            1 --> Distributed over one edge
            2 --> Distributed over all edges
            3 --> Starting from one corner
            4 --> Starting from all corner
        :param downsampling: down-sampling value for home margin
        :return self.home : home position of drone
        """
        home_margin = 6.66 / downsampling
        if placed_pattern == 0:
            in_cage = 0.0
            while in_cage == 0.0:
                self.home = np.array([np.random.randint(min(self.corners[0]), max(self.corners[0])),
                                      np.random.randint(min(self.corners[0]), max(self.corners[1]))])
                in_cage = self.check_boundaries(is_home=True)
                self.orientation = np.random.randint(360)
        elif placed_pattern == 1:  # Distributed over one edge (bottom edge)
            aux = np.random.randint(round(self.corners[0][1] - self.corners[0][0] - 2 * home_margin)) + \
                  self.corners[0][0] + home_margin
            self.home = np.array([aux,
                                  (self.corners[1][1] - self.corners[1][0]) / (
                                              self.corners[0][1] - self.corners[0][0]) *
                                  (aux - self.corners[0][0]) + self.corners[1][0] + home_margin])
            self.orientation = 180
        elif placed_pattern == 2:  # Distributed over all edges randomly
            edge = np.random.randint(4)
            if edge == 0:  # bottom
                aux = np.random.randint(round(self.corners[0][1] - self.corners[0][0] - 2 * home_margin)) + \
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
                                      (aux - self.corners[0][2] + home_margin) + self.corners[1][2]])
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
                self.home = np.array([self.corners[0][3] + 1.5 * home_margin,
                                      self.corners[1][3] - home_margin])
                self.orientation = 45

    def get_distance(self):
        self.distance = [abs(self.k_array[i]*(self.corners[0][i]-self.position[0])-(self.corners[1][i]-self.position[1]))
                         /np.sqrt(self.k_array[i]**2+1) for i in range(len(self.k_array))]

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
        if not is_home:
            pos = [self.position[0], self.position[1]]
        else:
            pos = [self.home[0], self.home[1]]

        # Control if is insade the cage. The equation is control=m(x-a)
        control1 = self.m1 * (pos[0] - self.corners[0][0]) + self.corners[1][0]  # Y must be above the line
        control2 = self.m2 * (pos[0] - self.corners[0][2]) + self.corners[1][2]  # Y must be below the line
        control3 = self.m3 * (pos[0] - self.corners[0][2]) + self.corners[1][2]  # Y must be below the line
        control4 = self.m4 * (pos[0] - self.corners[0][0]) + self.corners[1][0]  # Y must be above the line

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

    def detect_person(self, people):
        """
        :param people: person class with people position
        :return:
            detected: number of objects have been detected
            pos_detected: the position of object has been detected
        """
        detected = 0
        pos_detected = []
        for person in people:
            y_idx, x_idx = np.nonzero(self.vision)
            if (round(person.position[0]) in x_idx) and (round(person.position[1]) in y_idx)\
                    and (person.detected is False):
                detected += 1
                pos_detected.append((person.position[0],
                                     person.position[1]))
                # person.detected = True
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
            # if mission_parameters.isDebug:
            #    drone_action_id = np.random.randint(mission_parameters.num_simple_actions)
            # else:
            #    drone_action_id = mission_parameters.action_id[self.index]
            if mission_parameters.action_id is None:
                drone_action_id = np.random.randint(mission_parameters.num_simple_actions)
            else:
                drone_action_id = mission_parameters.action_id[self.index]
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
