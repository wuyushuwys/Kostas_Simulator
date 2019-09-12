#!/usr/bin/env/python3
import numpy as np
import matplotlib.pyplot as plt
from time import time
import argparse
import os.path
from Environment import Environment
from GeneralMissionParameters import GeneralMissionParameters
from Reward import Reward
from Person import Person
from Drone import Drone


class Simulation:

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
            # self.reward.total += self.reward.person_detected
            # self.drones[drone_idx].reward += self.reward.person_detected

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
                    self.general_mission_parameters.position_detected.\
                        append(self.general_mission_parameters.position_people[idx_ppl])
                    if self.environment.info_flag:
                        print("One person was detected at position: {}, for a total of {} people detected."
                              .format(self.general_mission_parameters.position_people[idx_ppl],
                                      len(self.general_mission_parameters.position_detected)))
                    # self.reward.total += self.reward.person_detected
                    # self.drones[drone_idx].reward += self.reward.person_detected
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
        drones = [Drone(dowmsampling=self.environment.downsampling, index=i, status_net=True,
                        placed_pattern=self.general_mission_parameters.drone_placement_pattern,
                        mode=Drone.Mode(previous='FreeFly',
                                        actual=self.general_mission_parameters.mission_actual,
                                        parameters_destination=np.array([])),
                        speed=self.general_mission_parameters.speed,
                        vision=np.zeros(shape=(self.environment.X_pos.shape[0],
                                               self.environment.X_pos.shape[1])),
                        radius_vision=(10*20/3)/self.environment.downsampling,  # Radius for vision (pixels)
                        angular_vision=60,  # Degrees of vision (<180)
                        std_drone_speed=0.1,  # Standard deviation for the speed of the drone
                        std_drone_orientation=0.1,  # Standard deviation for the orientation of the drone
                        std_drone_direction=0.1,  # Standard deviation for the direction of the drone
                        vision_on=True, corners=self.environment.corners)
                  for i in range(self.general_mission_parameters.num_drones)]
        return drones

    def generate_people(self, max_person_speed=20/3, position=[]):
        """
         Create the target structures
        :param position: generate certain position for each people
        :param max_person_speed: max speed of target
        :return: list of person
        """
        if len(position)==0:
            people = [Person(index=i, orientation=0, speed=0, max_person_speed=max_person_speed,
                             corners=self.environment.corners, std_person=0)
                  for i in range(self.general_mission_parameters.num_people)]
        else:
            people = [Person(index=i, orientation=0, speed=0, max_person_speed=max_person_speed,
                             corners=self.environment.corners, std_person=0, position = position[i])
                  for i in range(self.general_mission_parameters.num_people)]
        return people

    def __init__(self, mission_name='FreeFly', num_drones=6, num_people=3, person_position=[], plot_flag=False, info_flag=True,
                 downsampling=6, max_time=900, drone_placement_pattern=0):
        """
        :param plot_flag:
        :param downsampling:
        :param max_time:
        :param drone_placement_pattern:
                0 --> Random position within the cage
                1 --> Distributed over one edge
                2 --> Distributed over all edges
                3 --> Starting from one corner
                4 --> Starting from all corner
        """
        my_path = os.path.abspath(os.path.dirname(__file__))

        self.environment = Environment(os.path.join(my_path, "./Kostas Research Center 2.png"),
                                       corners=np.array([[111, 408, 300, 7], [43, 127, 517, 435]]),
                                       downsampling=downsampling,              # downsampling parameter
                                       plot_flag=plot_flag,                    # plot flag
                                       info_flag=info_flag,                    # Info flag
                                       max_time=max_time)                      # max running time
        self.general_mission_parameters = \
            GeneralMissionParameters(name=mission_name,
                                     drone_placement_pattern=drone_placement_pattern,
                                     isDebug=False,
                                     accomplished=False,  # The mission has not been accomplished at the beginning
                                     distance_thres=5,
                                     speed=(20/3)/self.environment.downsampling,  # Default speed for the drones,
                                                        # equivalent to 1m/s
                                                        #speed=(5/3)/self.environment.downsampling,
                                     num_simple_actions=6,  # Number of simple actions for the 'Random_action' mode
                                     num_people=num_people,
                                     num_drones=num_drones)
        self.drones = self.generate_drones()
        self.person = self.generate_people(position=person_position)
        self.reward = Reward()
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
                                 self.drones[drone_idx].distance,
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
                    if self.time_step == 0:
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
        observations = []
        for drone_idx in range(min(self.general_mission_parameters.num_drones, len(self.drones))):
            detected_objects, position_people = self.drones[drone_idx].detect_person(self.person)
            if detected_objects > 0:
                detection_distribution = ((np.sign(np.random.rand(1, detected_objects) -
                                                   self.drones[drone_idx].p_misdetection) + 1) / 2)[0]
                num_ppl_detected = np.sum(detection_distribution)
                true_detected_people = []
                for i in range(len(detection_distribution)):
                    if detection_distribution[i]==1:
                        true_detected_people.append(position_people[i])
                self.general_mission_parameters.position_people = true_detected_people
                # self.general_mission_parameters.position_people = position_people[np.nonzero(detection_distribution)]
                if self.environment.info_flag:
                    print("Drone {} detected {} people out of {} objects detected"
                        .format(drone_idx, int(num_ppl_detected), detected_objects))
                if num_ppl_detected > 0:
                    for person in self.person:
                        if tuple(person.position) in true_detected_people and person.detected is False:
                            self.reward.total += self.reward.person_detected
                            self.drones[drone_idx].reward += self.reward.person_detected
                            person.detected = True
                    self.mission_update(drone_idx)
            self.drones[drone_idx].get_distance()
            observations.append((self.drones[drone_idx].index,
                                 self.drones[drone_idx].mode.actual,
                                 self.drones[drone_idx].status_net,
                                 (self.drones[drone_idx].position[0], self.drones[drone_idx].position[1]),
                                 np.mod(self.drones[drone_idx].direction, 360),
                                 np.mod(self.drones[drone_idx].orientation, 360),
                                 self.drones[drone_idx].speed,
                                 self.drones[drone_idx].distance,
                                 self.general_mission_parameters.position_people))

        # Action
        for drone_idx in range(0, min(self.general_mission_parameters.num_drones, len(self.drones))):
            if self.drones[drone_idx].mode.actual != "Off":
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
                    self.drones[drone_idx].orientation += self.drones[drone_idx].std_drone_orientation * np.random.normal()
                    # Same direction plus a random from N(0,1)
                    self.drones[drone_idx].direction += self.drones[drone_idx].std_drone_direction * np.random.normal()
                    # If the drone changed the flying mode, do not move while planning the new mode
                    if self.drones[drone_idx].mode.previous == self.drones[drone_idx].mode.actual:
                        # New position = previous position + speed/s x 1s
                        self.drones[drone_idx].position = self.drones[drone_idx].position + np.array(
                            [self.drones[drone_idx].speed * np.cos(np.deg2rad(self.drones[drone_idx].direction - 90)),
                             self.drones[drone_idx].speed * np.sin(np.deg2rad(self.drones[drone_idx].direction - 90))])
                        self.reward.total -= self.reward.cost_movement
                        self.drones[drone_idx].reward -= self.reward.cost_movement
                    self.drones[drone_idx].speed = self.drones[drone_idx].speed + \
                                                   self.drones[drone_idx].std_drone_speed * np.random.normal(0, 1)
            self.drones[drone_idx].mode.previous = self.drones[drone_idx].mode.actual

        # People updates with random variables
        for person_idx in range(0, min(self.general_mission_parameters.num_people, len(self.person))):
            self.person[person_idx].random_walk()

        # Check if mission is done or all theself.general_mission_parameters.position_people =  drones have crashed
        is_done = self.is_mission_done()

        team_reward = self.reward.total - old_total_reward
        self.time_step += 1

        if self.environment.plot_flag:
            for person in self.person:
                if person.detected is True:
                    plt.plot(person.position[0], person.position[1],
                             'ro', markersize=3, markeredgewidth=3, fillstyle='none')
            plt.title("Time step {}, Step Reward = {}".format(self.time_step, team_reward))
            plt.xlabel("Total Reward {}".format(self.reward.total))
            self.fig.canvas.draw()
            if self.time_step == 1:
                plt.pause(0.1)
            else:
                plt.pause(0.001)
            if is_done:
                plt.close()

        # Provide both individual rewards and the team reward
        rewards = [drone.reward for drone in self.drones]
        #rewards.append(team_reward)
        rewards.append(0)
        return observations, rewards, is_done

    def is_mission_done(self):
        all_drones_off = len([drone for drone in self.drones if drone.mode.actual == 'Off']) == len(self.drones)
        is_done = self.general_mission_parameters.accomplished or all_drones_off or \
                    self.time_step >= self.environment.max_time
        return is_done

    def render(self):
        if not hasattr(self, 'fig'):
            # Generate a new plot
            self.fig = plt.figure(figsize=(30, 15))

        # Draw the background
        plt.clf()
        plt.imshow(self.environment.background, origin='lower')
        for i in range(4):
            plt.plot([self.environment.corners[0][-1 + i], self.environment.corners[0][i]],
                     [self.environment.corners[1][-1 + i], self.environment.corners[1][i]],
                     c='k', LineWidth=1)

        # Draw the people
        for person_idx in range(len(self.person)):
            self.person[person_idx].plot_person()
            self.person[person_idx].plot_velocity()

        # Draw the drones
        for drone_idx in range(0, len(self.drones)):
            self.drones[drone_idx].plot_drone_home()
            boundary_check = self.drones[drone_idx].check_boundaries()
            if not boundary_check:
                plt.plot(self.drones[drone_idx].position[0], self.drones[drone_idx].position[1],
                         'ks', markersize=6, fillstyle='none')
            else:
                # Color for the status of the drone
                self.drones[drone_idx].plot_status()
            self.drones[drone_idx].plot_velocity()
            self.drones[drone_idx].plot_vision()

        plt.title("Time step {}".format(self.time_step), fontsize=16)
        plt.xlabel("Total Reward {:.3f}".format(self.reward.total), fontsize=16)

        if self.time_step == 0:
            plt.pause(5.0)
        else:
            plt.pause(0.001)

        if self.is_mission_done():
            plt.close()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Kostas Simulator",
                                    usage='use "%(prog)s --help" for more information',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parse.add_argument('--num_drones', type=int, default=6, help="number of drones")
    parse.add_argument('--max_time', type=int, default=900, help="max running time")
    parse.add_argument('--plot_flag', type=str, default='True', help="plotting flag")
    parse.add_argument('--info_flag', type=str, default='True', help="info flag")
    parse.add_argument('--drone_placement_pattern',type=int, default=0,
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

    simulation = Simulation(mission_name='Random_action',
                            num_drones=args.num_drones,
                            num_people=6,
                            plot_flag=eval(args.plot_flag),
                            info_flag=eval(args.info_flag),
                            max_time=args.max_time,
                            drone_placement_pattern=args.drone_placement_pattern)
    print("SIMULATION STARTS")
    t = time()
    print("Position of Targets")
    for person_idx in range(min(simulation.general_mission_parameters.num_people, len(simulation.person))):
        print("({})".format(simulation.person[person_idx].position))
    print('Mission when locating a person: ' + simulation.general_mission_parameters.name)

    while simulation.time_step < simulation.environment.max_time and not simulation.general_mission_parameters.accomplished:
        """
        the ob(observations) is an list of tuple, which refer the observation of current time, in the format of
         (
         index,                --> index of the drone
         actual mode, 
         status_net, 
         drone position(x, y), --> tuple 
         direction, 
         orientation, 
         distance              --> An array of disrances [Bottom, Right, Top, Left ] in the smae scale as the dowmsampled image.
         position_people       --> list of tuple contains people that have been detected by the drone currently
         )
         The tuple above is stand for one drone's ob, thus we have 6(the number of drone) tuples to indicate the drone's
         OB.
         the re(reward) is the step reward 
        """

        ob, re, done_flag = simulation.step()
        if done_flag:
            break
    if simulation.time_step >= simulation.environment.max_time:
        print("Drones run out of battery")

    print("Total Reward is: {}\nSIMULATION ENDS in {} seconds".format(simulation.reward.total, round(time() - t, 2)))
    if simulation.environment.plot_flag:
        plt.show(block=True)
