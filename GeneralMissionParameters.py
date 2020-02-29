import numpy as np


# general_mission_parameters
class GeneralMissionParameters:
    def __init__(self, name, drone_placement_pattern=0, action_id=None, isDebug=False,
                 accomplished=False, distance_thres=0, position_detected=[],
                 position_people=np.array([]), speed=0.0, num_simple_actions=0,
                 num_drones=0, num_people=0, environment=None):
        """
        distance_thres: Threshold of the distance to consider the drone is in a specific point
        Home position for the drone
        placed_pattern:
            0 --> Random position within the cage
            1 --> Distributed over one edge
            2 --> Distributed over all edges
            3 --> Starting from one corner
            4 --> Starting from all corner

        position_people - people detected at the current time step
        position_detected - people detected since the beginning of the episode
        """

        self.name = name  # Name of mission
        self.drone_placement_pattern = drone_placement_pattern
        self.mission_start_position = [[]] * num_drones
        if name == "Random_action":
            self.mission_actual = name  # Random Generation Flag
        elif name == "Raster_motion":
            self.mission_actual = "Raster_motion"
            corners = np.array(environment.corners).T
            self.mission_start_position = [
                corners[0] + np.linalg.norm(corners[0] - corners[3]) / num_drones * i * (
                        corners[3] - corners[0]) / np.linalg.norm(
                    corners[0] - corners[3]) + 3 * 6.66 / environment.downsampling for i in range(num_drones)]
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
        self.drone_home_position = np.array([])
