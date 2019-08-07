import numpy as np


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
