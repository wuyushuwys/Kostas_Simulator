# Cost and Reward of the mission
class Reward:
    def __init__(self, total=0, person_detected=900, cost_movement=0.1,
                 cost_camera_use=0, cost_communications=0, cost_crash=100):
        self.total = total
        self.person_detected = person_detected
        self.cost_movement = cost_movement
        self.cost_camera_use = cost_camera_use
        self.cost_communications = cost_communications
        self.cost_crash = cost_crash
