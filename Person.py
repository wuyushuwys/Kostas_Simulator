import numpy as np
import matplotlib.pyplot as plt


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
        # Slope of the boundaries
        self.m1 = (self.corners[1][1] - self.corners[1][0]) / (self.corners[0][1] - self.corners[0][0])
        self.m2 = (self.corners[1][1] - self.corners[1][2]) / (self.corners[0][1] - self.corners[0][2])
        self.m3 = (self.corners[1][3] - self.corners[1][2]) / (self.corners[0][3] - self.corners[0][2])
        self.m4 = (self.corners[1][3] - self.corners[1][0]) / (self.corners[0][3] - self.corners[0][0])
        # generate in cage random position
        in_cage = 0
        while in_cage == 0:
            self.position = np.array([np.random.randint(min(self.corners[0]), max(self.corners[0])),
                                      np.random.randint(min(self.corners[0]) + 1 / 4 * max(self.corners[1]),
                                                        max(self.corners[1]))])
            in_cage = self.check_boundaries()
        self.orientation = orientation
        self.speed = speed
        self.max_person_speed = max_person_speed
        self.std_person = std_person


    def check_boundaries(self):
        """
        Checks if a drone is within the range of the cage of KRI
        """
        # Control if is insade the cage. The equation is control=m(x-a)
        control1 = self.m1 * (self.position[0] - self.corners[0][0]) + self.corners[1][0]  # Y must be above the line
        control2 = self.m2 * (self.position[0] - self.corners[0][2]) + self.corners[1][2]  # Y must be below the line
        control3 = self.m3 * (self.position[0] - self.corners[0][2]) + self.corners[1][2]  # Y must be below the line
        control4 = self.m4 * (self.position[0] - self.corners[0][0]) + self.corners[1][0]  # Y must be above the line

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
