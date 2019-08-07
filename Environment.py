import numpy as np
import cv2
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, background, corners, downsampling=1,
                 plot_flag=False, info_flag=True, max_time=900):
        fig = np.array(plt.imread(background))
        self.background = cv2.resize(fig, (round(fig.shape[1] / downsampling), round(fig.shape[0] / downsampling)))
        self.corners = (corners / downsampling).astype(float)
        self.x_pos = range(self.background.shape[1])  # Dimensions of the picture
        self.y_pos = range(self.background.shape[0])
        self.max_x_pos = self.background.shape[1]
        self.max_y_pos = self.background.shape[0]
        self.X_pos, self.Y_pos = np.meshgrid(self.x_pos, self.y_pos)
        self.downsampling = downsampling
        self.plot_flag = plot_flag
        self.info_flag = info_flag
        self.max_time = max_time
