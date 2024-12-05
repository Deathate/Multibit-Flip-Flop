import sys

sys.path.append("hello_world/src")

import cv2
import numpy as np
from utility_image_wo_torch import *


def draw_layout(file_name, die_size, bin_width, bin_height, placement_rows):
    print(die_size)
    print(die_size.x_lower_left)
    # a = np.full((512, 512, 3), 127, np.uint8)
    # cv2.imwrite(file_name, a)
    # plot_images(a)
