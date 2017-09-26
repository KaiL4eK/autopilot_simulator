from __future__ import print_function
from sim_map import *

import numpy as np
import cv2

filename = 'two_obstacles.pmap'
resolution_m_px = 0.02

sim_map = get_map_from_file(filename)
sim_map_img = sim_map.get_image(resolution_m_px)

cv2.imshow('1', cv2.flip(sim_map_img, 0))
cv2.waitKey(0)
