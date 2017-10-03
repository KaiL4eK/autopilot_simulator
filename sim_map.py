from __future__ import print_function

import xml.etree.ElementTree as etree
from common import *

import numpy as np
import numba as nb
import cv2

@nb.njit
def clip(val, vmin, vmax):
    return max(vmin, min(val, vmax))

@nb.njit
def check_collision_np (obstacle_points, map_size, bot_radius, bot_point):
    if  bot_point[0] - bot_radius <= 0 or \
        bot_point[1] - bot_radius <= 0 or \
        bot_point[0] + bot_radius >= map_size[0] or \
        bot_point[1] + bot_radius >= map_size[1]:
        return True

    for i in nb.prange(obstacle_points.shape[0]):
        # if type(obstacle) is RectObstacle:
        # https://stackoverflow.com/a/1879223

        dist_x = bot_point[0] - clip(bot_point[0], obstacle_points[i][0], obstacle_points[i][2])
        dist_y = bot_point[1] - clip(bot_point[1], obstacle_points[i][3], obstacle_points[i][1])

        if (dist_x*dist_x + dist_y*dist_y) < bot_radius**2:
            return True

    return False


class SimMap(object):
    def __init__(self, size):
        self.size = size

        # self.width = size[0]
        # self.height = size[1]

        self.ul = Point(0, self.size[1])
        self.lr = Point(self.size[0], 0)

        self.ur = Point(self.size[0], self.size[1])
        self.ll = Point(0, 0)

        self.map_lines = [Line(self.ul, self.ur), 
                          Line(self.ur, self.lr),
                          Line(self.lr, self.ll),
                          Line(self.ll, self.ul)]

        self.obstacles = []
        
        self.obstacle_lines_np = None
        self.obstacle_points_ul_lr_np = None

    def get_obstacle_lines(self):
        obstacle_lines = []
        for obstacle in self.obstacles:
            obstacle_lines.extend(obstacle.lines)

        obstacle_lines.extend(self.map_lines)

        if self.obstacle_lines_np is None:
            self.obstacle_lines_np = np.zeros(shape=(len(obstacle_lines), 4), dtype=np.float32)
            for idx, obstacle_line in enumerate(obstacle_lines):
                self.obstacle_lines_np[idx] = obstacle_line.get_as_np_array_p0_d()

                 # = np.array((obstacle_line.p0.x, obstacle_line.p0.y))
                # self.obstacle_lines_np[idx][1] = np.array((obstacle_line.d.x, obstacle_line.d.y))

        return self.obstacle_lines_np

    def get_obstacle_points(self):
        if self.obstacle_points_ul_lr_np is None:
            self.obstacle_points_ul_lr_np = np.zeros(shape=(len(self.obstacles), 4), dtype=np.float32)
            for idx, obstacle in enumerate(self.obstacles):
                self.obstacle_points_ul_lr_np[idx] = obstacle.get_obstacle_np_ul_lr()

        return self.obstacle_points_ul_lr_np
                    

    def get_image(self, resol_m_px):
        resol_m_px = float(resol_m_px)
        img_width  = self.size[0]  / resol_m_px
        img_height = self.size[1] / resol_m_px

        img = np.ones(shape=(int(img_height), int(img_width), 3), dtype=np.uint8) * 255

        if self.obstacles:
            for obstacle in self.obstacles:
                cv2.rectangle(img, (int(obstacle.ul.x / resol_m_px), int(obstacle.ul.y / resol_m_px)), 
                                   (int(obstacle.lr.x / resol_m_px), int(obstacle.lr.y / resol_m_px)), 
                              color=(0, 0, 255), thickness=-1)

        return img


# class CircleObstacle(SimObject):
#     def __init__ (self, x=0, y=0, radius=0):
#         # super(self.__class__, self).__init__(x, y, 0)
#         super().__init__(x, y, 0)
#         self.r = radius

class RectObstacle(object):
    def __init__ (self, ul=Point(0, 0), size=(0, 0)):

        self.width = size[0]
        self.height = size[1]

        self.x = ul.x + self.width/2
        self.y = ul.y - self.height/2

        self.ul = Point(self.x - self.width/2, self.y + self.height/2)
        self.lr = Point(self.x + self.width/2, self.y - self.height/2)

        self.ur = Point(self.x + self.width/2, self.y + self.height/2)
        self.ll = Point(self.x - self.width/2, self.y - self.height/2)

        self.lines = np.array([ Line(self.ul, self.ur), 
                                Line(self.ur, self.lr),
                                Line(self.lr, self.ll),
                                Line(self.ll, self.ul)], dtype=object)

    def get_obstacle_np_ul_lr(self):
        return np.array([self.ul.x, self.ul.y, self.lr.x, self.lr.y], dtype=np.float32)

def get_map_from_file(filename):

    tree = etree.parse(filename)
    root = tree.getroot()    

    for child in root:
        if child.tag == 'size':
            map_size = (float(child.attrib['width']), float(child.attrib['height']))
            new_map = SimMap(size=map_size)

        if child.tag == 'obstacles':
            for obstacle in child:
                if obstacle.tag == 'rectangle':
                    ul = (float(obstacle.attrib['ul_x']), float(obstacle.attrib['ul_y']))
                    obst_size = (float(obstacle.attrib['width']), float(obstacle.attrib['height']))
                    # print(ul)
                    new_map.obstacles.append(RectObstacle(ul=Point(ul[0], ul[1]), size=obst_size))

    return new_map
