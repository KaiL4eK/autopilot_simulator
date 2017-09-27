from __future__ import print_function

import xml.etree.ElementTree as etree
from common import *

import numpy as np
import numba as nb
import cv2


def check_collision (map_data, bot_radius, bot_point):
    if  bot_point.x - bot_radius <= 0 or \
        bot_point.y - bot_radius <= 0 or \
        bot_point.x + bot_radius >= map_data.width or \
        bot_point.y + bot_radius >= map_data.height:
        return True

    for obstacle in map_data.obstacles:
        # if type(obstacle) is RectObstacle:
        # https://stackoverflow.com/a/1879223
        nearest_x = np.clip(bot_point.x, obstacle.ul.x, obstacle.lr.x)
        nearest_y = np.clip(bot_point.y, obstacle.lr.y, obstacle.ul.y)

        dist_x = bot_point.x - nearest_x
        dist_y = bot_point.y - nearest_y

        if (dist_x*dist_x + dist_y*dist_y) < bot_radius**2:
            return True

    return False

sim_map_spec = [('width', nb.float32), 
                ('height', nb.float32),
                ('ul', point_type),
                ('lr', point_type),
                ('ll', point_type),
                ('map_lines', line_type[4]),
                ('obstacles', point_type)]
# @nb.jitclass(sim_map_spec)
class SimMap(object):
    def __init__(self, size):
        self.width = size[0]
        self.height = size[1]

        self.ul = Point(0, self.height)
        self.lr = Point(self.width, 0)

        self.ur = Point(self.width, self.height)
        self.ll = Point(0, 0)

        self.map_lines = [Line(self.ul, self.ur), 
                          Line(self.ur, self.lr),
                          Line(self.lr, self.ll),
                          Line(self.ll, self.ul)]

        self.obstacles = []
        self.obstacle_lines_np = None

    def get_obstacle_lines(self):
        obstacle_lines = []
        for obstacle in self.obstacles:
            obstacle_lines.extend(obstacle.lines)

        obstacle_lines.extend(self.map_lines)

        if self.obstacle_lines_np is None:
            self.obstacle_lines_np = np.zeros(shape=(len(obstacle_lines), 2, 2), dtype=np.float32)
            for idx, obstacle_line in enumerate(obstacle_lines):
                self.obstacle_lines_np[idx][0] = np.array((obstacle_line.p0.x, obstacle_line.p0.y))
                self.obstacle_lines_np[idx][1] = np.array((obstacle_line.d.x, obstacle_line.d.y))

        return self.obstacle_lines_np

    def get_image(self, resol_m_px):
        resol_m_px = float(resol_m_px)
        img_width  = self.width  / resol_m_px
        img_height = self.height / resol_m_px

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

rect_obst_spec = [('width', nb.float32), 
                  ('height', nb.float32),
                  ('x', nb.float32),
                  ('y', nb.float32),
                  ('ul', point_type),
                  ('lr', point_type),
                  ('ur', point_type),
                  ('ll', point_type),
                  ('lines', line_type[4])]
# @nb.jitclass(rect_obst_spec)
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
