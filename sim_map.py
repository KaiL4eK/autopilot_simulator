from __future__ import print_function

import xml.etree.ElementTree as etree
from common import *

import numpy as np
import cv2

class SimMap:
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

    def get_obstacle_lines(self):
        obstacle_lines = []
        for obstacle in self.obstacles:
            obstacle_lines.extend(obstacle.lines)

        obstacle_lines.extend(self.map_lines)

        return obstacle_lines

    def get_image(self, resol_m_px):
        resol_m_px = float(resol_m_px)
        img_width  = self.width  / resol_m_px
        img_height = self.height / resol_m_px

        img = np.ones(shape=(int(img_height), int(img_width), 3), dtype=np.uint8) * 255

        if self.obstacles:
            for obstacle in self.obstacles:
                if type(obstacle) is RectObstacle:
                    cv2.rectangle(img, (int(obstacle.ul.x / resol_m_px), int(obstacle.ul.y / resol_m_px)), 
                                       (int(obstacle.lr.x / resol_m_px), int(obstacle.lr.y / resol_m_px)), 
                                  color=(0, 0, 255), thickness=-1)

        return img


class CircleObstacle(SimObject):
    def __init__ (self, x=0, y=0, radius=0):
        # super(self.__class__, self).__init__(x, y, 0)
        super().__init__(x, y, 0)
        self.r = radius

class RectObstacle(SimObject):
    def __init__ (self, ul=Point(0, 0), size=(0, 0)):
        # super(self.__class__, self).__init__(x, y, 0)
        width = size[0]
        height = size[1]

        x = ul.x + width/2
        y = ul.y - height/2

        super().__init__(x, y, 0)
        self.w = width
        self.h = height

        self.ul = Point(x - width/2, y + height/2)
        self.lr = Point(x + width/2, y - height/2)

        self.ur = Point(x + width/2, y + height/2)
        self.ll = Point(x - width/2, y - height/2)

        # print(self.x, self.y)
        # print(self.ul, self.lr)

        self.lines = [Line(self.ul, self.ur), 
                      Line(self.ur, self.lr),
                      Line(self.lr, self.ll),
                      Line(self.ll, self.ul)]

def get_map_from_file(filename):

    tree = etree.parse(filename)
    root = tree.getroot()    

    for child in root:
        # print(child.tag)

        if child.tag == 'size':
            map_size = (int(child.attrib['width']), int(child.attrib['height']))
            new_map = SimMap(size=map_size)

        if child.tag == 'obstacles':
            for obstacle in child:
                if obstacle.tag == 'rectangle':
                    ul = (int(obstacle.attrib['ul_x']), int(obstacle.attrib['ul_y']))
                    obst_size = (int(obstacle.attrib['width']), int(obstacle.attrib['height']))
                    # print(ul)
                    new_map.obstacles.append(RectObstacle(ul=Point(ul[0], ul[1]), size=obst_size))

    return new_map
