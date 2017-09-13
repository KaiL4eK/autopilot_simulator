from __future__ import print_function

import cv2
import numpy as np
import math as m

#   Axis of map:
#   |y
#   |
#   |
#   ------x

class SimObject(object):
    def __init__ (self, x=0, y=0, theta=0):
        self.x      = x
        self.y      = y
        self.theta  = theta

    def get_distance_to (self, dist_object=None):
        if dist_object is None:
            return 0

        dx = dist_object.x - self.x
        dy = dist_object.y - self.y

        return m.hypot(dx, dy)

    def get_base_vectors_to (self, dist_object=None):
        if dist_object is None:
            return (0, 0)

        dx = dist_object.x - self.x
        dy = dist_object.y - self.y

        dist = m.hypot(dx, dy)

        return (dx / dist, dy / dist)

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class CircleObstacle(SimObject):
    def __init__ (self, x=0, y=0, radius=0):
        # super(self.__class__, self).__init__(x, y, 0)
        super().__init__(x, y, 0)
        self.r      = radius


class CircleTarget(SimObject):
    def __init__ (self, x=0, y=0):
        # super(self.__class__, self).__init__(x, y, 0)
        super().__init__(x, y, 0)
        self.r      = 0.01

class Robot(SimObject):
    def __init__ (self, x=0, y=0, theta=0):
        # super(self.__class__, self).__init__(x, y, theta)
        super().__init__(x, y, theta)
        self.r      = 0.56
        self.reset_speed()

    def reset_speed (self):
        self.ux     = 0     # m / sec
        self.uy     = 0     # m / sec
        self.wz     = 0     # degree / sec

    def sample_step(self, dt=0):
        self.x      += self.ux * dt * m.cos(m.radians(self.theta)) \
                        + self.uy * dt * m.cos(m.radians(self.theta + 90))
        self.y      += self.uy * dt * m.sin(m.radians(self.theta + 90)) \
                        + self.ux * dt * m.sin(m.radians(self.theta))
        self.theta  += self.wz * dt

class SimManager:
    def __init__ (self, dt=0, bot=None, target=None, obstacles=None, map_size_m=(0, 0)):
        self.bot = bot
        self.target = target
        self.obstacles = obstacles
        self.dt = dt
        self.t = 0
        self.path = []
        self.map_size = map_size_m

        self.target_dist = self.bot.get_distance_to(self.target)
        self.target_dir = self.bot.get_base_vectors_to(self.target)

    def show_map (self, resolution_m_px=1):
        width = self.map_size[0] / float(resolution_m_px)
        height = self.map_size[1] / float(resolution_m_px)

        # print('Draw map: %d / %d' % (width, height))

        img = np.ones(shape=(int(height), int(width), 3), dtype=np.uint8) * 255

        for point in self.path:
            cv2.circle(img, center=(int(point[0] / resolution_m_px), int(point[1] / resolution_m_px)), 
                            radius=1, thickness=-1, color=(0, 0, 255))

        if self.bot:
            cv2.circle(img, center=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)), 
                            radius=int(self.bot.r/resolution_m_px), 
                            thickness=-1, color=(0, 255, 0))
            cv2.line(img,   pt1=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)),
                            pt2=(int((self.bot.x + m.cos(m.radians(self.bot.theta))) / resolution_m_px), int((self.bot.y - m.sin(m.radians(self.bot.theta))) / resolution_m_px)),
                            color=(0, 255, 0),
                            thickness=1 )
            cv2.circle(img, center=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)), 
                            radius=1, thickness=-1, color=(255, 0, 0))

            (dx, dy) = self.target_dir

            cv2.line(img,   pt1=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)),
                            pt2=(int((self.bot.x + dx) / resolution_m_px), int((self.bot.y + dy) / resolution_m_px)),
                            color=(255, 0, 0),
                            thickness=1 )

        if self.target:
            cv2.circle(img, center=(int(self.target.x / resolution_m_px), int(self.target.y / resolution_m_px)), 
                            radius=int(self.target.r/resolution_m_px), 
                            thickness=-1, color=(255, 0, 0))

        cv2.imshow('2', img)
        cv2.waitKey(1)

    def sample_step (self, inputs):
        if max(inputs) > 1:
            print('Norm is incorrect %s' % inputs)
            inputs /= max(inputs)

        self.bot.ux = inputs[0] * 10
        self.bot.uy = inputs[1] * 10
        self.bot.wz = inputs[2] * 10

        self.bot.sample_step(self.dt)

        self.target_dist = self.bot.get_distance_to(self.target)
        self.target_dir = self.bot.get_base_vectors_to(self.target)

        self.path.append((self.bot.x, self.bot.y))

        self.t += self.dt

        if self.check_collision():
            return False

        if self.check_target_reached():
            return False

        return True

    def check_target_reached (self):
        if self.target_dist < self.target.r:
            return True

        return False

    def check_collision (self):
        if  self.bot.x - self.bot.r <= 0 or \
            self.bot.y - self.bot.r <= 0 or \
            self.bot.x + self.bot.r >= self.map_size[0] or \
            self.bot.y + self.bot.r >= self.map_size[1]:
            return True

        return False

    def get_fitness (self):
        return self.target_dist

    def get_state (self):
        return self.target_dir

    def process_input (self):
        key = cv2.waitKey(0) & 0xFF

        # if the 'ESC' key is pressed, Quit
        if key == 27:
            quit()
        if key == 82:
            print("up")
            return (1, 0, 0)
        elif key == 84:
            print("down")
            return (-1, 0, 0)
        elif key == 81:
            print("left")
            return (0, -1, 0)
        elif key == 83:
            print("right")
            return (0, 1, 0)
        # 255 is what the console returns when there is no key press...
        # elif key != 255:
            # print(key))
        else:
            return (0, 0, 0)

class SonarSensor:
    def __init__ (self, angle=90, distance_max=4.5, distance_min=0.1):
        self.angle = angle
        self.dist_max = distance_max
        self.dist_min = distance_min
        self.nrows    = angle / 0.1

        self.row_ranges = np.zeros(shape=(self.nrows), dtype=np.uint8)



if __name__ == '__main__':
    sim = SimManager(dt=0.005, # 200 Hz
                        bot=Robot(x=2, y=3.5, theta=0),
                        target=CircleTarget(x=8, y=3.5),
                        obstacles=[], map_size_m=(10, 7))

    while True:

        sim.show_map(resolution_m_px=0.01)
        
        inputs = sim.process_input()

        if not sim.sample_step(inputs):
            exit(1)

