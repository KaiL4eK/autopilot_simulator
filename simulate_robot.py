from __future__ import print_function

import time
import cv2
import numpy as np
import math as m
from common import *

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


class Map:
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height

        self.ul = Point(0, height)
        self.lr = Point(width, 0)

        self.ur = Point(width, height)
        self.ll = Point(0, 0)

        self.map_lines = [Line(self.ul, self.ur), 
                          Line(self.ur, self.lr),
                          Line(self.lr, self.ll),
                          Line(self.ll, self.ul)]

        self.obstacles = [RectObstacle(ul=Point(6, 10), width=2, height=5),
                          RectObstacle(ul=Point(13, 5), width=2, height=5)]

    def get_obstacle_lines(self):
        obstacle_lines = []
        for obstacle in self.obstacles:
            obstacle_lines.extend(obstacle.lines)

        obstacle_lines.extend(self.map_lines)

        return obstacle_lines


class State:
    def __init__(self, x=0, y=0, z=0, fi=0, theta=0, psi=0):
        self.x      = x
        self.y      = y
        self.z      = z
        self.fi     = fi
        self.theta  = theta
        self.psi    = psi


class Physics:
    def __init__(self):
        pass

    def update_state(self):
        pass

class CircleObstacle(SimObject):
    def __init__ (self, x=0, y=0, radius=0):
        # super(self.__class__, self).__init__(x, y, 0)
        super().__init__(x, y, 0)
        self.r = radius

class RectObstacle(SimObject):
    def __init__ (self, ul=Point(0, 0), width=0, height=0):
        # super(self.__class__, self).__init__(x, y, 0)
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

        sensors_shift = 0.15
        self.sensors = [SonarSensor(base_dist=sensors_shift, stheta=0),  # Front
                        SonarSensor(base_dist=sensors_shift, stheta=-90),  # Right
                        SonarSensor(base_dist=sensors_shift, stheta=90),  # Left
                        SonarSensor(base_dist=sensors_shift, stheta=180)]  # Rear

        # self.sensors = [SonarSensor(base_dist=sensors_shift, stheta=0)]  # Rear

        for sonar in self.sensors:
            sonar.update_base_point(x=self.x, y=self.y, theta=self.theta)

    def reset_speed (self):
        self.ux     = 0     # m / sec
        self.uy     = 0     # m / sec
        self.wz     = 0     # degree / sec

    def sample_step(self, dt=0):
        self.x      += self.ux * dt * m.cos(m.radians(self.theta)) \
                        - self.uy * dt * m.sin(m.radians(self.theta))
        self.y      += self.uy * dt * m.cos(m.radians(self.theta)) \
                        + self.ux * dt * m.sin(m.radians(self.theta))
        self.theta  += self.wz * dt

        for sonar in self.sensors:
            sonar.update_base_point(x=self.x, y=self.y, theta=self.theta)
        # print(self.x, self.y, self.theta)

    def proccess_sonar_sensors(self, obstacles_lines):
        for sonar in self.sensors:
            sonar.update(obstacles_lines)

class SonarRay:
    def __init__(self, theta=0, max_range=1):
        self.theta      = theta
        self.max_range  = max_range
        self.range      = max_range

        self.line       = Line()
        self.ray        = Ray()

    def update_base(self, x=0, y=0, theta=0):
        self.ray.p0     = Point(x, y)
        self.ray.theta  = theta + self.theta
        self.line.from_ray(ray=self.ray, length=self.range)

    def update(self, lines):
        self.line.from_ray(ray=self.ray, length=self.max_range)
        self.range  = self.max_range
        for line in lines:
            r = self.line.intersect_line(line)
            if r > 0:
                self.range = min([r * self.max_range, self.range])

        self.line.from_ray(ray=self.ray, length=self.range)
        return self.range

class SonarSensor:
    def __init__ (self, base_dist=0, stheta=0, angle=40, distance_max=4.5, distance_min=0.02):
        
        self.base_dist = base_dist
        self.stheta = stheta

        self.angle    = angle
        self.dist_max = distance_max
        self.dist_min = distance_min

        self.rays     = []
        self.range    = 4.5

        self.base_x     = 0
        self.base_y     = 0
        self.base_theta = 0

        for i in range(int(-angle/2), int((angle/2)+1) ):
            self.rays.append(SonarRay(i, self.dist_max))

    def update_base_point (self, x=0, y=0, theta=0):
        self.base_theta = self.stheta + theta
        self.base_x = x + self.base_dist * m.cos(m.radians(self.base_theta))
        self.base_y = y + self.base_dist * m.sin(m.radians(self.base_theta))

        for ray in self.rays:
            ray.update_base(self.base_x, self.base_y, self.base_theta)

        # print(self.base_theta, self.base_x, self.base_y)

    def get_base_point (self):
        return (self.base_x, self.base_y)

    def get_range (self):
        return self.range

    def get_left_right_angles (self, theta=0):
        return (theta + self.stheta + self.angle/2, 
                theta + self.stheta - self.angle/2)

    def update(self, lines):
        self.range = self.dist_max
        for ray in self.rays:
            ray_range_m = ray.update(lines)

            self.range = min([ray_range_m, self.range])

        self.range = max([self.range, self.dist_min])


class SimManager:
    def __init__ (self, map_data, dt=0, bot=None, target=None):
        self.bot = bot
        self.target = target
        # self.obstacles = map_data.obstacles
        self.dt = dt
        self.t = 0
        self.prev_control_upd_t = 0
        self.path = []
        self.map_data = map_data

        self.target_dist = self.bot.get_distance_to(self.target)
        self.target_dir = self.bot.get_base_vectors_to(self.target)

    def show_map (self, resolution_m_px=1):
        width  = self.map_data.width  / float(resolution_m_px)
        height = self.map_data.height / float(resolution_m_px)

        # print('Draw map: %d / %d' % (width, height))

        img = np.ones(shape=(int(height), int(width), 3), dtype=np.uint8) * 255

        if self.map_data.obstacles:
            for obstacle in self.map_data.obstacles:
                if type(obstacle) is RectObstacle:
                    cv2.rectangle(img, (int(obstacle.ul.x / resolution_m_px), int(obstacle.ul.y / resolution_m_px)), 
                                       (int(obstacle.lr.x / resolution_m_px), int(obstacle.lr.y / resolution_m_px)), 
                                  color=(0, 0, 255), thickness=-1)

        if self.bot:
            if self.check_collision():
                bot_clr = (255, 255, 0)
            else:
                bot_clr = (0, 255, 0)

            cv2.circle(img, center=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)), 
                            radius=int(self.bot.r/resolution_m_px), 
                            thickness=-1, color=bot_clr)

            dir_line = Line()
            dir_line.from_ray(ray=Ray(p0=Point(self.bot.x, self.bot.y), theta=self.bot.theta))
            print("Bot position:", (self.bot.x, self.bot.y, self.bot.theta))
            # print("Line segment #1 runs from", dir_line.p0, "to", dir_line.p1)

            cv2.line(img,   pt1=(int(dir_line.p0.x / resolution_m_px), int(dir_line.p0.y / resolution_m_px)),
                            pt2=(int(dir_line.p1.x / resolution_m_px), int(dir_line.p1.y / resolution_m_px)),
                            color=(0, 255, 0),
                            thickness=1 )

            for i, sonar in enumerate(self.bot.sensors):
                sonar_x, sonar_y = sonar.get_base_point()
                cv2.circle(img, center=(int(sonar_x / resolution_m_px), int(sonar_y / resolution_m_px)), 
                                radius=2, thickness=-1, color=(0, 0, 0))

                for ray in sonar.rays:
                    cv2.line(img,   pt1=(int(ray.line.p0.x / resolution_m_px), int(ray.line.p0.y / resolution_m_px)),
                                    pt2=(int(ray.line.p1.x / resolution_m_px), int(ray.line.p1.y / resolution_m_px)),
                                    color=(0, 255 * i / len(self.bot.sensors), 0),
                                    thickness=1 )

                range = sonar.get_range()
                left_angle, right_angle = sonar.get_left_right_angles(self.bot.theta)

                cv2.ellipse(img, center=(int(sonar_x / resolution_m_px), int(sonar_y / resolution_m_px)),
                                 axes=(int(range / resolution_m_px), int(range / resolution_m_px)), angle=0, startAngle=right_angle, endAngle=left_angle, 
                                 color=(255, 255 * i / len(self.bot.sensors), 255), thickness=3)


            cv2.circle(img, center=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)), 
                            radius=2, thickness=-1, color=(255, 0, 0))

            (dx, dy) = self.target_dir

            cv2.line(img,   pt1=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)),
                            pt2=(int((self.bot.x + dx) / resolution_m_px), int((self.bot.y + dy) / resolution_m_px)),
                            color=(255, 0, 0), thickness=1 )

        if self.target:
            cv2.circle(img, center=(int(self.target.x / resolution_m_px), int(self.target.y / resolution_m_px)), 
                            radius=max(int(self.target.r/resolution_m_px), 2), 
                            thickness=-1, color=(255, 0, 0))

        for point in self.path:
            cv2.circle(img, center=(int(point[1] / resolution_m_px), int(point[2] / resolution_m_px)), 
                            radius=1, thickness=-1, color=(0, 0, 255 * point[0] / self.t))

        cv2.imshow('2', cv2.flip(img, 0))
        cv2.waitKey(30)

    def sample_step (self, inputs):
        inputs = np.clip(inputs, -1, 1)

        # if max(inputs) > 1:
            # print('Norm is incorrect %s' % inputs)
            # inputs /= max(inputs)

        self.t += self.dt

        # if self.t - self.prev_control_upd_t >= 5/1000:
        self.bot.ux = inputs[0] * 100
        self.bot.uy = inputs[1] * 100
        self.bot.wz = inputs[2] * 1000
            # self.prev_control_upd_t = self.t

        self.bot.sample_step(self.dt)
        self.bot.proccess_sonar_sensors(self.map_data.get_obstacle_lines())

        # if self.check_collision():
            # return False

        self.target_dist = self.bot.get_distance_to(self.target)
        self.target_dir = self.bot.get_base_vectors_to(self.target)

        self.path.append((self.t, self.bot.x, self.bot.y))

        return True

    def check_target_reached (self):
        if self.target_dist < self.target.r:
            return True

        return False

    def check_collision (self):
        if  self.bot.x - self.bot.r <= 0 or \
            self.bot.y - self.bot.r <= 0 or \
            self.bot.x + self.bot.r >= self.map_data.width or \
            self.bot.y + self.bot.r >= self.map_data.height:
            return True

        for obstacle in self.map_data.obstacles:
            if type(obstacle) is RectObstacle:
                # https://stackoverflow.com/a/1879223
                nearest_x = np.clip(self.bot.x, obstacle.ul.x, obstacle.lr.x)
                nearest_y = np.clip(self.bot.y, obstacle.lr.y, obstacle.ul.y)

                # print(nearest_x, nearest_y)

                dist_x = self.bot.x - nearest_x
                dist_y = self.bot.y - nearest_y

                dist_sq = dist_x**2 + dist_y**2

                if dist_sq < self.bot.r**2:
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
        if key == ord('w'):
            # print("up")
            return (1, 0, 0)
        elif key == ord('s'):
            # print("down")
            return (-1, 0, 0)
        elif key == ord('q'):
            # print("rleft")
            return (0, 0, 1)
        elif key == ord('e'):
            # print("rright")
            return (0, 0, -1)
        elif key == ord('a'):
            # print("left")
            return (0, 1, 0)
        elif key == ord('d'):
            # print("right")
            return (0, -1, 0)

        # 255 is what the console returns when there is no key press...
        # elif key != 255:
            # print(key))
        else:
            return (0, 0, 0)


if __name__ == '__main__':
    sim = SimManager(dt=0.001, # 200 Hz
                        bot=Robot(x=2, y=8, theta=0),
                        target=CircleTarget(x=18, y=5),
                        map_data=Map(width=20, height=10))

    while True:

        inputs = sim.process_input()

        step_start = time.time()
        if not sim.sample_step(inputs):
            exit(1)

        step_end = time.time()
        print("Step time:", (step_end - step_start) * 1000, "ms")

        sim.show_map(resolution_m_px=0.02)
            
