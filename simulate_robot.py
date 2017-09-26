from __future__ import print_function

import time
import cv2
import numpy as np
import math as m
from common import *
from sim_map import *
from sensors import *

#   Axis of map:
#   |y
#   |
#   |
#   ------x


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

    def get_state(self):
        return State(self.x, self.y, self.theta)

    def reset_speed (self):
        self.ux     = 0     # m / sec
        self.uy     = 0     # m / sec
        self.wz     = 0     # degree / sec

    def sample_step(self, dt=0):
        new_x = self.x + self.ux * dt * m.cos(m.radians(self.theta)) \
                       - self.uy * dt * m.sin(m.radians(self.theta))

        new_y = self.y + self.uy * dt * m.cos(m.radians(self.theta)) \
                       + self.ux * dt * m.sin(m.radians(self.theta))

        new_t = self.theta + self.wz * dt

        self.x      = new_x
        self.y      = new_y
        self.theta  = new_t

        for sonar in self.sensors:
            sonar.update_base_point(x=self.x, y=self.y, theta=self.theta)
        # print(self.x, self.y, self.theta)

    def proccess_sonar_sensors(self, obstacles_lines):
        for sonar in self.sensors:
            sonar.update(obstacles_lines)

class State:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

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
        img = self.map_data.get_image(resolution_m_px)

        if self.bot:
            if check_collision(self.map_data, self.bot, self.bot.get_state()):
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

        if not self.bot.sample_step(self.dt):
            print('Collision')

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
    filename = 'two_obstacles.pmap'
    sim = SimManager(dt=0.001, # 200 Hz
                        bot=Robot(x=2, y=8, theta=0),
                        target=CircleTarget(x=18, y=5),
                        map_data=get_map_from_file(filename))

    while True:

        inputs = sim.process_input()

        step_start = time.time()
        if not sim.sample_step(inputs):
            exit(1)

        step_end = time.time()
        print("Step time:", (step_end - step_start) * 1000, "ms")

        sim.show_map(resolution_m_px=0.02)
            
