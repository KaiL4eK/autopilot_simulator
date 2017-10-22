from __future__ import print_function

import sys
sys.path.append('../')

import time
import cv2
import numpy as np
import numba as nb
import math as m
from qfs.common import *
from qfs.sim_map import *
from qfs.sensors import *

from itertools import repeat
from multiprocessing import Pool

#   Axis of map:
#   |y
#   |
#   |
#   ------x

debug = False

class Robot(object):
    def __init__ (self, x, y, theta=0):

        self.r      = 0.56
        
        self.initial_x = x

        self.air_resistance = np.array([0.25, 0.25, 0], dtype=np.float32)
        # self.force_rates = np.array([2 * 9.81, 2 * 9.81, 2 * 9.81 * 10], dtype=np.float32)
        self.force_rates = np.array([5 * 9.81, 5 * 9.81, 5 * 9.81 * 10], dtype=np.float32)
        self.forces = np.array([0., 0., 0.], dtype=np.float32)
        self.speeds = np.array([0., 0., 0.], dtype=np.float32)  # m / sec, m / sec, degree / sec
        self.state  = np.array([x, y, theta], dtype=np.float32)  # x, y, theta

        sensors_shift = 0.2
        self.sensors = [SonarSensor(base_dist=sensors_shift, stheta=0),     # Front
                        SonarSensor(base_dist=sensors_shift, stheta=90),    # Left
                        # SonarSensor(base_dist=sensors_shift, stheta=180),   # Rear
                        SonarSensor(base_dist=sensors_shift, stheta=-90)]   # Right

    def getTheta(self):
        return self.state[2]

    def getY(self):
        return self.state[1]

    def getX(self):
        return self.state[0]

    def set_control_inputs(self, inputs):
        self.forces = inputs * self.force_rates

    def get_sensors_values(self):
        return np.array([sensor.range for sensor in self.sensors]) 

    def get_state_point(self):
        return Point(self.getX(), self.getY())

    def np_get_state_point(self):
        return self.state[0:2]

    def sample_step(self, dt):
        update_state(dt, self.state, self.speeds, self.forces, self.air_resistance)

    def proccess_sonar_sensors(self, obstacles_lines):
        for sonar in self.sensors:
            sonar.update_base_point(self.getX(), self.getY(), self.getTheta())
            sonar.update(obstacles_lines)

@nb.njit(nb.void(nb.float32, nb.float32[3], nb.float32[3], nb.float32[3], nb.float32[3]))
def update_state(dt, position, speed, force, air_resistance):
    # dt, x, y, th, ux, uy, wz
    t_cos = m.cos(m.radians(position[2]))
    t_sin = m.sin(m.radians(position[2]))

    # R     = np.matrix([[t_cos, -t_sin, 0], [t_sin, t_cos, 0], [0, 0, 1]], dtype=np.float32)

    speed[0] += ((force[0] * t_cos - force[1] * t_sin) - speed[0] * air_resistance[0]) * dt
    speed[1] += ((force[0] * t_sin + force[1] * t_cos) - speed[1] * air_resistance[1]) * dt
    speed[2] += (force[2] - speed[2] * air_resistance[2]) * dt

    position[0] += speed[0] * dt
    position[1] += speed[1] * dt
    position[2] += speed[2] * dt


class SimManager:
    bot_control_period_s    = 5/1000.
    bot_sensors_period_s    = 25/1000.

    time_step = bot_control_period_s

    def __init__ (self, map_data, bot, target, save_path=True):
        self.bot = bot
        self.target = np.array(target, dtype=np.float32)

        self.dt = self.time_step
        self.bot_collision = False
        self.t = 0

        self.prev_control_upd_t = 0
        self.prev_sensors_upd_t = 0

        self.save_path = save_path
        self.path = []

        self.distances = []
        self.map_data = map_data

        self.target_dir = np_get_base_vectors_to(self.bot.np_get_state_point(), 
                                                 self.target)

        self.bot.proccess_sonar_sensors(self.map_data.get_obstacle_lines())


    def sample_step (self, inputs):
        if debug:
            step_start = time.time()

        inputs = np.clip(inputs, -1, 1).astype(np.float32)

        self.t += self.dt

        if debug:
            self.bot.set_control_inputs(inputs)
        else:
            if self.t - self.prev_control_upd_t >= self.bot_control_period_s:
                self.bot.set_control_inputs(inputs)
                self.prev_control_upd_t = self.t

        self.bot.sample_step(float(self.dt))

        if debug:
            sensors_start = time.time()
            self.bot.proccess_sonar_sensors(self.map_data.get_obstacle_lines())
            sensors_end = time.time()
        else:
            if self.t - self.prev_sensors_upd_t >= self.bot_sensors_period_s:
                self.bot.proccess_sonar_sensors(self.map_data.get_obstacle_lines())
                self.prev_sensors_upd_t = self.t           

        self.bot_collision = check_collision_np(self.map_data.get_obstacle_points(), self.map_data.size, self.bot.r, self.bot.np_get_state_point())

        self.target_dir = np_get_base_vectors_to(self.bot.np_get_state_point(), 
                                                 self.target)

        if self.save_path:
            self.path.append((self.t, self.bot.getX(), self.bot.getY()))

        self.distances.append(np_get_distance_to_x3_incr(self.bot.np_get_state_point(), 
                                                         self.target))

        if debug:
            step_end = time.time()
            print(sim.get_state())
            print(inputs)
            print(self.bot.speeds)
            print("Bot position:", (self.bot.getX(), self.bot.getY(), self.bot.getTheta()))
            bot_state_calc_t = (sensors_start - step_start) * 1000
            sensors_calc_t   = (sensors_end - sensors_start) * 1000
            other_calc_t     = (step_end - sensors_end) * 1000
            step_calc_t      = (step_end - step_start) * 1000
            print("Bot calc time:", bot_state_calc_t, "ms / Ratio:", bot_state_calc_t / step_calc_t * 100, "%" )
            print("Sensors time:", sensors_calc_t, "ms / Ratio:", sensors_calc_t / step_calc_t * 100, "%")
            print("Other time:", other_calc_t, "ms / Ratio:", other_calc_t / step_calc_t * 100, "%")
            print("Step time:", step_calc_t, "ms")

        return True

    # Try to minimize this function
    def get_fitness (self):
    
        result = np.mean(self.distances) * \
                    (1 + m.fabs(self.target[0] - self.bot.getX()) / (self.target[0] - self.bot.initial_x))

        if self.bot_collision:
            result *= 2

        return result 


    def get_state (self):   #, degrees_2_degrees(self.bot.getTheta())/180.
        return np.hstack([self.target_dir, self.bot.get_sensors_values()])

    def show_map (self, resolution_m_px=1):
        img = self.map_data.get_image(resolution_m_px)


        if self.bot_collision:
            bot_clr = (255, 255, 0)
        else:
            bot_clr = (0, 255, 0)

        cv2.circle(img, center=(int(self.bot.getX() / resolution_m_px), int(self.bot.getY() / resolution_m_px)), 
                        radius=int(self.bot.r/resolution_m_px), 
                        thickness=-1, color=bot_clr)

        dir_line = line_from_radial(base_point=self.bot.get_state_point(), theta=self.bot.getTheta())

        cv2.line(img,   pt1=(int(dir_line.p0.x / resolution_m_px), int(dir_line.p0.y / resolution_m_px)),
                        pt2=(int(dir_line.p1.x / resolution_m_px), int(dir_line.p1.y / resolution_m_px)),
                        color=(0, 255, 0),
                        thickness=1 )

        for i, sonar in enumerate(self.bot.sensors):
            sonar_x, sonar_y = sonar.base_x, sonar.base_y
            cv2.circle(img, center=(int(sonar_x / resolution_m_px), int(sonar_y / resolution_m_px)), 
                            radius=2, thickness=-1, color=(0, 0, 0))

            dist_max = 4.5

            for ray_value, ray_angle in zip(sonar.ray_values * dist_max, sonar.ray_angles):
                ray_line = line_from_radial(base_point=sonar.get_state_point(), length=ray_value, theta=sonar.base_theta + ray_angle)

                cv2.line(img,   pt1=(int(ray_line.p0.x / resolution_m_px), int(ray_line.p0.y / resolution_m_px)),
                                pt2=(int(ray_line.p1.x / resolution_m_px), int(ray_line.p1.y / resolution_m_px)),
                                color=(0, 0, 0),
                                thickness=1 )

            range = sonar.range * dist_max
            left_angle, right_angle = sonar.get_left_right_angles(self.bot.getTheta())

            cv2.ellipse(img, center=(int(sonar_x / resolution_m_px), int(sonar_y / resolution_m_px)),
                             axes=(int(range / resolution_m_px), int(range / resolution_m_px)), angle=0, startAngle=right_angle, endAngle=left_angle, 
                             color=(100, 0, 0), thickness=3)

        # cv2.circle(img, center=(int(self.bot.getX() / resolution_m_px), int(self.bot.getY() / resolution_m_px)), 
                        # radius=2, thickness=-1, color=(255, 0, 0))

        cv2.line(img,   pt1=(int(self.bot.getX() / resolution_m_px), int(self.bot.getY() / resolution_m_px)),
                        pt2=(int((self.bot.getX() + self.target_dir[0]) / resolution_m_px), int((self.bot.getY() + self.target_dir[1]) / resolution_m_px)),
                        color=(255, 0, 0), thickness=1 )

        cv2.circle(img, center=(int(self.target[0] / resolution_m_px), int(self.target[1] / resolution_m_px)), 
                        radius=3, thickness=-1, color=(255, 0, 0))

        for point in self.path:
            cv2.circle(img, center=(int(point[1] / resolution_m_px), int(point[2] / resolution_m_px)), 
                            radius=1, thickness=-1, color=(0, 0, 255 * point[0] / self.t))

        cv2.imshow('2', cv2.flip(img, 0))
        cv2.waitKey(30)

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
    debug = True
    filename = '../maps/two_obstacles.pmap'
    filename = '../maps/maze.pmap'
    sim = SimManager(bot=Robot(x=2, y=10),
                     target=[36, 2],
                     map_data=get_map_from_file(filename))

    while True:

        sim.show_map(resolution_m_px=0.03)

        inputs = sim.process_input()

        if not sim.sample_step(inputs):
            exit(1)

        
            
