from __future__ import print_function

import time
import cv2
import numpy as np
import numba as nb
import math as m
from common import *
from sim_map import *
from sensors import *

from itertools import repeat
from multiprocessing import Pool

#   Axis of map:
#   |y
#   |
#   |
#   ------x

debug = False

circle_target_spec = [('x', nb.float32), 
                      ('y', nb.float32),
                      ('r', nb.float32)]
@nb.jitclass(circle_target_spec)
class CircleTarget(object):
    def __init__ (self, x=0, y=0):
        self.x  = x
        self.y  = y
        self.r  = 0.01

    def get_state_point(self):
        return Point(self.x, self.y)

class Robot(object):
    def __init__ (self, x, y, theta):

        self.x = x
        self.y = y
        self.theta = theta

        self.r      = 0.56
        
        self.ux     = 0.0     # m / sec
        self.uy     = 0.0     # m / sec
        self.wz     = 0.0     # degree / sec

        self.ax     = 0.0
        self.ay     = 0.0
        self.eps_z  = 0.0

        sensors_shift = 0.2
        self.sensors = [SonarSensor(base_dist=sensors_shift, stheta=0),     # Front
                        SonarSensor(base_dist=sensors_shift, stheta=90),    # Left
                        # SonarSensor(base_dist=sensors_shift, stheta=180),   # Rear
                        SonarSensor(base_dist=sensors_shift, stheta=-90)]   # Right

    def set_control_inputs(self, inputs):
        # inputs = inputs * 2 * 9.81

        if debug:
            # self.ax = inputs[0]
            # self.ay = inputs[1]
            # self.eps_z = inputs[2]

            self.ux = inputs[0] * 100
            self.uy = inputs[1] * 100
            self.wz = inputs[2] * 1000
        else:
            # self.ax = inputs[0]
            # self.ay = inputs[1]
            # self.eps_z = inputs[2]
            self.ux = inputs[0] * 10
            self.uy = inputs[1] * 10
            self.wz = inputs[2] * 100

    def get_sensors_values(self):
        return np.array([sensor.range for sensor in self.sensors]) 

    def get_state_point(self):
        return Point(self.x, self.y)

    def get_state_tuple(self):
        return (self.x, self.y, self.theta)

    def sample_step(self, dt=0):
        self.ux += self.ax * dt;
        self.uy += self.ay * dt;
        self.wz += self.eps_z * dt;

        state = np.ndarray(shape=3, dtype=np.float32)
        self.t_cos = m.cos(m.radians(self.theta))
        self.t_sin = m.sin(m.radians(self.theta))

        state[0] = self.x + self.ux * dt * self.t_cos \
                       - self.uy * dt * self.t_sin

        state[1] = self.y + self.uy * dt * self.t_cos \
                       + self.ux * dt * self.t_sin

        state[2] = self.theta + self.wz * dt

        # state = get_new_state(np.array([dt, self.x, self.y, self.theta, self.ux, self.uy, self.wz], dtype=np.float32))

        self.x      = state[0]
        self.y      = state[1]
        self.theta  = state[2]

    def proccess_sonar_sensors(self, obstacles_lines):
        for sonar in self.sensors:
            sonar.update_base_point(self.x, self.y, self.theta)
            sonar.update(obstacles_lines)

        # with Pool() as pool:
            # pool.starmap(sonar_update, zip(self.sensors, repeat(obstacles_lines)))

@nb.njit(nb.float32[3](nb.float32[7]))
def get_new_state(data):
    # dt, x, y, t, ux, uy, wz
    t_cos = m.cos(m.radians(data[4]))
    t_sin = m.sin(m.radians(data[4]))

    new_x = data[1] + data[4] * data[0] * t_cos \
                    - data[5] * data[0] * t_sin

    new_y = data[2] + data[5] * data[0] * t_cos \
                    + data[4] * data[0] * t_sin

    new_t = data[3] + data[6] * data[0]

    return np.array([new_x, new_y, new_t])

class SimManager:
    def __init__ (self, map_data, dt=0, bot=None, target=None):
        self.bot = bot
        self.target = target

        self.dt = dt
        self.bot_collision = False
        self.t = 0

        self.prev_control_upd_t = 0
        self.prev_sensors_upd_t = 0

        self.path = []
        self.map_data = map_data

        self.target_dir = get_base_vectors_to(self.bot.get_state_point(), 
                                              self.target.get_state_point())

        self.bot.proccess_sonar_sensors(self.map_data.get_obstacle_lines())


    def sample_step (self, inputs):
        if debug:
            step_start = time.time()

        inputs = np.clip(inputs, -1, 1)

        self.t += self.dt

        if debug:
            self.bot.set_control_inputs(inputs)
        else:
            if self.t - self.prev_control_upd_t >= 5/1000:
                self.bot.set_control_inputs(inputs)
                self.prev_control_upd_t = self.t

        self.bot.sample_step(self.dt)

        if debug:
            sensors_start = time.time()
            self.bot.proccess_sonar_sensors(self.map_data.get_obstacle_lines())
            sensors_end = time.time()
        else:
            if self.t - self.prev_sensors_upd_t >= 25/1000:
                self.bot.proccess_sonar_sensors(self.map_data.get_obstacle_lines())
                self.prev_sensors_upd_t = self.t           

        # self.bot_collision = check_collision(self.map_data, self.bot.r, self.bot.get_state_point())
        self.bot_collision = check_collision_np(self.map_data.get_obstacle_points(), self.map_data.size, self.bot.r, self.bot.get_state_tuple())

        self.target_dir = get_base_vectors_to(self.bot.get_state_point(), 
                                              self.target.get_state_point())

        self.path.append((self.t, self.bot.x, self.bot.y))

        if debug:
            step_end = time.time()
            print(sim.get_state())
            # print("Bot position:", (self.bot.x, self.bot.y, self.bot.theta))
            bot_state_calc_t = (sensors_start - step_start) * 1000
            sensors_calc_t   = (sensors_end - sensors_start) * 1000
            other_calc_t     = (step_end - sensors_end) * 1000
            step_calc_t      = (step_end - step_start) * 1000
            print("Bot calc time:", bot_state_calc_t, "ms / Ratio:", bot_state_calc_t / step_calc_t * 100, "%" )
            print("Sensors time:", sensors_calc_t, "ms / Ratio:", sensors_calc_t / step_calc_t * 100, "%")
            print("Other time:", other_calc_t, "ms / Ratio:", other_calc_t / step_calc_t * 100, "%")
            print("Step time:", step_calc_t, "ms")

        return True

    def get_fitness (self):
        result = get_distance_to(self.bot.get_state_point(), 
                                 self.target.get_state_point())

        if self.bot_collision:
            result *= 5

        return result

    def get_state (self):
        # pass
        return np.hstack([self.target_dir, self.bot.get_sensors_values()])#, to_radians(self.bot.theta)])


    def show_map (self, resolution_m_px=1):
        img = self.map_data.get_image(resolution_m_px)

        if self.bot:
            if self.bot_collision:
                bot_clr = (255, 255, 0)
            else:
                bot_clr = (0, 255, 0)

            cv2.circle(img, center=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)), 
                            radius=int(self.bot.r/resolution_m_px), 
                            thickness=-1, color=bot_clr)


            dir_line = line_from_radial(base_point=self.bot.get_state_point(), theta=self.bot.theta)

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
                left_angle, right_angle = sonar.get_left_right_angles(self.bot.theta)

                cv2.ellipse(img, center=(int(sonar_x / resolution_m_px), int(sonar_y / resolution_m_px)),
                                 axes=(int(range / resolution_m_px), int(range / resolution_m_px)), angle=0, startAngle=right_angle, endAngle=left_angle, 
                                 color=(100, 0, 0), thickness=3)

            # cv2.circle(img, center=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)), 
                            # radius=2, thickness=-1, color=(255, 0, 0))


        if self.target:
            target_vect = Point(self.target_dir[0], self.target_dir[1])

            cv2.line(img,   pt1=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)),
                            pt2=(int((self.bot.x + target_vect.x) / resolution_m_px), int((self.bot.y + target_vect.y) / resolution_m_px)),
                            color=(255, 0, 0), thickness=1 )

            cv2.circle(img, center=(int(self.target.x / resolution_m_px), int(self.target.y / resolution_m_px)), 
                            radius=max(int(self.target.r/resolution_m_px), 2), 
                            thickness=-1, color=(255, 0, 0))

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
    filename = 'two_obstacles.pmap'
    filename = 'maze.pmap'
    sim = SimManager(dt=0.001, # 200 Hz
                        bot=Robot(x=2, y=10, theta=0),
                        target=CircleTarget(x=36, y=2),
                        map_data=get_map_from_file(filename))

    while True:

        sim.show_map(resolution_m_px=0.02)

        inputs = sim.process_input()

        if not sim.sample_step(inputs):
            exit(1)

        
            
