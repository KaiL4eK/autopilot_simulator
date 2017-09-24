from __future__ import print_function

import cv2
import numpy as np
import math as m

#   Axis of map:
#   ------x
#   |y
#   |
#   |


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

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def get_int_tuple(self):
        print((int(self.x), int(self.y)))
        return 

class Line:
    def __init__(self, p0=Point(0, 0), p1=Point(0, 0)):
        self.p0 = p0
        self.p1 = p1
        self.d  = p1-p0

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
    def __init__ (self, x=0, y=0, width=0, height=0):
        # super(self.__class__, self).__init__(x, y, 0)
        super().__init__(x, y, 0)
        self.w = width
        self.h = height

        self.ul = Point(x - width/2, y - height/2)
        self.lr = Point(x + width/2, y + height/2)

        self.ur = Point(x + width/2, y - height/2)
        self.ll = Point(x - width/2, y + height/2)

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

        sensors_shift = 0.2
        self.sensors = [SonarSensor(base_dist=sensors_shift, stheta=0),  # Front
                        SonarSensor(base_dist=sensors_shift, stheta=-90),  # Right
                        SonarSensor(base_dist=sensors_shift, stheta=90),  # Left
                        SonarSensor(base_dist=sensors_shift, stheta=180)]  # Rear

    def reset_speed (self):
        self.ux     = 0     # m / sec
        self.uy     = 0     # m / sec
        self.wz     = 0     # degree / sec

    def sample_step(self, dt=0):
        self.x      += self.ux * dt * m.cos(m.radians(self.theta)) \
                        + self.uy * dt * m.sin(m.radians(self.theta))
        self.y      += self.uy * dt * m.cos(m.radians(self.theta)) \
                        - self.ux * dt * m.sin(m.radians(self.theta))
        self.theta  += self.wz * dt
        print(self.x, self.y, self.theta)

class SimManager:
    def __init__ (self, dt=0, bot=None, target=None, obstacles=None, map_size_m=(0, 0)):
        self.bot = bot
        self.target = target
        self.obstacles = obstacles
        self.dt = dt
        self.t = 0
        self.prev_control_upd_t = 0
        self.path = []
        self.map_size = map_size_m

        self.target_dist = self.bot.get_distance_to(self.target)
        self.target_dir = self.bot.get_base_vectors_to(self.target)

    def show_map (self, resolution_m_px=1):
        width = self.map_size[0] / float(resolution_m_px)
        height = self.map_size[1] / float(resolution_m_px)

        # print('Draw map: %d / %d' % (width, height))

        img = np.ones(shape=(int(height), int(width), 3), dtype=np.uint8) * 255

        if self.obstacles:
            for obstacle in self.obstacles:
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

            cv2.line(img,   pt1=(int(self.bot.x / resolution_m_px), int(self.bot.y / resolution_m_px)),
                            pt2=(int((self.bot.x + m.cos(m.radians(self.bot.theta))) / resolution_m_px), 
                                 int((self.bot.y - m.sin(m.radians(self.bot.theta))) / resolution_m_px)),
                            color=(0, 255, 0),
                            thickness=1 )

            for sonar in self.bot.sensors:
                sonar_x, sonar_y = sonar.get_base_point(x=self.bot.x, y=self.bot.y, theta=self.bot.theta)
                cv2.circle(img, center=(int(sonar_x / resolution_m_px), int(sonar_y / resolution_m_px)), 
                                radius=2, thickness=-1, color=(0, 0, 0))

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



        cv2.imshow('2', img)
        cv2.waitKey(1)

    def sample_step (self, inputs):
        if max(inputs) > 1:
            print('Norm is incorrect %s' % inputs)
            inputs /= max(inputs)

        self.t += self.dt

        # if self.t - self.prev_control_upd_t >= 5/1000:
        self.bot.ux = inputs[0] * 100
        self.bot.uy = inputs[1] * 100
        self.bot.wz = inputs[2] * 1000
            # self.prev_control_upd_t = self.t

        self.bot.sample_step(self.dt)

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
            self.bot.x + self.bot.r >= self.map_size[0] or \
            self.bot.y + self.bot.r >= self.map_size[1]:
            return True

        for obstacle in self.obstacles:
            if type(obstacle) is RectObstacle:
                # https://stackoverflow.com/a/1879223
                nearest_x = np.clip(self.bot.x, obstacle.ul.x, obstacle.lr.x)
                nearest_y = np.clip(self.bot.y, obstacle.ul.y, obstacle.lr.y)

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
        if key == 82:
            print("up")
            return (1, 0, 0)
        elif key == 84:
            print("down")
            return (-1, 0, 0)
        elif key == 81:
            print("left")
            return (0, 0, 1)
        elif key == 83:
            print("right")
            return (0, 0, -1)
        elif key == ord('a'):
            print("left")
            return (0, -1, 0)
        elif key == ord('d'):
            print("right")
            return (0, 1, 0)

        # 255 is what the console returns when there is no key press...
        # elif key != 255:
            # print(key))
        else:
            return (0, 0, 0)

class SonarRay:
    def __init__(self, theta=0):
        self.theta  = theta

    # def 

class SonarSensor:
    def __init__ (self, base_dist=0, stheta=0, angle=40, distance_max=4.5, distance_min=0.02):
        
        self.base_dist = base_dist
        self.stheta = stheta

        self.angle    = angle
        self.dist_max = distance_max
        self.dist_min = distance_min

        self.rays     = []

        for i in range(int(-angle/2), int((angle/2)+1) ):
            self.rays.append(SonarRay(i))

    def get_base_point (self, x=0, y=0, theta=0):
        base_x = x + self.base_dist * m.cos(m.radians(self.stheta + theta))
        base_y = y - self.base_dist * m.sin(m.radians(self.stheta + theta))

        return (base_x, base_y)

if __name__ == '__main__':
    sim = SimManager(dt=0.001, # 200 Hz
                        bot=Robot(x=2, y=5, theta=0),
                        target=CircleTarget(x=18, y=5),
                        obstacles=[RectObstacle(x=5, y=5, width=1, height=1)], map_size_m=(20, 10))

    while True:

        sim.show_map(resolution_m_px=0.02)
        
        inputs = sim.process_input()

        if not sim.sample_step(inputs):
            exit(1)

