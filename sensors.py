from common import *
import numba as nmb

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
    def __init__ (self, base_dist=0, stheta=0, angle=40, distance_max=4.5, distance_min=0.2):
        
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

        for i in range(int(-angle/2), int((angle/2)+1), 2 ):
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

