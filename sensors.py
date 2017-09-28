from common import *
import numba as nb
import numpy as np

# sonar_spec = [('base_dist', nb.float32), 
#               ('stheta', nb.float32),
#               ('angle', nb.float32),
#               ('dist_max', nb.float32),
#               ('dist_min', nb.float32),
#               ('nrows', nb.int32),
#               ('ray_angles', nb.float32[:]),
#               ('ray_values', nb.float32[:]),
#               ('range', nb.float32),
#               ('base_x', nb.float32),
#               ('base_y', nb.float32),
#               ('base_theta', nb.float32)]
# @nb.jitclass(sonar_spec)
class SonarSensor(object):
    def __init__ (self, base_dist, stheta):
        
        self.base_dist = base_dist
        self.stheta = stheta

        self.angle    = 30
        self.dist_max = 4.5
        self.dist_min = 0.02

        self.nrows      = int(self.angle / 1 + 1)
        self.ray_angles = np.zeros(shape=(self.nrows), dtype=np.float32)

        for i in nb.prange(self.nrows):
            self.ray_angles[i] = -self.angle / 2 + (self.angle / self.nrows * i)

        self.ray_values = np.ones(shape=(self.nrows), dtype=np.float32)

        self.range      = self.dist_max

        self.base_x     = 0
        self.base_y     = 0
        self.base_theta = 0

    def update_base_point (self, x, y, theta):
        self.base_theta = self.stheta + theta
        self.base_x = x + self.base_dist * m.cos(m.radians(self.base_theta))
        self.base_y = y + self.base_dist * m.sin(m.radians(self.base_theta))

    def get_left_right_angles (self, theta=0):
        return (theta + self.stheta + self.angle/2, 
                theta + self.stheta - self.angle/2)

    def get_state_point(self):
        return Point(self.base_x, self.base_y)

    def update(self, lines):

        self.range = update_sonar(self.nrows, self.ray_angles, self.ray_values, lines, self.dist_max, self.get_state_point(), self.base_theta)


@nb.njit(nogil=True)
def update_sonar(nrows, ray_angles, ray_values, lines, dist_max, base_point, base_theta):
    ray_values = np.ones(shape=(nrows), dtype=np.float32)
    for ray_idx in nb.prange(nrows):
        ray_line_np = line_from_radial_np(base_point=base_point, 
                                          length=dist_max, 
                                          theta=base_theta + ray_angles[ray_idx])
        for i in nb.prange(len(lines)):
            r = intersect_line_np(ray_line_np, lines[i])
            if r > 0:
                ray_values[ray_idx] = np.array([r, ray_values[ray_idx]]).min()

    return ray_values.min()

# sonar_type = nb.deferred_type()
# sonar_type.define(SonarSensor.class_type.instance_type)
