import math as m
# import numba as nb
import numpy as np

# @nb.njit(nb.float32(nb.float32))
def to_radians(degree):
    if degree > 180:
        degree -= 360.

    if degree < -180:
        degree += 360.

    return m.radians(degree)

# @nb.njit(nb.float32(nb.float32))
def degrees_2_degrees(degree):
    if degree > 180:
        degree -= 360.

    if degree < -180:
        degree += 360.

    return degree

# point_spec = [('x', nb.float32), 
              # ('y', nb.float32)]
# @nb.jitclass(point_spec)
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y   

    # def __str__(self):
    #     return "(%s, %s)" % (self.x, self.y) 

# point_type = nb.deferred_type()
# point_type.define(Point.class_type.instance_type)

################ Just for tests ######################
# @nb.njit(nb.float32(nb.float32[2], nb.float32[2]))
def get_distance_to_4 (p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return m.hypot(dx, dy)

# @nb.njit
def get_distance_to (from_object, to_object):
    dx = to_object.x - from_object.x
    dy = to_object.y - from_object.y

    return m.hypot(dx, dy)

# @nb.njit
def get_base_vectors_to (from_object, to_object):
    dx = to_object.x - from_object.x
    dy = to_object.y - from_object.y

    dist = m.hypot(dx, dy)

    return np.array([dx / dist, dy / dist], dtype=np.float32)
################ ############## ######################

# @nb.njit(nb.float32(nb.float32[2], nb.float32[2]))
def np_get_distance_to_x2_incr (p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return m.hypot(dx * 2, dy)

# @nb.njit(nb.float32(nb.float32[2], nb.float32[2]))
def np_get_distance_to (p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return m.hypot(dx, dy)

# @nb.njit(nb.float32[2](nb.float32[2], nb.float32[2]))
def np_get_base_vectors_to (p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = m.hypot(dx, dy)
    if m.fabs(dist) < 0.0000001:
        dist = 1

    return np.array([dx / dist, dy / dist], dtype=np.float32)
    
    # dp = p1 - p2
    # dist = m.hypot(dp[0], dp[1])
    # return dp / dist


# line_spec = [('p0', point_type), 
             # ('p1', point_type),
             # ('d', point_type)]
# @nb.jitclass(line_spec)
class Line(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

        self.d  = Point(p1.x-p0.x, p1.y-p0.y)

    def get_as_np_array_p0_d(self):
        return np.array([self.p0.x, self.p0.y, self.d.x, self.d.y], dtype=np.float32)

    def get_as_np_array_p0_p1(self):
        return np.array([self.p0.x, self.p0.y, self.p1.x, self.p1.y], dtype=np.float32)

# @nb.njit(nb.float32(nb.float32[4], nb.float32[4]))
def intersect_line_np( line_np, other_line_np ):
    
    line_d_x    = line_np[2]
    line_d_y    = line_np[3]
    line_p0_x   = line_np[0]
    line_p0_y   = line_np[1]

    other_line_d_x = other_line_np[2]
    other_line_d_y = other_line_np[3]
    other_line_p0_x = other_line_np[0]
    other_line_p0_y = other_line_np[1]

    DET_TOLERANCE = 0.00000001

    DET = (-line_d_x * other_line_d_y + line_d_y * other_line_d_x)

    if m.fabs(DET) < DET_TOLERANCE: 
        return -1.

    dx = (other_line_p0_x - line_p0_x)
    dy = (other_line_p0_y - line_p0_y)
    r = (-other_line_d_y  * dx + other_line_d_x * dy) / DET
    s = (-line_d_y        * dx + line_d_x       * dy) / DET

    if s < 0 or s > 1:
        return -1.

    return r

# line_type = nb.deferred_type()
# line_type.define(Line.class_type.instance_type)

# @nb.njit(nb.float32[4](nb.float32[2], nb.float32, nb.float32))
def line_from_radial_np(base_point, theta, length=1.):

    return np.array([base_point[0], base_point[1], length * m.cos(m.radians(theta)), length * m.sin(m.radians(theta))], dtype=np.float32)

# @nb.njit
def line_from_radial(base_point, theta, length=1.):
    new_line = Line(base_point, Point(0, 0)) 
    new_line.d = Point(length * m.cos(m.radians(theta)), length * m.sin(m.radians(theta)))
    new_line.p1 = Point(new_line.p0.x + new_line.d.x, new_line.p0.y + new_line.d.y)

    return new_line
