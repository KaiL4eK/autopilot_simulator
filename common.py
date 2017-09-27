import math as m
import numba as nb
import numpy as np


point_spec = [('x', nb.float32), 
              ('y', nb.float32)]
@nb.jitclass(point_spec)
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # def __str__(self):
    #     return "(%s, %s)" % (self.x, self.y) 

point_type = nb.deferred_type()
point_type.define(Point.class_type.instance_type)

@nb.njit
def get_distance_to (from_object, to_object):
    dx = to_object.x - from_object.x
    dy = to_object.y - from_object.y

    return m.hypot(dx, dy)

@nb.njit
def get_base_vectors_to (from_object, to_object):
    dx = to_object.x - from_object.x
    dy = to_object.y - from_object.y

    dist = m.hypot(dx, dy)

    return np.array([dx / dist, dy / dist], dtype=np.float32)


line_spec = [('p0', point_type), 
             ('p1', point_type),
             ('d', point_type)]
@nb.jitclass(line_spec)
class Line(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

        self.d  = Point(p1.x-p0.x, p1.y-p0.y)

    # def __str__(self):
        # return "(%s, %s)" % (self.p0, self.p1) 

    def intersect_line( self, other_line_np ):
        
        other_line_d_x = other_line_np[1][0]
        other_line_d_y = other_line_np[1][1]
        other_line_p0_x = other_line_np[0][0]
        other_line_p0_y = other_line_np[0][1]

        DET_TOLERANCE = 0.00000001
        
        # print(nb.type(Line))

        # if np.array([self.p0.x, self.p1.x]).min() > np.array([other_line.p0.x, other_line.p1.x]).max() or \
        #    np.array([self.p0.x, self.p1.x]).max() < np.array([other_line.p0.x, other_line.p1.x]).min() or \
        #    np.array([self.p0.y, self.p1.y]).min() > np.array([other_line.p0.y, other_line.p1.y]).max() or \
        #    np.array([self.p0.y, self.p1.y]).max() < np.array([other_line.p0.y, other_line.p1.y]).min():
        #    return -1;

        DET = (-self.d.x * other_line_d_y + self.d.y * other_line_d_x)

        if m.fabs(DET) < DET_TOLERANCE: 
            return -1.

        r = (-other_line_d_y  * (other_line_p0_x-self.p0.x) + other_line_d_x * (other_line_p0_y-self.p0.y)) / DET
        s = (-self.d.y        * (other_line_p0_x-self.p0.x) + self.d.x       * (other_line_p0_y-self.p0.y)) / DET

        if s < 0 or s > 1:
            return -1.

        return r

@nb.njit
def intersect_line1( line_np, other_line_np ):
    
    line_d_x    = line_np[2]
    line_d_y    = line_np[3]
    line_p0_x   = line_np[0]
    line_p0_y   = line_np[1]

    other_line_d_x = other_line_np[1][0]
    other_line_d_y = other_line_np[1][1]
    other_line_p0_x = other_line_np[0][0]
    other_line_p0_y = other_line_np[0][1]

    DET_TOLERANCE = 0.00000001
    
    # print(nb.type(Line))

    # if np.array([self.p0.x, self.p1.x]).min() > np.array([other_line.p0.x, other_line.p1.x]).max() or \
    #    np.array([self.p0.x, self.p1.x]).max() < np.array([other_line.p0.x, other_line.p1.x]).min() or \
    #    np.array([self.p0.y, self.p1.y]).min() > np.array([other_line.p0.y, other_line.p1.y]).max() or \
    #    np.array([self.p0.y, self.p1.y]).max() < np.array([other_line.p0.y, other_line.p1.y]).min():
    #    return -1;

    DET = (-line_d_x * other_line_d_y + line_d_y * other_line_d_x)

    if m.fabs(DET) < DET_TOLERANCE: 
        return -1.

    dx = (other_line_p0_x-line_p0_x)
    dy = (other_line_p0_y-line_p0_y)
    r = (-other_line_d_y  * dx + other_line_d_x * dy) / DET
    s = (-line_d_y        * dx + line_d_x       * dy) / DET

    if s < 0 or s > 1:
        return -1.

    return r

line_type = nb.deferred_type()
line_type.define(Line.class_type.instance_type)

@nb.njit
def line_from_radial(base_point, theta, length=1.):
    new_line = Line(base_point, Point(0, 0)) 
    new_line.d = Point(length * m.cos(m.radians(theta)), length * m.sin(m.radians(theta)))
    new_line.p1 = Point(new_line.p0.x + new_line.d.x, new_line.p0.y + new_line.d.y)

    return np.array([base_point.x, base_point.y, length * m.cos(m.radians(theta)), length * m.sin(m.radians(theta))])

@nb.njit
def testIntersection( l1, l2 ):
    """ prints out a test for checking by hand... """
    # print("Line segment #1 runs from", l1.p0, "to", l1.p1)
    # print("Line segment #2 runs from", l2.p0, "to", l1.p1)

    result = l1.intersect_line( l2 )
    # print("    Intersection result =", result)
    # print()

