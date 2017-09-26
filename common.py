import math as m

DET_TOLERANCE = 0.00000001

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

    def __str__(self):
        return "(%s, %s)" % (self.x, self.y) 

class Line:
    def __init__(self, p0=Point(0, 0), p1=Point(0, 0)):
        self.p0 = p0
        self.p1 = p1

        self.d  = p1-p0

    def __str__(self):
        return "(%s, %s)" % (self.p0, self.p1) 

    def intersect_line( self, other_line ):

        if min([self.p0.x, self.p1.x]) > max([other_line.p0.x, other_line.p1.x]) or \
           max([self.p0.x, self.p1.x]) < min([other_line.p0.x, other_line.p1.x]) or \
           min([self.p0.y, self.p1.y]) > max([other_line.p0.y, other_line.p1.y]) or \
           max([self.p0.y, self.p1.y]) < min([other_line.p0.y, other_line.p1.y]):
           return -1;

        DET = (-self.d.x * other_line.d.y + self.d.y * other_line.d.x)

        if m.fabs(DET) < DET_TOLERANCE: 
            return -1

        r = (-other_line.d.y  * (other_line.p0.x-self.p0.x) + other_line.d.x * (other_line.p0.y-self.p0.y)) / DET
        s = (-self.d.y        * (other_line.p0.x-self.p0.x) + self.d.x       * (other_line.p0.y-self.p0.y)) / DET

        if s < 0 or s > 1:
            return -1

        return r

    def from_ray(self, ray, length=1):
        self.p0 = ray.p0
        self.d  = Point(length * m.cos(m.radians(ray.theta)), length * m.sin(m.radians(ray.theta)))
        self.p1 = self.p0 + self.d

class Ray:
    def __init__(self, p0=Point(0, 0), theta=0):
        self.p0     = p0
        self.theta  = theta
