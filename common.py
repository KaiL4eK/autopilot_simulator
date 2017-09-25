

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

    def intersect_lines_r( l1, l2 ): 

        

        #----------------------------------------------
        DET = (-l1.d.x * l2.d.y + l1.d.y * l2.d.x)

        if math.fabs(DET) < DET_TOLERANCE: 
            return -1

        r = (-l2.d.y  * (l2.p0.x-l1.p0.x) +  l2.d.x * (l2.p0.y-l1.p0.y)) / DET

        return r