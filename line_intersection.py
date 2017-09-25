from __future__ import print_function, division

import numpy as np
from common import *

#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#

def perp( a ):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

# p1 = np.array( [0.0, 0.0] )
# p2 = np.array( [1.0, 0.0] )

# p3 = np.array( [4.0, -5.0] )
# p4 = np.array( [4.0, 2.0] )

# print( seg_intersect( p1,p2, p3,p4) )

# p1 = np.array( [2.0, 2.0] )
# p2 = np.array( [4.0, 3.0] )

# p3 = np.array( [6.0, 0.0] )
# p4 = np.array( [6.0, 3.0] )

# print( seg_intersect( p1,p2, p3,p4) )

#
# intersections.py
#
# Python for finding line intersections
#   intended to be easily adaptable for line-segment intersections
#

import math

def perpendicular( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def normalize(a):
    a = np.array(a)
    return a/np.linalg.norm(a)

def intersectLines1( pt1, pt2, ptA, ptB ): 
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
        
        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    d1p = perpendicular([dx1, dy1])
    print(d1p)

    # the second line is ptA + s*(ptB-ptA)
    x3, y3 = ptA;   x4, y4 = ptB;
    dx2 = x4 - x3;  dy2 = y4 - y3;

    d2 = [dx2, dy2]
    print(d2)

    dp = [x1 - x3, y1 - y3]

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x3, y3) + s(dx2, dy2)
    #
    # which is the same as
    #
    #    [ dx1  -dx2 ][ r ] = [ x3-x1 ]
    #    [ dy1  -dy2 ][ s ] = [ y3-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy2   dx2 ] [ x3-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y3-y1 ]
    #
    # where DET = (-dx1 * dy2 + dy1 * dx2)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy2 + dy1 * dx2)

    if math.fabs(DET) < DET_TOLERANCE: 
        return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy2  * (x3-x1) +  dx2 * (y3-y1))

    # find the scalar amount along the input line
    # s = DETinv * (-dy1 * (x3-x1) + dx1 * (y3-y1))

    # return the average of the two descriptions
    # xi = (x1 + r*dx1 + x3 + s*dx2)/2.0
    # yi = (y1 + r*dy1 + y3 + s*dy2)/2.0
    xi = (x1 + r*dx1)
    yi = (y1 + r*dy1)
    valid = 1
    if r < 0 or r > 1:
        valid = 0

    return ( xi, yi, valid, r, 0 )


def intersect_lines_r( l1, l2 ): 

    DET_TOLERANCE = 0.00000001

    #----------------------------------------------
    DET = (-l1.d.x * l2.d.y + l1.d.y * l2.d.x)

    if math.fabs(DET) < DET_TOLERANCE: 
        return -1

    r = (-l2.d.y  * (l2.p0.x-l1.p0.x) +  l2.d.x * (l2.p0.y-l1.p0.y)) / DET

    return r



import time
import timeit

def testIntersection( l1, l2 ):
    """ prints out a test for checking by hand... """
    print("Line segment #1 runs from", l1.p0, "to", l1.p1)
    print("Line segment #2 runs from", l2.p0, "to", l1.p1)

    result = intersect_lines_r( l1, l2 )
    print("    Intersection result =", result)
    print()

if __name__ == "__main__":

    pt1 = Point(10, 10)
    pt2 = Point(20, 20)

    pt3 = Point(10, 20)
    pt4 = Point(20, 10)

    pt5 = Point(40, 20)

    testIntersection( Line(pt1, pt2), Line(pt3, pt4) )
    testIntersection( Line(pt1, pt3), Line(pt2, pt4) )
    testIntersection( Line(pt1, pt2), Line(pt4, pt5) )

    x3 = np.array([1, 0])
    print("Input:", x3, "Output:", perpendicular(x3))

    # d1p = np.array([-10, 10])
    # d2  = np.array([10, -10])

    # print(timeit.timeit('np.inner(d1p, d2)', setup='import numpy as np; d1p = np.array([-10, 10]); d2  = np.array([10, -10])', number=1000000))
    # print(timeit.timeit('d1p[0] * d2[0] + d1p[1] * d2[1]', setup='import numpy as np; d1p = np.array([-10, 10]); d2  = np.array([10, -10])', number=1000000))


