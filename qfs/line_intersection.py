from __future__ import print_function, division

import numpy as np
from qfs.common import *


import time
import timeit


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
    # print("Input:", x3, "Output:", perpendicular(x3))

    # d1p = np.array([-10, 10])
    # d2  = np.array([10, -10])

    # print(timeit.timeit('testIntersection(Line(pt1, pt2), Line(pt3, pt4))', 
    #                     setup='import numpy as np; from common import Point, Line, testIntersection;\
    #                            pt1 = Point(10, 10); \
    #                            pt2 = Point(20, 20); pt3 = Point(10, 20); \
    #                            pt4 = Point(20, 10)', 
    #                     number=1000000))

    # print(timeit.timeit('d1p[0] * d2[0] + d1p[1] * d2[1]', setup='import numpy as np; d1p = np.array([-10, 10]); d2  = np.array([10, -10])', number=1000000))


