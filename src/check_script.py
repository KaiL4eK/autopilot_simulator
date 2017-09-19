#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy

from autopilot_simulator.srv import *

def send_message(msg):
    rospy.wait_for_service('process_quadro')
    try:
        serv = rospy.ServiceProxy('process_quadro', QuadroData)
        resp1 = serv(msg=msg)
        return resp1.resp
    except rospy.ServiceException, e:
        print("Service call failed: %s" % e)

def usage():
    return "%s [msg]" % sys.argv[0]

if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        msg = str(sys.argv[1])
    else:
        print(usage())
        sys.exit(1)

    print("Requesting '%s'" % msg)
    print("Response '%s'" % (send_message(msg)))
