#!/usr/bin/env python

from __future__ import print_function

import rospy
import numpy as np

from std_msgs.msg import String

from autopilot_simulator.srv import *

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def process_control(req):
    # print('Received: %s %s' % ((req.ex, req.ey), rospy.get_time()))
    return QuadroControlDataResponse(x_force=0, y_force=0)

def talker():
    pub = rospy.Publisher('test_topic', String, queue_size=10)

    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

def init():
    rospy.init_node('quardo_learn_server')
    s = rospy.Service('process_quadro_control', QuadroControlData, process_control)
    print('Ready to go!')

    try:
        talker()
    except rospy.ROSInterruptException:
        pass

    # rospy.spin()

if __name__ == "__main__":
    init()
