#!/usr/bin/env python

from __future__ import print_function

import rospy
import numpy as np

from autopilot_simulator.srv import *

def process_input(req):
	print('Processed: %s' % req.msg)
	return QuadroDataResponse(resp='Resp!')

def init():
	rospy.init_node('quardo_learn_server')
	s = rospy.Service('process_quadro', QuadroData, process_input)
	print('Ready to go!')
	rospy.spin()

if __name__ == "__main__":
	init()
