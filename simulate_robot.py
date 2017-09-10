import cv2
import numpy as np
import math as m

map_shape = (30, 20)				# meters

initial_state = (2, map_shape[1] / 2)	# x, y

#	Axis of map:
#	|y
#	|
#	|
#	------x

class SimObject(object):
	def __init__ (self, x=0, y=0, theta=0):
		self.x 		= x
		self.y 		= y
		self.theta	= theta

	def get_distance_to (self, dist_object=None):
		if dist_object is None:
			return 0

		dx = dist_object.x - self.x
		dy = dist_object.y - self.y

		return m.hypot(dx, dy)

	def get_base_vectors_to (self, dist_object=None):
		if dist_object is None:
			return (0, 0)

		dx = dist_object.x - self.x
		dy = dist_object.y - self.y

		dist = m.hypot(dx, dy)

		return (dx / dist, dy / dist)


class CircleObstacle(SimObject):
	def __init__ (self, x=0, y=0, radius=0):
		super(self.__class__, self).__init__(x, y, 0)
		self.r 		= radius


class CircleTarget(SimObject):
	def __init__ (self, x=0, y=0, radius=0):
		super(self.__class__, self).__init__(x, y, 0)
		self.r 		= radius

class Robot(SimObject):
	def __init__ (self, x=0, y=0, theta=0):
		super(self.__class__, self).__init__(x, y, theta)

		self.r 		= 0.56

		self.ux 	= 0
		self.uy		= 0
		self.wz		= 0

bot = Robot(x=initial_state[0], y=initial_state[1], theta=60)

obstacles = []

target = CircleTarget(x=25, y=(map_shape[1] / 2), radius=0.1)

def show_map (bot=None, target=None, obstacles=None, resolution_m_px=1):
	width = map_shape[0] / float(resolution_m_px)
	height = map_shape[1] / float(resolution_m_px)

	print('Draw map: %d / %d' % (width, height))

	img = np.ones(shape=(int(height), int(width), 3), dtype=np.uint8) * 255

	if bot:
		cv2.circle(img, center=(int(bot.x / resolution_m_px), int(bot.y / resolution_m_px)), 
						radius=int(bot.r/resolution_m_px), 
						thickness=-1, color=(0, 255, 0))
		cv2.line(img, 	pt1=(int(bot.x / resolution_m_px), int(bot.y / resolution_m_px)),
						pt2=(int((bot.x + m.cos(m.radians(bot.theta))) / resolution_m_px), int((bot.y - m.sin(m.radians(bot.theta))) / resolution_m_px)),
						color=(0, 255, 0),
						thickness=1 )
		cv2.circle(img, center=(int(bot.x / resolution_m_px), int(bot.y / resolution_m_px)), 
						radius=1, thickness=-1, color=(255, 0, 0))

		(dx, dy) = bot.get_base_vectors_to(target)

		cv2.line(img, 	pt1=(int(bot.x / resolution_m_px), int(bot.y / resolution_m_px)),
						pt2=(int((bot.x + dx) / resolution_m_px), int((bot.y - dy) / resolution_m_px)),
						color=(255, 0, 0),
						thickness=1 )

	if target:
		cv2.circle(img, center=(int(target.x / resolution_m_px), int(target.y / resolution_m_px)), 
						radius=int(target.r/resolution_m_px), 
						thickness=-1, color=(255, 0, 0))

	cv2.imshow('1', img)
	cv2.waitKey(0)

show_map(bot=bot, target=target, obstacles=obstacles, resolution_m_px=0.1)


