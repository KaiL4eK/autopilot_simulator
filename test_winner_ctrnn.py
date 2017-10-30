from __future__ import print_function

import os
import pickle

import time

import neat
import cv2

from qfs.simulate_robot import *

# load the winner
with open('winner-ctrnn', 'rb') as f:
    c = pickle.load(f)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-ctrnn')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

simulation_seconds = 10
resol = 0.04
time_const = SimManager.time_step
# sim_map = get_map_from_file('maps/maze.pmap')
# sim = SimManager(bot=Robot(x=2, y=10), target=[36, 12], map_data=sim_map)

sim_map = get_map_from_file('maps/two_obstacles.pmap')
sim = SimManager(bot=Robot(x=3, y=8), target=[18, 7.5], map_data=sim_map)
# sim = SimManager(bot=Robot(x=3, y=8), target=[10, 2], map_data=sim_map)
# sim = SimManager(bot=Robot(x=18, y=2), target=[3, 8], map_data=sim_map)

img = sim_map.get_image(resol)

net = neat.ctrnn.CTRNN.create(c, config, time_const)
net.reset()
while sim.t < simulation_seconds:
    inputs = sim.get_state()
    action = net.advance(inputs, time_const, time_const)
    sim.sample_step([action[0], action[1], 0])
    if sim.bot_collision:
        print('Failed!')
        break

for point in sim.path:
    time_rate = point[0] / simulation_seconds
    cv2.circle(img, center=(int(point[1] / resol), int(point[2] / resol)), 
                    radius=1, thickness=-1, 
                    color=(255 - (255 * time_rate), 0, (255 * time_rate)))
cv2.circle(img, center=(int(sim.target[0] / resol), int(sim.target[1] / resol)), 
			    radius=3, thickness=-1, 
			    color=(0, 0, 0))

cv2.imshow('0', cv2.flip(img, 0))
cv2.waitKey(0)
