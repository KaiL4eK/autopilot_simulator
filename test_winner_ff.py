from __future__ import print_function

import os
import pickle

from qfs.simulate_robot import *

import neat
from neat import nn

map_filename = 'maps/maze.pmap'
sim_map = get_map_from_file(map_filename)

# load the winner
with open('winner-ff', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)

resol = 0.03
simulation_seconds = 40

img = sim_map.get_image(resol)
sim = SimManager(bot=Robot(x=2, y=10, theta=0),
                 target=[36, 7.5],
                 map_data=sim_map)

while sim.t < simulation_seconds:

    inputs = sim.get_state()
    action = net.activate(inputs)
    
    sim.sample_step(action)
    if sim.bot_collision:
        print('Failed!')
        break


if img is not None:
    for point in sim.path:
        time_rate = point[0] / simulation_seconds
        cv2.circle(img, center=(int(point[1] / resol), int(point[2] / resol)), 
                        radius=1, thickness=-1, 
                        color=(255 - (255 * time_rate), 0, (255 * time_rate)))

cv2.imshow('1', cv2.flip(img, 0))
cv2.waitKey(0)
