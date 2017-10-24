from __future__ import print_function

import os
import pickle

from evaluate_ctrnn import *

# load the winner
with open('winner-ctrnn', 'rb') as f:
    c = pickle.load(f)

# print('Loaded genome:')
# print(c)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-ctrnn')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

simulation_seconds = 40

eval_genomes([(0, c)], config, img)

cv2.imshow('1', cv2.flip(img, 0))
cv2.waitKey(0)
