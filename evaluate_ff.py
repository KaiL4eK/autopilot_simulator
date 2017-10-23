import os
import sys

import pickle
import time

import neat
import cv2

from qfs.simulate_robot import *

simulation_seconds = 400.0
map_filename = 'maps/two_obstacles.pmap'
map_filename = 'maps/maze.pmap'

sim_map = get_map_from_file(map_filename)

resol = 0.03

def eval_genome(genome, config, img=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    sim = SimManager(bot=Robot(x=2, y=10),
                     target=[36, 7.5],
                     map_data=sim_map)

    while sim.t < simulation_seconds:
        inputs = sim.get_state()
        action = net.activate(inputs)
        sim.sample_step(action)
        if sim.bot_collision:
            break

    if img is not None:
        for point in sim.path:
            time_rate = point[0] / simulation_seconds
            cv2.circle(img, center=(int(point[1] / resol), int(point[2] / resol)), 
                            radius=1, thickness=-1, 
                            color=(255 - (255 * time_rate), 0, (255 * time_rate)))

    return -sim.get_fitness()


def eval_genomes(genomes, config):

    img = sim_map.get_image(resol)

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, img)

    cv2.imshow('1', cv2.flip(img, 0))
    cv2.waitKey(30)
