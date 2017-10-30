import os
import sys

import pickle
import time

import neat
import cv2

from qfs.simulate_robot import *

simulation_seconds = 10

sim_map = get_map_from_file('maps/two_obstacles.pmap')
sim_map = get_map_from_file('maps/maze.pmap')

resol = 0.04


def eval_genome(genome, config, img=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)


    # simulations = [ SimManager(bot=Robot(x=3, y=8), target=[18, 2], map_data=sim_map),
    #                 SimManager(bot=Robot(x=3, y=8), target=[18, 7.5], map_data=sim_map), 
    #                 SimManager(bot=Robot(x=3, y=8), target=[10, 2], map_data=sim_map),
    #                 SimManager(bot=Robot(x=3, y=8), target=[10, 7.5], map_data=sim_map),
    #                 SimManager(bot=Robot(x=18, y=2), target=[10, 2], map_data=sim_map),
    #                 SimManager(bot=Robot(x=18, y=2), target=[3, 8], map_data=sim_map),
    #                 SimManager(bot=Robot(x=18, y=2), target=[18, 7.5], map_data=sim_map),
    #                 ]

    simulations = [ SimManager(bot=Robot(x=2, y=10), target=[36, 2], map_data=sim_map),
                    SimManager(bot=Robot(x=2, y=10), target=[36, 12], map_data=sim_map),
                    # SimManager(bot=Robot(x=2, y=10), target=[20, 2], map_data=sim_map),
                    ]

    sim_values = []

    for idx_sim, sim in enumerate(simulations):

        while sim.t < simulation_seconds:
            inputs = sim.get_state()
            action = net.activate(inputs)
            sim.sample_step([action[0], action[1], 0])
            if sim.bot_collision:
                break

        if img is not None:
            for point in sim.path:
                time_rate = point[0] / simulation_seconds
                cv2.circle(img, center=(int(point[1] / resol), int(point[2] / resol)), 
                                radius=1, thickness=-1, 
                                color=(255 - (255 * time_rate), 0, (255 * time_rate)))
            cv2.circle(img, center=(int(sim.target[0] / resol), int(sim.target[1] / resol)), 
                radius=3, thickness=-1, 
                color=(0, 0, 0))

        sim_values.append(-sim.get_fitness())

    return np.min(sim_values)

def eval_genomes(genomes, config):

    img = sim_map.get_image(resol)

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, img)

    cv2.imshow('0', cv2.flip(img, 0))
    cv2.waitKey(30)
