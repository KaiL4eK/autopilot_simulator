import os
import sys

import pickle
import time

import neat
import cv2

from qfs.simulate_robot import *

simulation_seconds = 200

sim_maps = [ get_map_from_file('maps/two_obstacles.pmap'),
             get_map_from_file('maps/maze.pmap'),
             get_map_from_file('maps/second_map.pmap') ]

imgs = []

resol = 0.04
time_const = SimManager.time_step

def eval_genome(genome, config, imgs=None):
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)
    net.reset()

    simulations = [ SimManager(bot=Robot(x=3, y=8), target=[18, 2], map_data=sim_maps[0]),
                    SimManager(bot=Robot(x=2, y=10), target=[36, 2], map_data=sim_maps[1]),
                    SimManager(bot=Robot(x=2, y=17), target=[14, 15], map_data=sim_maps[2]) ]

    sim_values = [0, 0, 0]

    for idx_sim, sim in enumerate(simulations):

        while sim.t < simulation_seconds:
            inputs = sim.get_state()
            action = net.advance(inputs, time_const, time_const)
            sim.sample_step(action)
            if sim.bot_collision:
                break

        if imgs is not None:
            for point in sim.path:
                time_rate = point[0] / simulation_seconds
                cv2.circle(imgs[idx_sim], center=(int(point[1] / resol), int(point[2] / resol)), 
                                radius=1, thickness=-1, 
                                color=(255 - (255 * time_rate), 0, (255 * time_rate)))

        sim_values[idx_sim] = -sim.get_fitness()

    return min(sim_values)

def eval_genomes(genomes, config):

    imgs = [ sim_maps[0].get_image(resol),
             sim_maps[1].get_image(resol),
             sim_maps[2].get_image(resol) ]

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, imgs)

    for i, img in enumerate(imgs):
        cv2.imshow(str(i), cv2.flip(img, 0))

    cv2.waitKey(30)


