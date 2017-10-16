#!/usr/bin/env python3

"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function

import os
import sys
import getopt

import pickle
import time

import neat
import visualize

from simulate_robot import *

simulation_seconds = 40.0
map_filename = 'two_obstacles.pmap'
map_filename = 'maze.pmap'

resol = 0.03
time_const = SimManager.time_step

sim_map = get_map_from_file(map_filename)

def eval_genome(genome, config, img=None):
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)
    net.reset()

    sim = SimManager(bot=Robot(x=2, y=10),
                     target=CircleTarget(x=36, y=2),
                     map_data=sim_map)

    while sim.t < simulation_seconds:

        inputs = sim.get_state()

        action = net.advance(inputs, time_const, time_const)
        
        sim.sample_step(action)
        if sim.bot_collision:
            break


    if img is not None:
        for point in sim.path:
            time_rate = point[0] / simulation_seconds
            cv2.circle(img, center=(int(point[1] / resol), int(point[2] / resol)), 
                            radius=1, thickness=-1, 
                            color=(255 - (255 * time_rate), 0, (255 * time_rate)))

    return 100-sim.get_fitness()


def eval_genomes(genomes, config):

    img = sim_map.get_image(resol)

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, img)

    cv2.imshow('1', cv2.flip(img, 0))
    cv2.waitKey(30)

def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    if 0:
        pe = neat.ParallelEvaluator(4, eval_genome)
        winner = pop.run(pe.evaluate)
    else:
        winner = pop.run(eval_genomes)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    cv2.waitKey(0)

    visualize.plot_stats(stats, view=False, ylog=True, filename="pictures/feedforward-fitness.svg")
    visualize.plot_species(stats, view=False, filename="pictures/feedforward-speciation.svg")

    node_names = {-1: 'ext', -2: 'eyt', -3: 'sf', -4: 'sl', -5: 'sr', 0: 'ux', 1: 'uy', 2: 'wz'}
    visualize.draw_net(config, winner, False, node_names=node_names)

    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename="pictures/winner-feedforward.gv")
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename="pictures/winner-feedforward-enabled.gv", show_disabled=False)
    #visualize.draw_net(config, winner, view=False, node_names=node_names,
    #                   filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

if __name__ == '__main__':
    run()
