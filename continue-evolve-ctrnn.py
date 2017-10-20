from __future__ import print_function

import os
import sys
import getopt
import argparse

import pickle
import time

import neat
import visualize

from qfs.simulate_robot import *

populations = None

simulation_seconds = 40.0
map_filename = 'maps/two_obstacles.pmap'
map_filename = 'maps/maze.pmap'

resol = 0.03
time_const = SimManager.time_step

sim_map = get_map_from_file(map_filename)

def eval_genome(genome, config, img=None):
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)
    net.reset()

    sim = SimManager(bot=Robot(x=2, y=10),
                     target=CircleTarget(x=36, y=7.5),
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

def run(filepath):

    pop = neat.Checkpointer.restore_checkpoint(filepath)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(generation_interval=100, filename_prefix='checkpoints_ctrnn/chk_'))

    if 1:
        pe = neat.ParallelEvaluator(4, eval_genome)
        winner = pop.run(pe.evaluate, populations)
    else:
        winner = pop.run(eval_genomes, populations)

    # Save the winner.
    with open('winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    cv2.waitKey(0)

    visualize.plot_stats(stats, view=False, ylog=True, filename="pictures_ctrnn/feedforward-fitness.svg")
    visualize.plot_species(stats, view=False, filename="pictures_ctrnn/feedforward-speciation.svg")

    node_names = {-1: 'ext', -2: 'eyt', -3: 'sf', -4: 'sl', -5: 'sr', 0: 'ux', 1: 'uy', 2: 'wz'}
    visualize.draw_net(config, winner, False, node_names=node_names,
                       filename='pictures_ctrnn/Digraph.gv')
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename="pictures_ctrnn/winner-feedforward.gv")
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename="pictures_ctrnn/winner-feedforward-enabled.gv", show_disabled=False)
    #visualize.draw_net(config, winner, view=False, node_names=node_names,
    #                   filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NEAT xor experiment evaluated across multiple machines.")
    parser.add_argument(
        "checkpoint",
        help="path to checkpoint file",
        action="store",
        )

    ns = parser.parse_args()

    run(ns.checkpoint)
