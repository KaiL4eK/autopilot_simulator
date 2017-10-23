from __future__ import print_function

import os
import argparse

import pickle

import neat
import visualize
from evaluate_ff import *

def run(filepath, pop_count, render_flag):

    if pop_count is not None:
        pop_count = int(pop_count)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Checkpointer.restore_checkpoint(filepath)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(generation_interval=100, filename_prefix='checkpoints_ff/chk_'))

    if render_flag:
        winner = pop.run(eval_genomes, pop_count)
    else:
        pe = neat.ParallelEvaluator(4, eval_genome)
        winner = pop.run(pe.evaluate, pop_count)


    # Save the winner.
    with open('winner-ff', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    cv2.waitKey(0)

    visualize.plot_stats(stats, view=False, ylog=True, filename="pictures_ff/feedforward-fitness.svg")
    visualize.plot_species(stats, view=False, filename="pictures_ff/feedforward-speciation.svg")

    node_names = {-1: 'ext', -2: 'eyt', -3: 'sf', -4: 'sl', -5: 'sr', 0: 'ux', 1: 'uy', 2: 'wz'}
    visualize.draw_net(config, winner, False, node_names=node_names,
                       filename='pictures_ff/Digraph.gv')
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename="pictures_ff/winner-feedforward.gv")
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename="pictures_ff/winner-feedforward-enabled.gv", show_disabled=False)
    #visualize.draw_net(config, winner, view=False, node_names=node_names,
    #                   filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description")
    parser.add_argument(
        "checkpoint",   
        help="path to checkpoint file",
        action="store",
        )
    parser.add_argument(
        "--pops",
        help="Count of population",
        default=None,
        action="store",
        )
    parser.add_argument(
        "--simtime",
        help="Simulation time",
        default=None,
        action="store",
        )
    parser.add_argument(
        "--render",
        help="Simulation time",
        action="store_true",
        )


    ns = parser.parse_args()
    if ns.simtime is not None:
        simulation_seconds = int(ns.simtime)

    print('Simulation time:', simulation_seconds)

    run(ns.checkpoint, ns.pops, ns.render)
