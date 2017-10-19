from __future__ import print_function

import os
import sys
import argparse
import getopt

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

def eval_genome(genome, config):
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

    return 100-sim.get_fitness()

def run(config_file, addr, authkey, mode, workers):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=100, filename_prefix='checkpoints_ctrnn/chk_'))

    # setup an DistributedEvaluator
    de = neat.DistributedEvaluator(
        addr,  # connect to addr
        authkey,  # use authkey to authenticate
        eval_genome,  # use eval_genome() to evaluate a genome
        secondary_chunksize=40,  # send 4 genomes at once
        num_workers=workers,  # when in secondary mode, use this many workers
        worker_timeout=10,  # when in secondary mode and workers > 1,
                            # wait at most 10 seconds for the result
        mode=mode,  # wether this is the primary or a secondary node
                    # in most case you can simply pass
                    # 'neat.distributed.MODE_AUTO' as the mode.
                    # This causes the DistributedEvaluator to
                    # determine the mode by checking if address
                    # points to the localhost.
        )

    # start the DistributedEvaluator
    de.start(
        exit_on_stop=True,  # if this is a secondary node, call sys.exit(0) when
                            # when finished. All code after this line will only
                            # be executed by the primary node.
        secondary_wait=3,  # when a secondary, sleep this many seconds before continuing
                       # this is useful when the primary node may need more time
                       # to start than the secondary nodes.
        )

    winner = p.run(de.evaluate)

    # stop evaluator
    de.stop()

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner.
    with open('winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)

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


def addr_tuple(s):
    """converts a string into a tuple of (host, port)"""
    if "[" in s:
        # ip v6
        if (s.count("[") != 1) or (s.count("]") != 1):
            raise ValueError("Invalid IPv6 address!")
        end = s.index("]")
        if ":" not in s[end:]:
            raise ValueError("IPv6 address does specify a port to use!")
        host, port = s[1:end], s[end+1:]
        port = int(port)
        return (host, port)
    else:
        host, port = s.split(":")
        port = int(port)
        return (host, port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NEAT xor experiment evaluated across multiple machines.")
    parser.add_argument(
        "address",
        help="host:port address of the main node",
        type=addr_tuple,
        action="store",
        )
    parser.add_argument(
        "--workers",
        type=int,
        help="number of processes to use for evaluating the genomes",
        action="store",
        default=1,
        dest="workers",
        )
    parser.add_argument(
        "--authkey",
        action="store",
        help="authkey to use (default: 'neat-python')",
        default=b"neat-python",
        dest="authkey",
        )
    parser.add_argument(
        "--force-secondary","--force-slave",
        action="store_const",
        const=neat.distributed.MODE_SECONDARY,
        default=neat.distributed.MODE_AUTO,
        help="Force secondary mode (useful for debugging)",
        dest="mode",
        )
    ns = parser.parse_args()

    address = ns.address
    host, port = address
    workers = ns.workers
    authkey = ns.authkey
    mode = ns.mode

    if (host in ("0.0.0.0", "localhost", "")) and (mode == neat.distributed.MODE_AUTO):
        # print an error message
        # we are using auto-mode determination in this example,
        # which does not work well with '0.0.0.0' or 'localhost'.
        print("Please do not use '0.0.0.0' or 'localhost' as host.")
        # sys.exit(1)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')

    print("Starting Node...")
    print("Please ensure that you are using more than one node.")

    run(config_path, address, authkey, mode, workers)
