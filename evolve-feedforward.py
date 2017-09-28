"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function

import os
import sys
sys.path.append(os.path.expanduser('~/Dev/neat-python'))

import pickle
import time

# import cart_pole

import neat
import visualize

from simulate_robot import *

simulation_seconds = 10.0
map_filename = 'two_obstacles.pmap'

resol = 0.02
dt = 1/1000 # 200 Hz

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config, img, sim_map):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    sim = SimManager(dt=dt, # 200 Hz
                        bot=Robot(x=2, y=8, theta=0),
                        target=CircleTarget(x=18, y=5),
                        map_data=sim_map)

    # sim2 = SimManager(dt=dt,
    #                     bot=Robot(x=2, y=6, theta=0),
    #                     target=CircleTarget(x=13, y=7),
    #                     obstacles=[], map_size_m=map_shape)

    # sim3 = SimManager(dt=dt,
    #                     bot=Robot(x=11, y=1, theta=0),
    #                     target=CircleTarget(x=13, y=7),
    #                     obstacles=[], map_size_m=map_shape)

    # sims = [sim1]
    # fitnesses = [0, 0, 0]

    # for i, sim in enumerate(sims):
    while sim.t < simulation_seconds:

        inputs = sim.get_state()
        # print(inputs)

        action = net.activate(inputs)

        # print(action)
        
        sim.sample_step(action, False)
        if sim.bot_collision:
            break


    for point in sim.path:
        time_rate = point[0] / simulation_seconds
        cv2.circle(img, center=(int(point[1] / resol), int(point[2] / resol)), 
                        radius=1, thickness=-1, 
                        color=(255 - (255 * time_rate), 0, (255 * time_rate)))

        # fitnesses[i] = 

    return -sim.get_fitness()


def eval_genomes(genomes, config):

    sim_map = get_map_from_file(map_filename)

    img = sim_map.get_image(resol)

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, img, sim_map)

    cv2.imshow('1', cv2.flip(img, 0))
    cv2.waitKey(30)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # pe = neat.ParallelEvaluator(4, eval_genome)
    # winner = pop.run(pe.evaluate)
    winner = pop.run(eval_genomes)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    cv2.waitKey(0)

    visualize.plot_stats(stats, ylog=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, filename="feedforward-speciation.svg")

    # node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    # visualize.draw_net(config, winner, True, node_names=node_names)

    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward.gv")
    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward-enabled.gv", show_disabled=False)
    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()
