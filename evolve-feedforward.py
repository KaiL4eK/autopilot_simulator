"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function

import os
import pickle

# import cart_pole

import neat
import visualize

from simulate_robot import *

runs_per_net = 5
simulation_seconds = 10.0

map_shape = (15, 8)                # meters
initial_state = (2, map_shape[1] / 2)   # x, y
resol = 0.01

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config, img):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    sim = SimManager(dt=0.005, # 200 Hz
                        bot=Robot(x=2, y=2, theta=0),
                        target=CircleTarget(x=13, y=7),
                        obstacles=[], map_size_m=map_shape)

    while sim.t < simulation_seconds:

        inputs = sim.get_state()
        # print(inputs)

        action = net.activate(inputs)

        # print(action)
        
        if not sim.sample_step(action):
            break

    for point in sim.path:
        cv2.circle(img, center=(int(point[0] / resol), int(point[1] / resol)), 
                        radius=1, thickness=-1, color=(255, 0, 0))

    return -sim.get_fitness()


def eval_genomes(genomes, config):

    img = np.ones(shape=(int(map_shape[1] / resol), int(map_shape[0] / resol), 3), dtype=np.uint8) * 255

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, img)

    cv2.imshow('1', img)
    cv2.waitKey(1)


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

    # visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    # visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

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
