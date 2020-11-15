import multiprocessing
import os

import click
import neat
import gym
# import torch
import numpy as np
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.es_hyperneat import ESNetwork
from pytorch_neat.substrate import Substrate
from pytorch_neat.cppn import create_cppn

max_env_steps = 200


def make_env():
    env = gym.make("CartPole-v0") # obvservation space: 4; action space: 2
    state_space = env.env.action_space # 111
    return env

def make_net(genome, config, bs):
    #start by setting up a substrate for this bad cartpole boi
    params = {"initial_depth": 1,
            "max_depth": 2,
            "variance_threshold": 0.8,
            "band_threshold": 0.05,
            "iteration_level": 3,
            "division_threshold": 0.3,
            "max_weight": 34.0,
            "activation": "sigmoid"}
    input_cords = []
    output_cords = [(0.0, -1.0, -1.0)]
    sign = 1
    # we will use a 3 dimensional substrate, coords laid out here
    for i in range(4):
        input_cords.append((0.0 - i/10*sign, 0.0, 0.0))
        sign *= -1
    leaf_names = []
    for i in range(len(output_cords[0])):
        leaf_names.append(str(i) + "_in")
        leaf_names.append(str(i) + "_out")

    [cppn] = create_cppn(genome, config, leaf_names, ['cppn_out'])
    net_builder = ESNetwork(Substrate(input_cords, output_cords), cppn, params)
    net = net_builder.create_phenotype_network_nd('./genome_vis')
    return net

def reset_substrate(states):
    input_cords = []
    output_cords = [(0.0, -1.0, -1.0)]
    sign = -1
    for i in range(4):
        input_cords.append((0.0 - i/10*sign, 0.0, states[i]/100.))
        sign *= -1
    return Substrate(input_cords, output_cords)

def activate_net(net, states, **kwargs):
    #print(states)
    #new_sub = reset_substrate(states[0])
    #net.reset_substrate(new_sub)
    #network = net.create_phenotype_network_nd() 
    outputs = net.activate(states).numpy()
    #print(outputs[:,0])
    outputs = outputs[0] > 0.5
    assert False, outputs
    return outputs


@click.command()
@click.option("--n_generations", type=int, default=100)
def run(n_generations):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    # comment these 2 lines to not log
    logger = LogReporter("neat.log", evaluator)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
