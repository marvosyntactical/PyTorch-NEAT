import multiprocessing
import os

import click
import neat
import gym
# import torch
import numpy as np

from pytorch_neat import t_maze
from pytorch_neat.activations import tanh_activation
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.safe_es_hyperneat import ESNetwork
from pytorch_neat.substrate import Substrate
from pytorch_neat.cppn_safe import create_cppn


PARAMS = {"initial_depth": 1,
        "max_depth": 2,
        "variance_threshold": 0.55,
        "band_threshold": 0.34,
        "iteration_level": 3,
        "division_threshold": 0.21,
        "max_weight": 13.0,
        "activation": "tanh",
        "safe_baseline_depth": 2}

max_env_steps = 200


actions_dict = {}


def make_env():
    return gym.make("CartPole-v0")

def make_net(genome, config, bs):
    #start by setting up a substrate for this bad cartpole boi

    input_cords, output_cords, leaf_names = set_initial_coords()
    [cppn] = create_cppn(genome, config, leaf_names, ['cppn_out'])
    print(len(cppn.weights))
    net_builder = ESNetwork(Substrate(input_cords, output_cords), cppn, PARAMS)
    return net_builder

def set_initial_coords():
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
    return input_cords, output_cords, leaf_names
    

def reset_substrate(states):
    input_cords = []
    output_cords = [(0.0, -1.0, -1.0)]
    sign = -1
    for i in range(4):
        input_cords.append((0.0 - i/10*sign, 0.0, 0.0 + (states[i]/10)))
        sign *= -1
    return Substrate(input_cords, output_cords)

def execute_back_prop(genome_dict, champ_key, config):
    input_cords, output_cords, leaf_names = set_initial_coords()
    [cppn] = create_cppn(genome_dict[champ_key], config, leaf_names, ['cppn_out'])
    net_builder = ESNetwork(Substrate(input_cords, output_cords), cppn, PARAMS)
    champ_output = net_builder.safe_baseline(False)
    for key in genome_dict:
        if key != champ_key:
            [cppn_2] = create_cppn(genome_dict[key], config, leaf_names, ['cppn_out'])
            es_net = ESNetwork(Substrate(input_cords, output_cords), cppn_2, PARAMS)
            output = es_net.safe_baseline(True)
            loss_val = (champ_output - output).pow(2).mean()
            loss_val.backward()
            es_net.optimizer.step()
            print(cppn.weights)
            es_net.map_back_to_genome(genome_dict[key])
    return 

def activate_net(net,states):
    #print(states)
    new_sub = reset_substrate(states[0])
    net.reset_substrate(new_sub)
    network = net.create_phenotype_network_nd() 
    outputs = network.activate(states).numpy()
    #print(outputs)
    return outputs[:,0] > 0.5


@click.command()
@click.option("--n_generations", type=int, default=100)
def run(n_generations):
    # Load the config file, which is assumed to live in
    # the same directory as this script.

    total_grad_steps = 3

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

    # safe es-hyperneat evaluations should all use this conventions
    # where the number of iterations and gradients steps is set in the config
    # and then we recursively call this adjusting grads each time and
    # then returning to perform evolution after this
    def eval_genomes(genomes, config, grad_steps=0):
        genome_dict = {}
        champ_key = 0
        best_fitness = -10000
        for _, genome in genomes:
            genome_dict[genome.key] = genome
            genome.fitness = evaluator.eval_genome(genome, config)
            if genome.fitness > best_fitness:
                champ_key = genome.key
        if grad_steps == total_grad_steps:
            return
        else:
            execute_back_prop(genome_dict, champ_key, config)
            for _, genome in genomes:
                genome.fitness = evaluator.eval_genome(genome, config)
            grad_steps += 1
            self.eval_genomes(genomes, config, grad_steps)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    #logger = LogReporter("neat.log", evaluator.eval_genome)
    #pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
