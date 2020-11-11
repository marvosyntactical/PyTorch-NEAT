import multiprocessing
import os

import math
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
task = "Ant-v2"

# phenotype networks map states (input nodes) to actions (output nodes)
# so to specify input and output nodes in make_net i want to know states and actions

global state_space, action_space, task_dimensionality, output_coords
task_dimensionality = 3
_test_env = gym.make(task)
# NOTE interpretation of 111 scalar values? TODO
state_space = _test_env.env.observation_space.shape[0] # 111
# NOTE interpretation of output coords: (aus mujoco antv2 cfg)
# hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3 
action_space = _test_env.env.action_space.shape[0] # 8

output_coords = [
    (1.0, -.5, -.5),
    (1.0, -1.0, -1.0),
    (1.0, .5, -.5),
    (1.0, 1.0, -1.0),
    (1.0, .5, .5),
    (1.0, 1.0, 1.0),
    (1.0, -.5, .5),
    (1.0, -1.0, 1.0),
]

# assert False, (_test_env.env.observation_space.low, _test_env.env.observation_space.high, _test_env.env.observation_space.shape)
# assert False, (action_space, state_space) # NOTE should be (8, 111)

del _test_env

def make_env():
    env = gym.make(task)
    return env

def get_in_coords(states=None):
    """ make input node arrangement specifically for ant task """

    input_coords = []
    # we will use a 3 dimensional substrate, coords laid out here
    # NOTE i have no idea what the interpretation of the 111 states is
    # so I will just arrange them in a circle of radius 1
    # moving them closer to the center in proportion to their state value if their state value is smaller than a 100
    precision = 3
    for input_idx in range(state_space):
        state_value_factor = min(.1,max(1,states[input_idx]/100)) if states is not None else 1
        coord = math.e**(2j*math.pi*input_idx/state_space)
        assert round((coord.imag**2+coord.real**2)**.5, 1) == 1., (coord.imag, coord.real)
        coord *= state_value_factor # down scale distance from center according to state magnitude

        y,z = round(coord.imag,precision),round(coord.real,precision)
        assert not (y == 0.0 and z == 0.0) , (y,z)
        input_coords.append((-1., y, z))

    # assert False, input_coords

    return input_coords

def make_net(genome, config, bs, state_space_dim=111, action_space_dim=8):
    #start by setting up a substrate for this bad ant boi
    params = {"initial_depth": 1,
            "max_depth": 5,
            "variance_threshold": 0.8,
            "band_threshold": 0.05,
            "iteration_level": 3,
            "division_threshold": 0.3,
            "max_weight": 34.0,
            "activation": "sigmoid"}
    # FIXME this can just be list
    joint_name_dict = {
        0: "hip_4",
        1: "ankle_4",
        2: "hip_1",
        3: "ankle_1",
        4: "hip_2",
        5: "ankle_2",
        6: "hip_3",
        7: "ankle_3",
    }
    assert len(joint_name_dict) == len(output_coords)

    leaf_names = []
    for i in range(len(output_coords[0])):
        leaf_names.append(str(i) + "_in")
        leaf_names.append(str(i) + "_out")
    input_coords = get_in_coords()

    [cppn] = create_cppn(genome, config, leaf_names, ['cppn_out'])
    net_builder = ESNetwork(Substrate(input_coords, output_coords), cppn, params)
    net = net_builder.create_phenotype_network_nd('./genome_vis')
    return net

def reset_substrate(states):
    input_coords = get_in_coords(states)
    return Substrate(input_coords, output_coords)

def activate_net(net, states, **kwargs):
    #print(states)
    #new_sub = reset_substrate(states[0])
    #net.reset_substrate(new_sub)
    #network = net.create_phenotype_network_nd() 
    outputs = net.activate(states).numpy()
    #print(outputs[:,0])
    return outputs > 0.5 # NOTE warum konvertieren wir hier in bool?


@click.command()
@click.option("--n_generations", type=int, default=100)
def run(n_generations):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "_ant.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # should be 1 for NEAT
    # but high for NASnosearch pruning

    batch_size = 2

    print(f"running this particular net in {batch_size} environments in parallel")

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps, batch_size=batch_size
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
    logger = LogReporter("neat.log", evaluator.eval_genome)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
