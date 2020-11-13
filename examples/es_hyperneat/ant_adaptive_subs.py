import multiprocessing
import os

import time

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
from pytorch_neat.recurrent_net import RecurrentNet

from thop import *

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

def make_net(genome, config, bs, state_space_dim=111, action_space_dim=8) -> RecurrentNet:
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
    return net # RecurrentNet ((not necessarily actually recurrent))

def reset_substrate(states):
    input_coords = get_in_coords(states)
    return Substrate(input_coords, output_coords)

def activate_net(net, states, track_macs=True, custom_ops=None, verbose=False, **kwargs):
    """
    Activates net, prints time taken and optionally prints macs and params of net forward pass
    """

    # start = time.time()

    if track_macs:
        outputs, macs, params = profile(net, inputs=(states,), custom_ops=custom_ops, verbose=verbose)
        outputs = outputs.numpy()

        # print(f"Forward pass took {int(macs)} MAC operations for this net with {int(params)} params")
        # print(f"Took {round(end-start,5)}s to activate Network.")

    else:

        outputs = net(states) # torch.Tensor
        outputs = outputs.numpy()
        macs, params = None, None

    # Threshold activation between 0 and 1:
    return outputs > 0.5, macs, params # NOTE warum konvertieren wir hier in bool though?




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

    BATCHSIZE = 5

    print(f"evaluating this particular net in {BATCHSIZE} environments in parallel")

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps, batch_size=BATCHSIZE, track_macs=True
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    antlog = "ant.log"
    if "DISPLAY" in os.environ:
        # no display no space saving ~_{°n°}/
        if os.path.isfile(antlog) and "n" not in input("Remove previous 'ant.log? \n\t[Y/n]"):
            os.remove(antlog)

    # comment these 2 lines to not log
    logger = LogReporter(antlog, evaluator, log_macs=True)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
