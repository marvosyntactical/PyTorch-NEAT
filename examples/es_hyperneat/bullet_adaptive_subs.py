import multiprocessing
import os
import time
import math
import fire

import gym
import pybulletgym

import torch
import numpy as np

import thop

import neat

# HACK FIXME
import sys
sys.path.append("../../")

from pytorch_neat.multi_env_eval import CovarianceFilterEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.es_hyperneat import ESNetwork
from pytorch_neat.substrate import Substrate
from pytorch_neat.cppn import create_cppn
from pytorch_neat.recurrent_net import RecurrentNet, DUMMY_NET
from pytorch_neat.activations import piecewise_linear_activations


max_env_steps = 30 # TODO mujoco seems to do 5* this many steps, TODO find out frame skip settings
task = "AntMuJoCoEnv-v0"

# phenotype networks map states (input nodes) to actions (output nodes)
# so to specify input and output nodes in make_net id like to know states and actions

global state_space, action_space, task_dimensionality, output_coords
task_dimensionality = 3
_test_env = gym.make(task)
# NOTE interpretation of 111 scalar values? TODO
state_space = _test_env.env.observation_space.shape[0] # 111
# NOTE interpretation of output coords: (aus mujoco bulletv2 cfg)
# hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3
action_space = _test_env.env.action_space.shape[0] # 8

# assert False, (_test_env.env.observation_space.low, _test_env.env.observation_space.high, _test_env.env.observation_space.shape)
# assert False, (action_space, state_space) # NOTE should be (8, 111)
del _test_env

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

# assert False, "build proper output coords for task !!!"

def make_env():
    env = gym.make(task)
    # env.render()
    return env

def get_in_coords(states=None):
    """ make input node arrangement specifically for bullet task """

    input_coords = []
    # we will use a 3 dimensional substrate, coords laid out here
    # NOTE i have no idea what the interpretation of the 111 states is
    # so I will just arrange them in a circle of radius 1
    # moving them closer to the center in proportion to their state value if their state value is smaller than a 100
    precision = 3
    for input_idx in range(state_space):

        coord = math.e**(2j*math.pi*input_idx/state_space)
        assert round((coord.imag**2+coord.real**2)**.5, 1) == 1., (coord.imag, coord.real)

        state_value_factor = min(.1,max(1,states[input_idx]/100)) if states is not None else 1
        coord *= state_value_factor # down scale distance from center according to state magnitude

        y,z = round(coord.imag,precision),round(coord.real,precision)
        assert not (y == 0.0 and z == 0.0) , (y,z)
        input_coords.append((-1., y, z))

    return input_coords


def make_dummy_net(genome, config, batch_size, state_space_dim=111, action_space_dim=8) -> DUMMY_NET:
    return DUMMY_NET(
            in_feat=state_space_dim,
            hidden=20,
            out_feat=action_space_dim,
            )

def make_net(genome, config, batch_size, state_space_dim=111, action_space_dim=8) -> RecurrentNet:
    #start by setting up a substrate for this bad ant boi
    params = {
        "initial_depth": 1,
        "max_depth": 3,
        "variance_threshold": 0.8,
        "band_threshold": 0.05,
        "division_threshold": 0.3,
        "max_weight": 10.0,
        "activation": "hard_tanh"
    }
    assert params["activation"]in piecewise_linear_activations, f"for NAS without training, network activation needs to be piecewise linear"

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

    leaf_names = []
    for i in range(len(output_coords[0])):
        leaf_names.append(str(i) + "_in")
        leaf_names.append(str(i) + "_out")
    input_coords = get_in_coords()

    [cppn] = create_cppn(genome, config, leaf_names, ['cppn_out'])
    net_builder = ESNetwork(Substrate(input_coords, output_coords), cppn, params)
    net = net_builder.create_phenotype_network_nd()
    return net # RecurrentNet ((not necessarily actually recurrent))

def reset_substrate(states):
    input_coords = get_in_coords(states)
    return Substrate(input_coords, output_coords)

def activate_net(net, inputs, track_macs=True, custom_ops=None, verbose=False, **kwargs) ->torch.Tensor:
    """
    Activates net, prints time taken and optionally prints macs and params of net forward pass
    """


    if track_macs:
        # FIX profiling
        outputs, macs, params = thop.profile(net, inputs=inputs, custom_ops=custom_ops, verbose=verbose)
    else:
        outputs = net(*inputs) # this is also what is done with inputs inside profile:
        # silver@hal9000:~/projects/evo/PyTorch-NEAT/pytorch_neat/pytorch-OpCounter/thop/profile.py

        macs, params = None, None

    assert type(outputs) == type(()) # PyTorch_Neat.recurrent_net.forward returns (outputs, activations)
    assert len(outputs) == 2, len(outputs)

    # Dont use thresh activation here
    return outputs, macs, params

def factory_eval_genomes(evaluator):
    def eval_genomes(genomes, config) -> None:
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)
    return eval_genomes


def run(
        cfg: str = "bullet.cfg",
	log_file: str = "bullet.log",
	n_generations: int = 100,
	batch_size: int = 2,
	log_macs: bool = False,
	dummy: bool = False,
    ):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), cfg)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    net_factory_function = make_net if not dummy else make_dummy_net

    print(f"evaluating this particular net in {batch_size} environments in parallel")

    evaluator = CovarianceFilterEvaluator(
        net_factory_function,
	activate_net,
	make_env=make_env,
	max_env_steps=max_env_steps,
	batch_size=batch_size,
	track_macs=log_macs
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    # no display no space saving ~_{°n°}/
    if 0 and "DISPLAY" in os.environ:
        if os.path.isfile(log_file) and "n" not in input("Remove previous 'bullet.log? \n\t[Y/n]"):
            os.remove(log_file)

    # comment these 2 lines to not log
    logger = LogReporter(log_file, evaluator, log_macs=log_macs)
    pop.add_reporter(logger)

    pop.run(factory_eval_genomes(evaluator), n_generations)


if __name__ == "__main__":
    fire.Fire(run)
