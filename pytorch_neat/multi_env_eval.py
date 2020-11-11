# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import numpy as np


class MultiEnvEvaluator:
    def __init__(self, make_net, activate_net, batch_size=1, max_env_steps=None, make_env=None, envs=None):
        """

        :param batch_size: how many phenotypes (or genomes) to evaluate at once

        """
        # use either list of initialized environments or making function
        if envs is None:
            self.envs = [make_env() for _ in range(batch_size)]
        else:
            self.envs = envs
        self.make_net = make_net
        self.activate_net = activate_net
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps

    def eval_genome(self, genome, config, debug=False):
        # evaluate self.batch_size genomes simultaneously stepwise in multiple openai gym envs until all done 
        net = self.make_net(genome, config, self.batch_size)

        fitnesses = np.zeros(self.batch_size)
        states = [env.reset() for env in self.envs]
        dones = [False] * self.batch_size

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break

            # activate network based on world state
            # FIXME activate net inputs into single phenotype the world state
            # but states is batch list of world states??
            actions = self.activate_net(
                net, states, debug=bool(debug), step_num=step_num)

            assert len(actions) == self.batch_size, (actions, type(actions), len(self.envs))

            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    # openAI gym env.step returns: 
                    # world state, reward, end of rollout?/dead?/finish?, debug info
                    state, reward, done, _ = env.step(action) 
                    fitnesses[i] += reward
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        return sum(fitnesses) / len(fitnesses)

def CovarianceFilterEvaluator(MultiEnvEvaluator):
    def __init__(self, *args, **kwargs):
        super(CovarianceFilterEvaluator, self).__init__(*args, **kwargs)
    
    def eval_genome(self, genome, config, debug=False):
        raise NotImplementedError(f"IMPLEMENT ME")
        # TODO are these different genomes/nets?
        # TODO implement NAS rejection sampling here FIXME NOTE TODO
        # evaluate self.batch_size genomes simultaneously stepwise in multiple openai gym envs until all done 
        net = self.make_net(genome, config, self.batch_size)

        assert False, net.cppn

        fitnesses = np.zeros(self.batch_size)
        states = [env.reset() for env in self.envs]
        dones = [False] * self.batch_size

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(
                    net, states, debug=True, step_num=step_num)
            else:
                actions = self.activate_net(net, states)
            assert len(actions) == len(self.envs)
            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    # openAI gym env.step returns: 
                    # world state, reward, end of rollout?/dead?/finish?, debug info
                    state, reward, done, _ = env.step(action) 
                    fitnesses[i] += reward
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        return sum(fitnesses) / len(fitnesses)
