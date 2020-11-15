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
import torch


class MultiEnvEvaluator:
    def __init__(self, make_net, activate_net, batch_size=1, max_env_steps=None, make_env=None, envs=None, track_macs=False):
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
        self.track_macs = bool(track_macs)
        if self.track_macs:
            self.CUMMACS = 0

    def eval_genome(self, genome, config, debug=False):
        # evaluate self.batch_size genomes simultaneously stepwise in multiple openai gym envs until all done 
        net = self.make_net(genome, config, self.batch_size) # net is torch module?

        fitnesses = np.zeros(self.batch_size)
        states = [env.reset() for env in self.envs]
        dones = [False] * self.batch_size

        __RENDER__ = True

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break

            # activate network based on world state
            # FIXME activate net inputs into single phenotype the world state
            # but states is batch list of world states??
            actions, macs, _params = self.activate_net(
                net, states, prev_activs=prev_activs, prev_outputs=prev_outputs, step_num=step_num, track_macs=self.track_macs)
            if self.track_macs:
                self.CUMMACS += int(macs)

            assert len(actions) == self.batch_size, (actions, type(actions), len(self.envs))

            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    # openAI gym env.step returns: 
                    # world state, reward, end of rollout?/dead?/finish?, debug info
                    state, reward, done, _ = env.step(action) 
                    if __RENDER__ and self.batch_size <= 5:
                        # beware of the big bad wolf
                        env.render()
                    fitnesses[i] += reward
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        return sum(fitnesses) / len(fitnesses)

class CovarianceFilterEvaluator(MultiEnvEvaluator):
    """
    Try to use https://github.com/BayesWatch/nas-without-training/blob/master/search.py to prune
    """
    def __init__(self, make_net, activate_net, batch_size=1, max_env_steps=None, make_env=None, envs=None, track_macs=False):
        """
        :param batch_size: how many phenotypes (or genomes) to evaluate at once
        """
        # use either list of initialized environments or making function
        if envs is None:
            self.envs = [make_env() for _ in range(batch_size)]
        else:
            assert len(envs) == batch_size
            self.envs = envs
        self.make_net = make_net
        self.activate_net = activate_net
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.track_macs = bool(track_macs)
        if self.track_macs:
            self.CUMMACS = 0

    def eval_genome(self, genome, config):
        # evaluate self.batch_size genomes simultaneously stepwise in multiple openai gym envs until all done 
        net = self.make_net(genome, config, self.batch_size) # net is torch module?

        fitnesses = np.zeros(self.batch_size)
        states_trajectory = [torch.Tensor([env.reset() for env in self.envs]).to(dtype=net.dtype)]
        dones = [False] * self.batch_size

        step_num = 0

        __RENDER__ = True
        track_jacobian_until = 3
        correlation_threshold = .5  # TODO determine from samples
        correlation_scores = []
        BAD_JACOBIAN_FITNESS = -200


        net.set_grad(True)
        prev_activs, prev_outputs = net.init_activs_and_outputs()
        
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break

            ##### vvv https://arxiv.org/pdf/2006.04647.pdf vvv ####
            states = states_trajectory[-1]

            if step_num <= track_jacobian_until:
                # prep inputs for getting jacobian
                states.requires_grad_(True)
                net.zero_grad()

                # activate network based on world state
                outputs, macs, _params = self.activate_net(
                    net, inputs=(states, prev_activs, prev_outputs), step_num=step_num, track_macs=self.track_macs)
                prev_outputs, prev_activs = outputs[0], outputs[1]

                # compute backward pass to get jacobian of params w.r.t. to x
                try:
                    prev_outputs.backward(torch.ones_like(prev_outputs), retain_graph=True)
                except RuntimeError as e:
                    print(correlation_scores)
                    raise e

                # forward
                assert states.requires_grad == True
                for p in net.parameters():
                    assert p.requires_grad == True, p
                assert prev_activs == None or prev_activs.requires_grad == True
                assert prev_outputs.requires_grad == True

                assert states.grad is not None, "somehow the gradient from the RNN outputs was not backpropped to the inputs"

                jacobian = states.grad.clone().detach().reshape(self.batch_size, -1).cpu().numpy()

                try:
                    correlation_score = eval_score(jacobian)
                except Exception as e:
                    print(e)
                    correlation_score = np.nan
                correlation_scores += [correlation_score]

            else:
                # normal action calculation
                # dont need to store gradients anymore now so stop doing it to be speed
                with torch.no_grad():
                    # activate network based on world state
                    outputs, macs, _params = self.activate_net(
                        net, inputs=(states, prev_activs, prev_outputs), step_num=step_num, track_macs=self.track_macs)
                    prev_outputs, prev_activs = outputs[0], outputs[1]

            if self.track_macs:
                self.CUMMACS += int(macs)
            
            if step_num == track_jacobian_until:
                # evaluate fitnesses and if too low, like, don't even simulate any further, dawg
                assert len(correlation_scores) == track_jacobian_until, (correlation_scores, track_jacobian_until)
                average_correlation_score = sum(correlation_scores) / track_jacobian_until
                assert False, average_correlation_score

                print(f"after evaluating this net for {track_jacobian_until} times, we decide")
                if average_correlation_score < correlation_threshold:
                    print(f"it is not worth the effort, as it got only {average_correlation_score} on average during these evaluations")
                    return BAD_JACOBIAN_FITNESS
                else:
                    print(f"we should investigate it further, as it got {average_correlation_score}...")

                    # dont need to save grads on these anymore now
                    del states_trajectory[:-1]
                    prev_outputs.requires_grad = False
                    prev_activs.requires_grad = False

                    net.set_grad(False) 

            ##### ^^^ https://arxiv.org/pdf/2006.04647.pdf ^^^ ####

            actions = prev_outputs.clone().detach().numpy()
            assert len(actions) == self.batch_size, (actions, type(actions), len(self.envs))

            with torch.no_grad():
                states_ = torch.zeros_like(states_trajectory[-1]).to(dtype=states_trajectory[-1].dtype).requires_grad_(states_trajectory[-1].requires_grad)

            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    # openAI gym env.step returns: 
                    # world state, reward, end of rollout?/dead?/finish?, debug info
                    state, reward, done, _ = env.step(action) 
                    if __RENDER__ and self.batch_size <= 5:
                        # beware bof bhe big bad blue breen bof beath
                        env.render()
                    fitnesses[i] += reward
                    if not done:
                        states_[i] = torch.Tensor(state).to(dtype=states_trajectory[-1].dtype)
                    dones[i] = done
            if all(dones):
                break
            if step_num <= track_jacobian_until:
                states_trajectory += [states_]
            else:
                states_trajectory = [states_]

        return sum(fitnesses) / len(fitnesses)

def eval_score(jacob):
    """https://arxiv.org/pdf/2006.04647.pdf"""

    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))






