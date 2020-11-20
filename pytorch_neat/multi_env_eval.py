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
from torch.utils.tensorboard import SummaryWriter


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

        __RENDER__ = False # batch size also needs to be below certain thresh to prevent desktop from death

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
    def __init__(
            self, 
            make_net, 
            activate_net, 
            batch_size=1,
             max_env_steps=None, 
            make_env=None, 
            envs=None, 
            track_macs=False,
            tb_write=True
            ):
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

        # default `log_dir` is "runs" - we'll be more specific here
        if tb_write:
            self.tb_writer = SummaryWriter('runs/ant_v2') 
        else:
            self.tb_writer = None

    def eval_genome(self, genome, config, **kwargs):
        # evaluate self.batch_size genomes simultaneously stepwise in multiple openai gym envs until all done 

        try: # on CTRL^C, close tb writer if its not None

            net = self.make_net(genome, config, self.batch_size) # net is torch module

            fitnesses = np.zeros(self.batch_size)

            # store state batch from each observation along trajectory to be able to calculate grad later
            states_trajectory = [torch.Tensor([env.reset() for env in self.envs]).to(dtype=net.dtype)]

            dones = [False] * self.batch_size

            step_num = 0

            __RENDER__ = False
            correlation_threshold = float("-inf")  # achieve this much to survive # TODO determine from samples
            correlation_scores = []
            BAD_JACOBIAN_FITNESS = -200
            track_jacobian_until = 8 # track for first few steps
            retain_graph = True # I think I always need this
            recurse_for_jacobian = False # use first states as grad input? or most recent states?

            if recurse_for_jacobian:
                states_trajectory[0].requires_grad_(True)
                grad_inputs = states_trajectory[0]

            net.set_grad(True)
            prev_activs, prev_outputs = net.init_activs_and_outputs(torch_method=torch.randn, require_grad=False)

            if self.tb_writer is not None:
                _probe_inputs_ = (states_trajectory[0], prev_activs, prev_outputs)
                if net.n_hidden > 0:
                    # ^ net must return Tensors only, see recurrent_net forward pass
                    try:
                        # use jit to find out module code graph
                        self.tb_writer.add_graph(net, _probe_inputs_)
                        self.tb_writer.close()
                    except RuntimeError:
                        print([type(t) for t in _probe_inputs_])
                        raise
            
            while True:
                # simulate in batch parallel environments, breaking if all done or max_env_steps reached
                step_num += 1
                if self.max_env_steps is not None and step_num == self.max_env_steps:
                    break

                ##### vvv https://arxiv.org/pdf/2006.04647.pdf vvv ####
                states = states_trajectory[-1]

                if step_num <= track_jacobian_until: # FIXME only backprop once at step trju

                    if not recurse_for_jacobian:
                        states.requires_grad_(True)
                        grad_inputs = states

                    net.zero_grad()

                    # activate network based on world state
                    outputs, macs, _params = self.activate_net(
                        net, inputs=(states, prev_activs, prev_outputs), step_num=step_num, track_macs=self.track_macs)
                    prev_outputs, prev_activs = outputs

                    print("actions:")
                    print(prev_outputs)

                    # compute backward pass to get jacobian of params w.r.t. to x
                    prev_outputs.backward(torch.ones_like(prev_outputs), retain_graph=retain_graph if step_num < track_jacobian_until else False)

                    # forward
                    assert grad_inputs.requires_grad == True
                    for p in net.parameters():
                        assert p.requires_grad == True, p

                    # TODO find out if this has to be set
                    # assert prev_activs == None or prev_activs.grad_fn is not None
                    assert prev_outputs.requires_grad == True

                    assert grad_inputs.grad is not None, "somehow the gradient from the RNN outputs was not backpropped to the inputs"

                    jacobian = grad_inputs.grad.clone().detach().reshape(self.batch_size, -1).numpy()

                    print("jacobian:")
                    print(jacobian.shape)
                    print(jacobian)

                    try:
                        correlation_score = eval_score(jacobian)
                    except Exception as e:
                        # got NaN
                        print(e)
                        correlation_score = np.nan
                    correlation_scores += [correlation_score]

                    for name, p in net.named_parameters():
                        if hasattr(p, "grad") and p.grad is not None:
                            if (p!=0).any():
                                print(f"found param '{name}' with non-zero jacobian:")
                                print(p.grad.clone().detach().numpy())

                else:
                    # normal action calculation
                    # dont need to store gradients anymore now so stop doing it to be speed
                    with torch.no_grad():
                        # activate network based on world state
                        outputs, macs, _params = self.activate_net(
                            net, inputs=(states, prev_activs, prev_outputs), step_num=step_num, track_macs=self.track_macs)
                        prev_outputs, prev_activs = outputs

                if self.track_macs:
                    self.CUMMACS += int(macs)
                
                if step_num == track_jacobian_until:
                    # evaluate fitnesses and if too low, like, don't even simulate any further, dawg

                    # FIXME uncomment this:
                    # assert len(correlation_scores) == track_jacobian_until, (correlation_scores, track_jacobian_until) 
                    average_correlation_score = sum(correlation_scores) / track_jacobian_until

                    print(f"after evaluating this net {track_jacobian_until} times, we decide")
                    if average_correlation_score < correlation_threshold:
                        print(f"it is not worth the effort, as it got only {average_correlation_score} on average during these evaluations")
                        return BAD_JACOBIAN_FITNESS
                    else:
                        print(f"we should investigate it further, as it got:")
                        print(f"\tSCORE: {str(average_correlation_score).upper()} ...")

                        # dont need to save grads anymore now
                        del states_trajectory[:-1]
                        prev_outputs = prev_outputs.detach()
                        prev_activs = prev_activs.detach() if prev_activs is not None else None
                        net.set_grad(False) 

                ##### ^^^ https://arxiv.org/pdf/2006.04647.pdf ^^^ ####

                # VVV after tracking jacobian for some steps, resume normally without grad VVV

                actions = prev_outputs.clone().detach().numpy()
                assert len(actions) == self.batch_size, (actions, type(actions), len(self.envs))

                leaf_states_require_grad = False # NEEDS TO BE FALSE always 
                states_ = torch.zeros_like(states_trajectory[-1]).to(dtype=states_trajectory[-1].dtype).requires_grad_(leaf_states_require_grad)

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

        except KeyboardInterrupt:
            if self.tb_writer is not None:
                self.tb_writer.close()
            raise # 'Aborted!'

def eval_score(jacob):
    """https://arxiv.org/pdf/2006.04647.pdf"""

    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    input(v)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))






