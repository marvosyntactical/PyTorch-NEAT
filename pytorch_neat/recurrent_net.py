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

import torch
import torch.nn as nn
import numpy as np
from .activations import str_to_activation, sigmoid_activation
import json

# def sparse_mat(shape, conns):
#     idxs, weights = conns
#     if len(idxs) > 0:
#         idxs = torch.LongTensor(idxs).t()
#         weights = torch.FloatTensor(weights)
#         mat = torch.sparse.FloatTensor(idxs, weights, shape)
#     else:
#         mat = torch.sparse.FloatTensor(shape[0], shape[1])
#     return mat


def create_linear_layer_from_conns(shape, conns, dtype=torch.float64, bias=None):
    """ 
    create nn.Linear(*shape) layer with specified connection weights
    dtype should be differentiable
    for RecurrentNet, bias should be None atm; since for several Linear layers
    a global bias gets added only afterwards
    """
    assert dtype.is_floating_point or dtype.is_complex, dtype

    mat = torch.zeros(shape, dtype=dtype)
    layer = nn.Linear(*shape, bias=bias)

    idxs, weights = conns
    assert len(idxs) == len(weights)

    if not len(idxs) == 0:
        # insert specified connection weights if there are any
        rows, cols = np.array(idxs).transpose()
        w = torch.tensor(weights, dtype=dtype)

        mat[torch.LongTensor(rows), torch.LongTensor(cols)] = w

    layer.weight.data = mat
    assert isinstance(layer, nn.Module), type(layer)
    # input((type(layer), dtype, len(weights)))
    return layer

class RecurrentNet(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs,
                 input_to_hidden, hidden_to_hidden, output_to_hidden,
                 input_to_output, hidden_to_output, output_to_output,
                 hidden_responses, output_responses,
                 hidden_biases, output_biases,
                 batch_size=1,
                 use_current_activs=False,
                 activation=str_to_activation["relu"],
                 n_internal_steps=2,
                 dtype=torch.float64):
        super(RecurrentNet, self).__init__()

        self.use_current_activs = use_current_activs
        self.activation = activation
        self.n_internal_steps = n_internal_steps
        self.dtype = dtype

        self.batch_size = batch_size

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        if n_hidden > 0:
            self.input_to_hidden = create_linear_layer_from_conns(
                (n_hidden, n_inputs), input_to_hidden, dtype=dtype)
            self.hidden_to_hidden = create_linear_layer_from_conns(
                (n_hidden, n_hidden), hidden_to_hidden, dtype=dtype)
            self.output_to_hidden = create_linear_layer_from_conns(
                (n_hidden, n_outputs), output_to_hidden, dtype=dtype)
            self.hidden_to_output = create_linear_layer_from_conns(
                (n_outputs, n_hidden), hidden_to_output, dtype=dtype)
        
        self.input_to_output = create_linear_layer_from_conns(
            (n_outputs, n_inputs), input_to_output, dtype=dtype)
        self.output_to_output = create_linear_layer_from_conns(
            (n_outputs, n_outputs), output_to_output, dtype=dtype)

        if n_hidden > 0:
            self.register_parameter(name="hidden_responses", param=
                nn.Parameter(torch.tensor(hidden_responses).to(dtype=dtype)))
            self.register_parameter(name="hidden_biases", param=
                nn.Parameter(torch.tensor(hidden_biases).to(dtype=dtype)))
        self.register_parameter(name="output_responses", param=
            nn.Parameter(torch.tensor(output_responses).to(dtype=dtype)))
        self.register_parameter(name="output_biases", param=
            nn.Parameter(torch.tensor(output_biases).to(dtype=dtype)))

        self.batch_size=batch_size

    def set_grad(self, requires_grad: bool=True):
        self.requires_grad_(requires_grad) 
        self.requires_grad = requires_grad # to check this on first step for activ & output initialization

    def init_activs_and_outputs(self, torch_method=torch.randn, require_grad=None) -> (torch.Tensor, torch.Tensor):
        require_grad = self.requires_grad if require_grad is None else require_grad
        initial_activations = torch_method(self.batch_size, self.n_hidden, dtype=self.dtype).requires_grad_(require_grad)
        initial_outputs = torch_method(self.batch_size, self.n_outputs, dtype=self.dtype).requires_grad_(require_grad)
        return initial_activations, initial_outputs

    def forward(self, inputs, prev_activs, prev_outputs):
        '''
        inputs: (batch_size, n_inputs)

        returns: (batch_size, n_outputs)
        '''
        activs = prev_activs # to be able to choose which activations to use for outputs

        # precalculate output layer inputs before hidden loop
        output_inputs = self.input_to_output(inputs) + self.output_to_output(prev_outputs)

        if self.n_hidden > 0:
            ### INTERNAL LOOP ###
            for _ in range(self.n_internal_steps):
                activs = self.activation(
                    self.hidden_responses * (
                        self.input_to_hidden(inputs) +
                        self.hidden_to_hidden(activs) +
                        self.output_to_hidden(prev_outputs)
                        ) + self.hidden_biases)

            if self.use_current_activs:
                activs_for_output = activs
            else:
                activs_for_output = prev_activs

            output_inputs += self.hidden_to_output(activs_for_output)
        else:
            activs_for_output = None

        outputs = self.activation(self.output_responses * output_inputs + self.output_biases)
        return outputs, activs_for_output

    @staticmethod
    def create(genome, config, batch_size=1, activation=sigmoid_activation,
               prune_empty=False, use_current_activs=False, n_internal_steps=1):
        from neat.graphs import required_for_output

        genome_config = config.genome_config
        required = required_for_output(
            genome_config.input_keys, genome_config.output_keys, genome.connections)
        if prune_empty:
            nonempty = {conn.key[1] for conn in genome.connections.values() if conn.enabled}.union(
                set(genome_config.input_keys))

        input_keys = list(genome_config.input_keys)
        hidden_keys = [k for k in genome.nodes.keys()
                       if k not in genome_config.output_keys]
        output_keys = list(genome_config.output_keys)

        hidden_responses = [genome.nodes[k].response for k in hidden_keys]
        output_responses = [genome.nodes[k].response for k in output_keys]

        hidden_biases = [genome.nodes[k].bias for k in hidden_keys]
        output_biases = [genome.nodes[k].bias for k in output_keys]

        if prune_empty:
            for i, key in enumerate(output_keys):
                if key not in nonempty:
                    output_biases[i] = 0.0

        n_inputs = len(input_keys)
        n_hidden = len(hidden_keys)
        n_outputs = len(output_keys)

        input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
        hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
        output_key_to_idx = {k: i for i, k in enumerate(output_keys)}

        def key_to_idx(key):
            if key in input_keys:
                return input_key_to_idx[key]
            elif key in hidden_keys:
                return hidden_key_to_idx[key]
            elif key in output_keys:
                return output_key_to_idx[key]

        input_to_hidden = ([], [])
        hidden_to_hidden = ([], [])
        output_to_hidden = ([], [])
        input_to_output = ([], [])
        hidden_to_output = ([], [])
        output_to_output = ([], [])

        for conn in genome.connections.values():
            if not conn.enabled:
                continue

            i_key, o_key = conn.key
            if o_key not in required and i_key not in required:
                continue
            if prune_empty and i_key not in nonempty:
                print('Pruned {}'.format(conn.key))
                continue

            i_idx = key_to_idx(i_key)
            o_idx = key_to_idx(o_key)

            if i_key in input_keys and o_key in hidden_keys:
                idxs, vals = input_to_hidden
            elif i_key in hidden_keys and o_key in hidden_keys:
                idxs, vals = hidden_to_hidden
            elif i_key in output_keys and o_key in hidden_keys:
                idxs, vals = output_to_hidden
            elif i_key in input_keys and o_key in output_keys:
                idxs, vals = input_to_output
            elif i_key in hidden_keys and o_key in output_keys:
                idxs, vals = hidden_to_output
            elif i_key in output_keys and o_key in output_keys:
                idxs, vals = output_to_output
            else:
                raise ValueError(
                    'Invalid connection from key {} to key {}'.format(i_key, o_key))

            idxs.append((o_idx, i_idx))  # to, from
            vals.append(conn.weight)

        return RecurrentNet(n_inputs, n_hidden, n_outputs,
                            input_to_hidden, hidden_to_hidden, output_to_hidden,
                            input_to_output, hidden_to_output, output_to_output,
                            hidden_responses, output_responses,
                            hidden_biases, output_biases,
                            batch_size=batch_size,
                            activation=activation,
                            use_current_activs=use_current_activs,
                            n_internal_steps=n_internal_steps)

    @staticmethod
    def create_from_es(in_nodes, out_nodes, node_evals, batch_size=1, activation=sigmoid_activation,
               prune_empty=False, use_current_activs=False, n_internal_steps=1):
        hidden_responses = [1.0 for k in range(len(node_evals)-(len(in_nodes)+len(out_nodes)))]
        output_responses = [1.0 for k in range(len(out_nodes))]

        hidden_biases = [1.0 for k in range(len(node_evals)-(len(in_nodes)+len(out_nodes)))]
        output_biases = [1.0 for k in range(len(out_nodes))]

        input_key_to_idx = {k: i for i, k in enumerate(in_nodes)}
        output_key_to_idx = {k: i for i, k in enumerate(out_nodes)}
        hidden_key_to_idx = {}

        hidden_idx = -1

        def key_to_idx(key, hid_idx):
            if key in in_nodes:
                return input_key_to_idx[key]
            elif key in out_nodes:
                return output_key_to_idx[key]
            elif key in hidden_key_to_idx.keys():
                return hidden_key_to_idx[key]
            else:
                hid_idx += 1
                hidden_key_to_idx[key] = hid_idx
                return hid_idx

        input_to_hidden = ([], [])
        hidden_to_hidden = ([], [])
        output_to_hidden = ([], [])
        input_to_output = ([], [])
        hidden_to_output = ([], [])
        output_to_output = ([], [])

        # this could be optimized by first checking in out or hidden 
        # of ikey but for now this is how im doing it to keep it looking familiar
        for conn in node_evals:
            #pruning is done in the eshyperneat class
            i_key = conn[0]
            for x in conn[5]:
                o_key = x[0]
                i_idx = key_to_idx(i_key, hidden_idx)
                o_idx = key_to_idx(o_key, hidden_idx)
                add_conn = True
                if i_key in in_nodes and o_key not in out_nodes:
                    idxs, vals = input_to_hidden
                elif i_key not in in_nodes and i_key not in out_nodes and o_key not in in_nodes and o_key not in out_nodes:
                    idxs, vals = hidden_to_hidden
                elif i_key in out_nodes and o_key not in out_nodes and o_key not in in_nodes:
                    idxs, vals = output_to_hidden
                elif i_key in in_nodes and o_key in out_nodes:
                    idxs, vals = input_to_output
                elif i_key not in in_nodes and i_key not in out_nodes and o_key in out_nodes:
                    idxs, vals = hidden_to_output
                elif i_key in out_nodes and o_key in out_nodes:
                    idxs, vals = output_to_output
                else:
                    add_conn = False
                if add_conn == True:    
                    idxs.append((o_idx, i_idx))  # to, from
                    vals.append(float(x[1]))
        
        return RecurrentNet(len(in_nodes), len(node_evals)-(len(in_nodes)+len(out_nodes)), len(out_nodes),
                            input_to_hidden, hidden_to_hidden, output_to_hidden,
                            input_to_output, hidden_to_output, output_to_output,
                            hidden_responses, output_responses,
                            hidden_biases, output_biases,
                            batch_size=batch_size,
                            activation=activation,
                            use_current_activs=use_current_activs,
                            n_internal_steps=n_internal_steps)


class DUMMY_NET(nn.Module):

    def __init__(
            self, 
            in_feat=111, 
            hidden=5, 
            out_feat=8,
            batch_size=5,
            dtype=torch.float32,
            ):
        super(DUMMY_NET, self).__init__()

        self.activation = nn.ReLU()

        self.fc1 = nn.Linear(in_feat, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_feat)

        self.n_hidden = hidden
        self.n_outputs = out_feat
        self.n_internal_steps = 3
        self.batch_size = batch_size
        self.dtype = dtype
        self.use_current_activs = True

    def set_grad(self, requires_grad: bool=True):
        self.requires_grad_(requires_grad) 
        self.requires_grad = requires_grad # to check this on first step for activ & output initialization
    
    def init_activs_and_outputs(self, torch_method=torch.zeros, require_grad=None) -> (torch.Tensor, torch.Tensor):
        require_grad = self.requires_Grad if require_grad is None else require_grad
        initial_activations = torch_method(self.batch_size, self.n_hidden, dtype=self.dtype).requires_grad_(require_grad)
        initial_outputs = torch_method(self.batch_size, self.n_outputs, dtype=self.dtype).requires_grad_(require_grad)
        return initial_activations, initial_outputs

    def forward(self, inputs, prev_activs, prev_outputs):
        # sane ff implementation

        x = self.fc1(inputs)
        activs = self.activation(x)

        if self.n_hidden > 0:
            for _ in range(self.n_internal_steps):
                activs = self.activation(self.fc2(activs))

        if self.use_current_activs:
            activs_for_output  = activs
        else:
            activs_for_output = prev_activs
        outputs = self.fc3(activs_for_output)
        return (outputs, activs_for_output)
