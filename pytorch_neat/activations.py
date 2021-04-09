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
import torch.nn.functional as F


def sigmoid_activation(x):
    return torch.sigmoid(x)


def tanh_activation(x):
    return torch.tanh(x)


def abs_activation(x):
    return torch.abs(x)


def gauss_activation(x):
    return torch.exp(-5.0 * x**2)


def identity_activation(x):
    return x


def sin_activation(x):
    return torch.sin(x)


def relu_activation(x):
    return F.relu(x)

# hard activations to expand the pool of piecewise linear activations to pick from during evo

def hard_sigmoid_activation(x):
    return F.hard_sigmoid(x)

def hard_tanh_activation(x):
    return F.hardtanh(x)


str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'abs': abs_activation,
    'gauss': gauss_activation,
    'identity': identity_activation,
    'sin': sin_activation,
    'relu': relu_activation,
    'hard_sigmoid': hard_sigmoid_activation,
    'hard_tanh': hard_tanh_activation,
}

piecewise_linear_activations = {'relu', 'hard_sigmoid', 'hard_tanh', 'abs', 'identity'}
assert set([pw_linear in str_to_activation for pw_linear in piecewise_linear_activations]) == set([True])

