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
from neat.graphs import required_for_output

from .activations import str_to_activation
from .aggregations import str_to_aggregation

# Pointers:
# CPPN: http://axon.cs.byu.edu/Dan/778/papers/NeuroEvolution/stanley5.pdf
# cppn is used for encoding a neural network in the paper
# HyperNEAT: https://stars.library.ucf.edu/cgi/viewcontent.cgi?article=3177&context=facultybib2000

# Philosophy:
# cppn is meant for n-dimensional spatial tasks
# cppn is a NN that maps from 2 points (nodes in output NN) in connectivity space to how large the weight between themshould be
# and thus learns a spatial encoding of how a NN should be arranged in connectivity space
# cppn: 2n Input nodes, progressively more hidden nodes (use NEAT to evolve), 1 output node

# Implementation:
# cppn is viewed as directed graph starting at *output* node
# thus a node's children are its *incoming* nodes

class Node:
    def __init__(
        self,
        children,
        weights,
        response,
        bias,
        activation,
        aggregation,
        name=None,
        leaves=None,
    ):
        """
        children: list of Nodes,
            incoming nodes
        weights: list of floats,
            incoming weights
        response: float,
            the node's multiplication value,
            this is evolved by NEAT
        bias: float,
            added to activation at the end of calculation
        activation: torch function from .activations
        aggregation: torch function from .aggregations,
            contrary to neat-python, only allow sum and mul as aggregations,
            aggregate elementwise product of input, weights into scalar
        name: str
        leaves: dict of Leaves
        """
        self.children = children
        self.leaves = leaves
        self.weights = weights
        self.response = response
        self.bias = bias
        self.activation = activation
        self.activation_name = activation
        self.aggregation = aggregation
        self.aggregation_name = aggregation
        self.name = name
        if leaves is not None:
            assert isinstance(leaves, dict)
        self.leaves = leaves
        self.activs = None
        self.is_reset = None

    def __repr__(self):
        header = "Node({}, response={}, bias={}, activation={}, aggregation={})".format(
            self.name,
            self.response,
            self.bias,
            self.activation_name,
            self.aggregation_name,
        )
        child_reprs = []
        for w, child in zip(self.weights, self.children):
            child_reprs.append(
                "    <- {} * ".format(w) + repr(child).replace("\n", "\n    ")
            )
        return header + "\n" + "\n".join(child_reprs)

    def activate(self, xs, shape):
        """
        This is where computation happens (=> its nodewise)
        TODO: Is this parallelized somewhere ?

        xs: list of torch tensors
        shape: output tensor shape FIXME shouldnt this be the same as xs.shape??
        """
        if not xs:
            # no input or all zero, output will just be bias everywhere
            return torch.full(shape, self.bias)

        # for all incoming connections, multiply its weight with incoming data
        # ("elementwise product")
        inputs = [w * x for w, x in zip(self.weights, xs)] 
        try:
            # aggregation chosen by NEAT on node creation FIXME correct?
            pre_activs = self.aggregation(inputs) 

            # calc node's activation with inputs and bias;
            # these attributes are all evolved/chosen by NEAT FIXME correct?
            activs = self.activation(self.response * pre_activs + self.bias) 
            assert activs.shape == shape, "Wrong shape for node {}".format(self.name)
        except Exception:
            raise Exception("Failed to activate node {}".format(self.name))
        return activs

    def get_activs(self, shape):
        # starting at the output node of the CPPN, recursively get the activations of connected incoming nodes
        if self.activs is None:
            xs = [child.get_activs(shape) for child in self.children]
            self.activs = self.activate(xs, shape)
        return self.activs

    def __call__(self, **inputs):
        """
        Signature: TODO
        shape of first arg determines shape of activation tensors
        """
        assert self.leaves is not None
        assert inputs
        if "input_dict" in inputs:
            inputs = inputs["input_dict"]
        shape = list(inputs.values())[0].shape
        self.reset()
        for name in self.leaves.keys():
            assert (
                inputs[name].shape == shape
            ), "Wrong activs shape for leaf {}, {} != {}".format(
                name, inputs[name].shape, shape
            )
            self.leaves[name].set_activs(torch.Tensor(inputs[name]))
        return self.get_activs(shape)

    def _prereset(self):
        if self.is_reset is None:
            self.is_reset = False
            for child in self.children:
                child._prereset()  # pylint: disable=protected-access

    def _postreset(self):
        if self.is_reset is not None:
            self.is_reset = None
            for child in self.children:
                child._postreset()  # pylint: disable=protected-access

    def _reset(self):
        if not self.is_reset:
            self.is_reset = True
            self.activs = None
            for child in self.children:
                child._reset()  # pylint: disable=protected-access

    def reset(self):
        self._prereset()  # pylint: disable=protected-access
        self._reset()  # pylint: disable=protected-access
        self._postreset()  # pylint: disable=protected-access



class Leaf:
    # a leaf of our directed CPPN graph is an *input* node of the CPPN
    def __init__(self, name=None):
        self.activs = None
        self.name = name

    def __repr__(self):
        return "Leaf({})".format(self.name)

    def set_activs(self, activs):
        self.activs = activs

    def get_activs(self, shape):
        assert self.activs is not None, "Missing activs for leaf {}".format(self.name)
        assert (
            self.activs.shape == shape
        ), "Wrong activs shape for leaf {}, {} != {}".format(
            self.name, self.activs.shape, shape
        )
        return self.activs

    def _prereset(self):
        pass

    def _postreset(self):
        pass

    def _reset(self):
        self.activs = None

    def reset(self):
        self._reset()


def create_cppn(genome, config, leaf_names, node_names, output_activation=None):
    """
    Create cppn as described in HyperNEAT (linked above)
    
    :param genome:
    :param config:
    :param leaf_names: names of  input nodes aka  input keys
    :param node_names: names of output nodes aka output keys
    :param output_activation:
    """

    genome_config = config.genome_config

    required = required_for_output(
        genome_config.input_keys, genome_config.output_keys, genome.connections
    )


    # for HyperNEAT, there should be exactly one output node here
    # so it should look like: 
    # node_inputs = {0:[]}

    # Gather inputs and expressed connections. # incoming connections of each node i
    node_inputs = {i: [] for i in genome_config.output_keys}

    for cg in genome.connections.values():
        if not cg.enabled:
            continue

        i, o = cg.key
        if o not in required and i not in required:
            continue

        if i in genome_config.output_keys:
            continue

        if o not in node_inputs:
            node_inputs[o] = [(i, cg.weight)]
        else:
            node_inputs[o].append((i, cg.weight))

        if i not in node_inputs:
            node_inputs[i] = []
    # node inputs now contains for all nodes a list of tuples of incoming node, incoming weight

    # initialize set of all nodes with input nodes ("leaves")
    nodes = {i: Leaf() for i in genome_config.input_keys}

    assert len(leaf_names) == len(genome_config.input_keys)
    leaves = {name: nodes[i] for name, i in zip(leaf_names, genome_config.input_keys)}

    def build_node(idx):
        # Depth first search
        if idx in nodes: # already built node because its a leaf or its been built on different build_node DFS
            return nodes[idx]

        node = genome.nodes[idx]
        conns = node_inputs[idx]
        children = [build_node(i) for i, w in conns]
        weights = [w for i, w in conns]

        if idx in genome_config.output_keys and output_activation is not None:
            activation = output_activation
        else:
            activation = str_to_activation[node.activation]
        aggregation = str_to_aggregation[node.aggregation]
        nodes[idx] = Node(
            children,
            weights,
            node.response,
            node.bias,
            activation,
            aggregation,
            leaves=leaves,
        )
        return nodes[idx]

    for idx in genome_config.output_keys:
        # start DFS tree creation, starting at each output connection 
        build_node(idx)

    outputs = [nodes[i] for i in genome_config.output_keys]

    # assign names to input nodes/input keys/leaves FIXME correct?
    for name in leaf_names:
        leaves[name].name = name

    # assign names to output nodes/output keys FIXME correct?
    for i, name in zip(genome_config.output_keys, node_names):
        nodes[i].name = name

    return outputs

def clamp_weights_(weights, weight_threshold=0.2, weight_max=3.0):
    # TODO: also try LEO
    low_idxs = weights.abs() < weight_threshold
    weights[low_idxs] = 0
    weights[weights > 0] -= weight_threshold
    weights[weights < 0] += weight_threshold
    weights[weights > weight_max] = weight_max
    weights[weights < -weight_max] = -weight_max



def get_coord_inputs(in_coords, out_coords, batch_size=None):
    """
    get coord inputs.

    :param in_coords: 2D input coords: [[0.,1.],[-0.5,0.5]] # for HyperNEAT, all values between -1, 1 (corners of hypercube) 
    :param out_coords:
    :param batch_size:

    """
    n_in = len(in_coords)
    n_out = len(out_coords)

    if batch_size is not None:
        in_coords = in_coords.unsqueeze(0).expand(batch_size, n_in, 2)
        out_coords = out_coords.unsqueeze(0).expand(batch_size, n_out, 2)

        x_out = out_coords[:, :, 0].unsqueeze(2).expand(batch_size, n_out, n_in)
        y_out = out_coords[:, :, 1].unsqueeze(2).expand(batch_size, n_out, n_in)
        x_in = in_coords[:, :, 0].unsqueeze(1).expand(batch_size, n_out, n_in)
        y_in = in_coords[:, :, 1].unsqueeze(1).expand(batch_size, n_out, n_in)
    else:
        x_out = out_coords[:, 0].unsqueeze(1).expand(n_out, n_in)
        y_out = out_coords[:, 1].unsqueeze(1).expand(n_out, n_in)
        x_in = in_coords[:, 0].unsqueeze(0).expand(n_out, n_in)
        y_in = in_coords[:, 1].unsqueeze(0).expand(n_out, n_in)

    return (x_out, y_out), (x_in, y_in)

def get_nd_coord_inputs(in_coords, out_coords, outgoing, batch_size=None):
    """

    :param in_coords: 
    :param out_coords: 
    :param outgoíng: bool, coords outgoing or not? what the fuck does this mean FIXME correct?
    :param batch_size: int, how large batches should be

    """
    in_coords = torch.tensor(
        in_coords, dtype=torch.float32
    )
    out_coords = torch.tensor(
        out_coords, dtype=torch.float32
    )
    n_in = len(in_coords)
    n_out = len(out_coords)
    num_dimens = len(in_coords[0])

    # dict of key: val that look like this for each dim k=1,..,n:
    # if outgoing:
    #    k_in: Tensor of shape  n_out, n_in * n_in
    #    and
    #    k_out: Tensor of shape n_out*n_out , n_in
    # else:
    #    k_in: Tensor of shape  n_in * n_in, n_out
    #    and
    #    k_out: Tensor of shape n_in, n_out * n_out

    # FIXME correct?

    dimen_arrays = {} 
    
    if batch_size is not None:
        # FIXME TODO XXX implement this code block lul
        assert False
        in_coords = in_coords.unsqueeze(0).expand(batch_size, n_in, 2)
        out_coords = out_coords.unsqueeze(0).expand(batch_size, n_out, 2)

        x_out = out_coords[:, :, 0].unsqueeze(2).expand(batch_size, n_out, n_in)
        y_out = out_coords[:, :, 1].unsqueeze(2).expand(batch_size, n_out, n_in)
        x_in = in_coords[:, :, 0].unsqueeze(1).expand(batch_size, n_out, n_in)
        y_in = in_coords[:, :, 1].unsqueeze(1).expand(batch_size, n_out, n_in)

        for x in range(num_dimens):
            # i think the outgoing variant is just the transpose of non outgoing, TODO confirm:
            assert in_coords[:,x].unsqueeze(0).expand(n_out,n_in).transpose(0,1).allclose(
                in_coords[:,x].unsqueeze(1).expand(n_in, n_out)
            ), ":("
            assert out_coords[:,x].unsqueeze(1).expand(n_out,n_in).transpose(0,1).allclose(
                out_coords[:,x].unsqueeze(0).expand(n_in, n_out)
            ), ":("
            if outgoing:
                dimen_arrays[str(x) + "_out"] = out_coords[:, x].unsqueeze(1).expand(n_out, n_in) 
                dimen_arrays[str(x) + "_in"] = in_coords[:, x].unsqueeze(0).expand(n_out, n_in)
            else:
                dimen_arrays[str(x) + "_out"] = out_coords[:, x].unsqueeze(0).expand(n_in, n_out) 
                dimen_arrays[str(x) + "_in"] = in_coords[:, x].unsqueeze(1).expand(n_in, n_out)
    else:
        for x in range(num_dimens):
            # previous implementation was analogous to:
            # outgoing_out = out_coords[:, x].unsqueeze(1).expand(n_out, n_in)
            # outgoing_in  = out_coords[:, x].unsqueeze(0).expand(n_out, n_in)
            #
            # if outgoing:
            #     dimen_arrays[str(x) + "_out"] = outgoing_out # n_out , n_in^2 
            #     dimen_arrays[str(x) + "_in"] = outgoing_in # n_out^2 , n_in
            # else:
            #     dimen_arrays[str(x) + "_out"] = outgoing_out.transpose(0,1) # n_in^2, n_out
            #     dimen_arrays[str(x) + "_in"] = outgoing_in.transpose(0,1) # n_in, n_out^2

            dimen_arrays[str(x) + "_out"] = out_coords[:, x].unsqueeze(1).expand(n_out, n_in) # n_out , n_in^2 
            dimen_arrays[str(x) + "_in"] = in_coords[:, x].unsqueeze(0).expand(n_out, n_in) # n_out^2 , n_in
    return dimen_arrays
