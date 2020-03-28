import torch


class SafeMutate(torch.Function):

    @staticmethod
    def forward(ctx, cppn, xs, shape):
    """
    xs: list of torch tensors
    """
    if not xs:
        return torch.full(shape, cppn.bias)
    inputs = [w * x for w, x in zip(cppn.weights, xs)]
    try:
        pre_activs = cppn.aggregation(inputs)
        activs = cppn.activation(cppn.response * pre_activs + cppn.bias)
        assert activs.shape == shape, "Wrong shape for node {}".format(cppn.name)
    except Exception:
        raise Exception("Failed to activate node {}".format(cppn.name))
    return activs
    
    @staticmethod
    def backward(ctx, grads):
        result, = ctx.save_tensors
        return grads * result