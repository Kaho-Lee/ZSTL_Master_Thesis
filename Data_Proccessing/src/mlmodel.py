from enum import Enum, _reduce_ex_by_name
from functools import reduce, partial

import numpy as np
from typing import Tuple, List

import torch
import torch.nn
import torch.optim

from torch.nn import functional as thf

#from teacher_student.stucture_utils import EnumPBN


def std_initializer(*size, std=0.01):  # TODO implement a better initializer here...
    return std * torch.randn(size)


def zero_initializer(*size):
    return torch.zeros(size)


class FunctionalLayer:

    def __init__(self) -> None:
        super().__init__()

    @property
    def n_inputs(self):
        return 1

    @property
    def n_weights(self):
        return 0

    def _f(self, w):
        raise Exception("Thou should not use me")

    def __call__(self, w: List[torch.Tensor], x: torch.Tensor = None):
        """
        Gives the functional layers numerical parameters (weights) when needed and optionally
        applies it to the input `x`

        :param w: list of parameters (weights) if no parameters then use empty list
        :param x: optional input
        :return: a callable if no input is given, or the output if input is given
        """
        return self._f(w) if x is None else self._f(w)(x)

    def initialize_weights(self, x, output_shape=None, **kwargs) -> List[torch.Tensor]:
        return []


class FLinearLayer(FunctionalLayer):

    def __init__(self, n_outs=None, bias=True) -> None:
        super().__init__()
        self._bias = bias
        self.n_outs = n_outs

    @property
    def n_weights(self):
        return 2 if self._bias else 1

    def _f(self, w):
        return lambda x: thf.linear(x, *w)

    def initialize_weights(self, x, output_shape=None, **kwargs):
        initializer = kwargs.get('initializer', std_initializer)
        out_dim = self.n_outs if self.n_outs is not None else output_shape  # supports dynamic output selection
        assert out_dim is not None, "something wrong here... out_dim is None!"
        w = [initializer(out_dim, *x.shape[1:])]
        if self._bias:
            w.append(initializer(out_dim))
        return w


class FActivation(FunctionalLayer):

    def __init__(self, activation_function) -> None:
        super().__init__()
        self._activation_function = activation_function  # maybe (if needed)
        # use strings for better serialization?

    @property
    def n_weights(self):
        return 0

    def _f(self, w):
        return self._activation_function


class FuncRecursiveNet(FunctionalLayer):

    def __init__(self, layers: List[FunctionalLayer]):
        super().__init__()
        self.layers = layers  # this is a list of layers

        # create a list of start and end indices for the weights in each layer (since one layer may have 1 weight
        # variable and another may have 2 or 0....)
        self.weight_indices = reduce(lambda lst, l: lst + [lst[-1] + l.n_weights], self.layers, [0])

    @property
    def n_weights(self):
        return sum(l.n_weights for l in self.layers)

    def _f(self, w):
        # just the recursive composition of layers!
        return lambda x: reduce(
            lambda z, ls: ls[0](w[ls[1]: ls[2]], z),
            zip(self.layers, self.weight_indices, self.weight_indices[1:]), x)

    def initialize_weights(self, x, output_dim=None, **kwargs):
        # maybe kwargs should support a list of initializers.... will include if useful
        weights = []
        z = x
        for l in self.layers:
            new_weights = l.initialize_weights(z, output_shape=output_dim, **kwargs)
            z = l(new_weights, z)
            weights.extend(new_weights)
        return weights


# class Architecture(EnumPBN):
#     LINEAR_SCALAR_NO_BIAS = FuncRecursiveNet([
#         FLinearLayer(None, False)
#     ])
#     MAML_TEST_SINUSOID = FuncRecursiveNet([
#         FLinearLayer(40),
#         FActivation(torch.relu),
#         FLinearLayer(40),
#         FActivation(torch.relu),
#         FLinearLayer()
#     ])


# TODO put me in the Testing folder
if __name__ == '__main__':
    xx = torch.randn(2, 10)
    print(xx)

    zz = torch.randn(4, 10)

    x = torch.tensor([[1.,2.,3.]])
    print(x)

    net = FuncRecursiveNet([
        FLinearLayer(1, False)
    ])

    print(net)

    init_w = net.initialize_weights(x)
    print(init_w, len(init_w))

    net_f = net(init_w)
    print(net_f)

    temp = net_f(x)
    print(temp)

    x = torch.tensor([[1.,1.,1.]])
    w = [torch.ones_like(p) for p in init_w]
    #net_f = net(w, x)
    print(net(w, x))
    net_f = net(w, x)
    print(net_f)

    init_w = [w.requires_grad_(True) for w in init_w]
    print(init_w)
    loss = torch.pow(net(init_w, x), 2)

    gd = lambda w, loss, eta=0.5: [sub_w - eta*g for sub_w, g in zip(w, \
    torch.autograd.grad(loss, w, create_graph=True) )]

    w_gd_1 = gd(init_w, loss)
    print('w_gd_1 ',w_gd_1)

    loss = torch.pow(net(w_gd_1, x), 2)
    w_gd_2 = gd(w_gd_1, loss)
    print('w_gd_2 ', w_gd_2)

    loss = torch.pow(net(w_gd_2, x), 2)
    hg = torch.autograd.grad(loss, init_w, retain_graph=True)
    print('hg ', hg)
    # net = FuncRecursiveNet([
    #     FLinearLayer(4, True),
    #     FActivation(thf.relu),
    #     FLinearLayer(5, False),
    #     FActivation(torch.tanh),
    #     # FLinearLayer(3)
    # ])

    # the_weights = net.initialize_weights(xx)

    # print(len(the_weights))

    # for _w in the_weights:
    #     print(_w.shape)

    # net_f = net(the_weights)
    # print(net_f)

    # print(net(the_weights, xx))

    # print(net_f(zz))
