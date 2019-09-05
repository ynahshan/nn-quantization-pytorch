import torch
import torch.nn as nn


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class QuantizationBase(object):
    def __init__(self, module, num_bits, symmetric=True):
        self.module = module
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.num_bins = int(2 ** num_bits)
        self.opt_params = {}
        self.named_params = []

    def register_buffer(self, name, value):
        if hasattr(self.module, name):
            delattr(self.module, name)
        self.module.register_buffer(name, value)
        setattr(self, name, getattr(self.module, name))

    def register_parameter(self, name, value):
        if hasattr(self.module, name):
            delattr(self.module, name)
        self.module.register_parameter(name, nn.Parameter(value))
        setattr(self, name, getattr(self.module, name))

        self.named_params.append((name, getattr(self.module, name)))

    def __add_optim_params__(self, optim_type, dataset, params):
        learnable_params = [d for n, d in params if n in self.learned_parameters()]
        self.opt_params[optim_type + '_' + dataset] = learnable_params

    def optim_parameters(self):
        return self.opt_params

    def loggable_parameters(self):
        return self.named_parameters()

    def named_parameters(self):
        named_params = [(n, p) for n, p in self.named_params if n in self.learned_parameters()]
        return named_params

    @staticmethod
    def learned_parameters():
        return []


class MinMaxQuantization(QuantizationBase):
    def __init__(self, module, num_bits, symmetric, stochastic=False):
        super(MinMaxQuantization, self).__init__(module, num_bits, symmetric)
        self.stochastic = stochastic
        if symmetric:
            self.qmax = 2 ** (self.num_bits - 1) - 1
            self.qmin = -self.qmax - 1
        else:
            self.qmax = 2 ** self.num_bits - 1
            self.qmin = 0

    def __quantize__(self, tensor, alpha):
        delta = (2 * alpha) / self.num_bins

        # quantize
        t_q = tensor / delta

        # stochastic rounding
        if self.stochastic and self.module.training:
            with torch.no_grad():
                noise = t_q.new_empty(t_q.shape).uniform_(-0.5, 0.5)
                t_q += noise

        # clamp and round
        t_q = torch.clamp(t_q, self.qmin, self.qmax)
        t_q = RoundSTE.apply(t_q)

        # de-quantize
        t_q = t_q * delta
        return t_q

    def __call__(self, tensor):
        alpha = tensor.abs().max()
        t_q = self.__quantize__(tensor, alpha)
        return t_q
