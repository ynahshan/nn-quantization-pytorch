import torch
import torch.nn as nn
from .uniform import UniformQuantization
import scipy.optimize as opt


class ClippedUniformQuantization(UniformQuantization):
    alpha_param_name = 'alpha'

    def __init__(self, module, num_bits, symmetric, stochastic=False):
        super(ClippedUniformQuantization, self).__init__(module, num_bits, symmetric, stochastic)

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q

    def __for_repr__(self):
        rpr = super(ClippedUniformQuantization, self).__for_repr__()
        return [(self.alpha_param_name, '{:.4f}'.format(getattr(self, self.alpha_param_name).item()))] + rpr


class LearnedStepSizeQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(LearnedStepSizeQuantization, self).__init__(module, num_bits, symmetric, stochastic)

        with torch.no_grad():
            maxabs = tensor.abs().max()

        self.register_parameter(self.alpha_param_name, tensor.new_tensor([maxabs]))

        self.__create_optim_params__()

    def __create_optim_params__(self):
        # TODO: create default configuration
        self.__add_optim_params__('SGD', 'imagenet', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('SGD', 'cifar10', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])

    @staticmethod
    def learned_parameters():
        return [
                LearnedStepSizeQuantization.alpha_param_name
                ]


class MaxAbsStaticQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(MaxAbsStaticQuantization, self).__init__(module, num_bits, symmetric)

        with torch.no_grad():
            self.register_buffer(self.alpha_param_name, tensor.new_tensor([tensor.abs().max()]))


class MseDirectQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(MseDirectQuantization, self).__init__(module, num_bits, symmetric, stochastic)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)
        err = torch.sum((xq - x) ** 2) / N
        return err.item()


class MseDecomposedQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(MseDecomposedQuantization, self).__init__(module, num_bits, symmetric, stochastic)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        if alpha > 0:
            xclamp = torch.clamp(x, -alpha, alpha)
            clip_err = torch.sum((xclamp - x) ** 2) / N

            xq = self.__quantize__(xclamp, alpha)
            quant_err = torch.sum((xq - x) ** 2) / N
            err = clip_err + quant_err
        else:
            err = torch.sum(x**2) / N
        return err.item()


class MseQuantEstimatedQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(MseQuantEstimatedQuantization, self).__init__(module, num_bits, symmetric, stochastic)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor), bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        clip_err = torch.sum((torch.clamp(x, -alpha, alpha) - x) ** 2) / N
        quant_err = alpha ** 2 / ((3 if self.symmetric else 12) * (2 ** (2 * self.num_bits)))
        err = clip_err + quant_err
        return err.item()


class AciqGausQuantization(ClippedUniformQuantization):
    gaus_mult = {1: 1.24, 2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
    gaus_mult_positive = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92, 8: 4.2}

    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(AciqGausQuantization, self).__init__(module, num_bits, symmetric, stochastic)

        with torch.no_grad():
            if self.symmetric:
                sigma = tensor.std()
                alpha_opt = self.gaus_mult[self.num_bits] * sigma
            else:
                # We need to measure std before ReLu.
                # Instead assume zero mean and multiply std after relu by 2 to approximation std before relu.
                sigma = 2 * torch.sqrt(torch.mean(tensor[tensor != 0]**2))
                alpha_opt = self.gaus_mult_positive[self.num_bits] * sigma

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([alpha_opt]))


class AciqLaplaceQuantization(ClippedUniformQuantization):
    laplace_mult = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
    laplace_mult_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}

    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(AciqLaplaceQuantization, self).__init__(module, num_bits, symmetric, stochastic)

        with torch.no_grad():
            if symmetric:
                b = tensor.abs().mean()
                alpha = self.laplace_mult[self.num_bits] * b
            else:
                # We need to measure b before ReLu.
                # Instead assume zero mean and multiply b after relu by 2 to approximation b before relu.
                b = 2 * tensor[tensor != 0].abs().mean()
                alpha = self.laplace_mult_positive[self.num_bits] * b

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([alpha]))
