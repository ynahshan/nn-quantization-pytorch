import torch
import torch.nn as nn
from .uniform import MinMaxQuantization
import scipy.optimize as opt


class LearnedStepSizeQuantization(MinMaxQuantization):
    alpha_param_name = 'alpha'

    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(LearnedStepSizeQuantization, self).__init__(module, num_bits, symmetric, stochastic)

        with torch.no_grad():
            maxabs = tensor.abs().max()

        self.register_parameter(self.alpha_param_name, tensor.new_tensor([maxabs]))

        self.__create_optim_params__()

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q

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


class MinimizedMseQuantization(MinMaxQuantization):
    alpha_param_name = 'alpha'

    def __init__(self, module, tensor, num_bits, symmetric, stochastic=False):
        super(MinimizedMseQuantization, self).__init__(module, num_bits, symmetric, stochastic)

        with torch.no_grad():
            alpha_opt = opt.minimize_scalar(lambda alpha: self.mse_direct(alpha, tensor), bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([alpha_opt]))

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q

    def mse_direct(self, alpha, x):
        if alpha > 0:
            xq = self.__quantize__(x, alpha)
            mse = torch.nn.MSELoss()(x, xq)
        else:
            mse = torch.norm(x) ** 2 / x.numel()
        return mse.cpu().numpy()

    def mse_clip_qest_relu(self, alpha, x):
        N = x[x > 0].numel()
        clip_err = torch.sum((torch.clamp(x, -alpha, alpha) - x) ** 2) / N
        quant_err = alpha ** 2 / (12 * (2 ** (2 * self.num_bits)))
        err = clip_err + quant_err
        return err.cpu().numpy()
