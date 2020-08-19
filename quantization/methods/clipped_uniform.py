import numpy as np
import scipy.optimize as opt
import torch

from .uniform import UniformQuantization


class ClippedUniformQuantization(UniformQuantization):
    alpha_param_name = 'alpha'

    def __init__(self, module, num_bits, symmetric, uint=False, stochastic=False, tails=False):
        super(ClippedUniformQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic,tails)

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q

    def __for_repr__(self):
        rpr = super(ClippedUniformQuantization, self).__for_repr__()
        return [(self.alpha_param_name, '{:.4f}'.format(getattr(self, self.alpha_param_name).item()))] + rpr


class FixedClipValueQuantization(ClippedUniformQuantization):
    def __init__(self, module, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(FixedClipValueQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)
        self.clip_value = kwargs['clip_value']
        self.device = kwargs['device']
        with torch.no_grad():
            self.register_buffer(self.alpha_param_name, torch.tensor([self.clip_value], dtype=torch.float32).to(self.device))


class MaxAbsStaticQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(MaxAbsStaticQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            self.register_buffer(self.alpha_param_name, tensor.new_tensor([tensor.abs().max()]))


class LearnedStepSizeQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, **kwargs):
        super(LearnedStepSizeQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            maxabs = tensor.abs().max()

        self.register_parameter(self.alpha_param_name, tensor.new_tensor([maxabs]))

        self.__create_optim_params__()

    def __create_optim_params__(self):
        # TODO: create default configuration
        self.__add_optim_params__('SGD', 'imagenet', [
            (self.alpha_param_name, {'params': [getattr(self, self.alpha_param_name)], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('SGD', 'cifar10', [
            (self.alpha_param_name, {'params': [getattr(self, self.alpha_param_name)], 'lr': 1e-1, 'momentum': 0, 'weight_decay': 0})
        ])

    @staticmethod
    def learned_parameters():
        return [
                LearnedStepSizeQuantization.alpha_param_name
                ]


class AngDistanceQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(AngDistanceQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        xq = self.__quantize__(x, alpha)

        norm_x = torch.norm(x)
        norm_xq = torch.norm(xq)
        cos = torch.dot(x.flatten(), xq.flatten()) / (norm_x * norm_xq)
        err = torch.acos(cos)
        return err.item()


class LpNormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(LpNormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        self.p = kwargs['lp']
        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        xq = self.__quantize__(x, alpha)
        err = torch.mean(torch.abs(xq - x) ** self.p)
        return err.item()


class L1NormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(L1NormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)
        err = torch.sum(torch.abs(xq - x)) / N
        return err.item()


class L2NormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(L2NormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)
        err = torch.sum(torch.abs(xq - x) ** 2) / N
        return err.item()


class L3NormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(L3NormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)
        err = torch.sum(torch.abs(xq - x) ** 3) / N
        return err.item()


class MseNoPriorQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(MseNoPriorQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        delta = (2 if self.symmetric else 1) * alpha / (self.num_bins - 1)
        if self.tails:
            Cx = torch.clamp(x,(-alpha if self.symmetric else 0.) - delta / 2, alpha + delta / 2)
        else:
            Cx = torch.clamp(x, -alpha if self.symmetric else 0., alpha)
        Ci = Cx - x

        N = x.numel() if self.symmetric else x[x != 0].numel()
        xq = self.__quantize__(x, alpha)

        qerr_exp = torch.sum((xq - Cx)) / N
        qerrsq_exp = torch.sum((xq - Cx) ** 2) / N
        cerr = torch.sum(Ci ** 2) / N
        mixed_err = 2 * torch.sum(Ci) * alpha * qerr_exp / N
        mse = qerrsq_exp + cerr + mixed_err
        return mse.item()


class LogLikeQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(LogLikeQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        with torch.no_grad():
            if symmetric:
                self.b = tensor.abs().mean()
            else:
                # We need to measure b before ReLu.
                # Instead assume zero mean and multiply b after relu by 2 to approximation b before relu.
                self.b = tensor[tensor != 0].abs().mean()

        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor)).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        delta = (2 if self.symmetric else 1) * alpha / (self.num_bins - 1)
        Nq = x[(x > 0) & (x <= alpha)].numel()
        Nc = x[x > alpha].numel()
        clip_err = ((x[x > alpha]- alpha) / self.b).sum() + Nc * torch.log(torch.clamp(self.b, 1e-30, 1e+30))
        q_err = Nq * np.log(np.max([delta, 1e-100]))
        # print("alpha={}, delta={}, q={}, c={}, tot={}".format(alpha,delta,q_err, clip_err.item(),clip_err.item() + q_err + add))
        return clip_err.item() + q_err


class MseUniformPriorQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(MseUniformPriorQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

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

    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(AciqGausQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            if self.symmetric:
                sigma = tensor.std()
                alpha_opt = self.gaus_mult[self.num_bits] * sigma
            else:
                # We need to measure std before ReLu.
                # Instead assume zero mean and multiply std after relu by 2 to approximation std before relu.
                sigma = torch.sqrt(torch.mean(tensor[tensor != 0]**2))
                alpha_opt = self.gaus_mult_positive[self.num_bits] * sigma

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([alpha_opt]))


class AciqLaplaceQuantization(ClippedUniformQuantization):
    laplace_mult = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
    laplace_mult_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}

    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(AciqLaplaceQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            if symmetric:
                b = tensor.abs().mean()
                alpha = self.laplace_mult[self.num_bits] * b
            else:
                # We need to measure b before ReLu.
                # Instead assume zero mean and multiply b after relu by 2 to approximation b before relu.
                b = tensor[tensor != 0].abs().mean()
                alpha = self.laplace_mult_positive[self.num_bits] * b

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([alpha]))
