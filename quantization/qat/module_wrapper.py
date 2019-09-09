import os
import torch.nn as nn
import torch
from quantization.methods.uniform import UniformQuantization
from quantization.methods.non_uniform import KmeansQuantization
from quantization.methods.non_uniform import LearnableDifferentiableQuantization, KmeansQuantization, LearnedCentroidsQuantization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_binning_hist(x, y, min, max):
    fig, ax = plt.subplots()
    x = np.array([float("{:.3f}".format(d)) for d in x])
    ax.errorbar(x, y, xerr=[x - min, max - x], fmt='o', ecolor='r', capsize=10, capthick=2)
    plt.bar(min, y, width=max - min, edgecolor='k', align='edge', alpha=0.5)
    ax.legend([str(x[x.argsort()]), str(y[x.argsort()])], fontsize=12, loc='upper center',
              bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2, fancybox=True)
    plt.yticks(size=12)
    plt.xlabel('binning', size=16)
    plt.ylabel('values in bin', size=16)

    return fig


def plot_centroids(centroids):
    fig, ax = plt.subplots()
    for i in range(centroids.shape[1]):
        plt.plot(np.arange(centroids.shape[0]), centroids[:, i])

    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('epoch', size=16)
    plt.ylabel('centroids', size=16)
    return fig


def plot_tensor_binning(tensor, B, v, name, step, ml_logger):
    x = v.cpu().numpy().flatten()
    y = torch.sum(B, dim=0).cpu().numpy()

    max = np.empty(B.shape[1])
    min = np.empty(B.shape[1])
    for i in range(B.shape[1]):
        b = tensor[B[:, i].type(torch.bool)]
        max[i] = b.max().cpu().item()
        min[i] = b.min().cpu().item()

    fig = plot_binning_hist(x, y, min, max)
    ml_logger.tf_logger.add_figure(name + '.binning', fig, step)


class ParameterModuleWrapper(nn.Module):
    def __init__(self, name, wrapped_module, quantization_scheduler, **kwargs):
        super(ParameterModuleWrapper, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.quantization_scheduler = quantization_scheduler
        self.forward_functor = kwargs['forward_functor']
        self.bit_weights = kwargs['bits_weight']
        self.bits_out = kwargs['bits_out']
        self.temperature = kwargs['temperature']
        self.enabled = True
        self.active = True
        self.centroids_hist = {}
        self.log_weight_hist = False
        self.log_mse = False
        self.log_clustering = False
        self.bn = kwargs['bn'] if 'bn' in kwargs else None

        setattr(self, 'weight', wrapped_module.weight)
        setattr(self, 'bias', wrapped_module.bias)
        delattr(wrapped_module, 'weight')
        delattr(wrapped_module, 'bias')

        if self.bit_weights is not None:
            self.weight_quantization_default = LearnedCentroidsQuantization(self,
                                                                            self.bit_weights,
                                                                            symmetric=True,
                                                                            tensor=self.weight,
                                                                            temperature=self.temperature)
            self.weight_quantization = self.weight_quantization_default
            if self.quantization_scheduler is not None:
                self.quantization_scheduler.add_quantization_params(self.weight_quantization.optim_parameters())

        if self.bit_weights is not None:
            self.inner_quantization = UniformQuantization(self, self.weight, 8, symmetric=True)
            if self.quantization_scheduler is not None:
                self.quantization_scheduler.add_quantization_params(self.inner_quantization.optim_parameters())

        if self.quantization_scheduler is not None:
            self.quantization_scheduler.register_module_quantization(self)

    def __enabled__(self):
        return self.enabled and self.active and self.bit_weights is not None

    def forward(self, *input):
        w = self.weight
        if self.__enabled__():
            # Quantize weights
            w = self.weight_quantization(w)
            # w = self.inner_quantization(w) # TODO: meanwhile disable 8 bit quantization

        out = self.forward_functor(*input, weight=w, bias=self.bias)

        return out

    def set_quant_method(self, method=None):
        if self.bit_weights is not None:
            if method is None:
                self.weight_quantization = self.weight_quantization_default
            elif method == 'kmeans':
                self.weight_quantization = KmeansQuantization(self.bit_weights)
            else:
                self.weight_quantization = self.weight_quantization_default

    # TODO: make it more generic
    def set_quant_mode(self, mode=None):
        if self.bit_weights is not None:
            if mode is not None:
                self.soft = self.weight_quantization.soft_quant
                self.hard = self.weight_quantization.hard_quant
            if mode is None:
                self.weight_quantization.soft_quant = self.soft
                self.weight_quantization.hard_quant = self.hard
            elif mode == 'soft':
                self.weight_quantization.soft_quant = True
                self.weight_quantization.hard_quant = False
            elif mode == 'hard':
                self.weight_quantization.soft_quant = False
                self.weight_quantization.hard_quant = True

    def log_state(self, step, ml_logger):
        if self.__enabled__():
            # TODO: make more generic
            if hasattr(self, 'alpha'):
                ml_logger.log_metric(self.name + '.alpha', self.alpha.item(),  step='auto')

            if self.weight_quantization is not None:
                for n, p in self.weight_quantization.loggable_parameters():
                    if p.numel() == 1:
                        ml_logger.log_metric(self.name + '.' + n, p.item(),  step='auto')
                    else:
                        for i, e in enumerate(p):
                            ml_logger.log_metric(self.name + '.' + n + '.' + str(i), e.item(),  step='auto')

            # plot weights binning
            if self.log_clustering:
                weight = self.weight.flatten()
                B, v = self.weight_quantization.clustering(weight)
                plot_tensor_binning(weight, B, v, self.name, step, ml_logger)

            if self.log_weight_hist:
                ml_logger.tf_logger.add_histogram(self.name + '.weight', self.weight.cpu().flatten(),  step='auto')

            if self.log_mse:
                weight_q = self.weight_quantization(self.weight.flatten())
                mse_q = torch.nn.MSELoss()(self.weight.flatten(), weight_q)
                ml_logger.log_metric(self.name + '.mse_q', mse_q.cpu().item(),  step='auto')

                weight_kmeans = KmeansQuantization(self.bit_weights)(self.weight.flatten())
                mse_kmeans = torch.nn.MSELoss()(self.weight.flatten(), weight_kmeans)
                ml_logger.log_metric(self.name + '.mse_kmeans', mse_kmeans.cpu().item(),  step='auto')


class ActivationModuleWrapper(nn.Module):
    def __init__(self, name, wrapped_module, quantization_scheduler, **kwargs):
        super(ActivationModuleWrapper, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.quantization_scheduler = quantization_scheduler
        self.bits_out_outer = kwargs['bits_out']
        self.bits_out_inner = 4
        self.enabled = True
        self.active = True
        self.temperature = kwargs['temperature'] # TODO: pass it directly to Quantization as kwargs

        if self.bits_out_outer is not None:
            # self.out_quantization_inner_default = LearnedStepQuantization(self, self.bits_out_inner, symmetric=False)
            # self.out_quantization_inner = self.out_quantization_inner_default

            self.out_quantization_outer = self.out_quantization_outer_default = None

            def __init_out_quantization_outer__(tensor):
                self.out_quantization_outer_default = LearnedCentroidsQuantization(self,
                                                                                   self.bits_out_outer,
                                                                                   symmetric=False,
                                                                                   tensor=tensor,
                                                                                   temperature=self.temperature)
                self.out_quantization_outer = self.out_quantization_outer_default

                self.quantization_scheduler.add_quantization_params(self.out_quantization_outer.optim_parameters())

            self.out_quantization_outer_init_fn = __init_out_quantization_outer__

            self.quantization_scheduler.register_module_quantization(self)

    def __enabled__(self):
        return self.enabled and self.active and self.bits_out_outer is not None

    def forward(self, *input):
        out = self.wrapped_module(*input)

        # Quantize output
        if self.__enabled__():
            # out, delta = self.out_quantization_inner.quantize(out, dequantize=False)

            self.verify_initialized(self.out_quantization_outer, out, self.out_quantization_outer_init_fn)

            # dequantize
            # out = out * delta

            out = self.out_quantization_outer.quantize(out)

            if isinstance(self.wrapped_module, torch.nn.ReLU):
                out = torch.nn.functional.relu(out)

        return out

    def set_quant_method(self, method=None):
        if self.bits_out_outer is not None:
            if method == 'kmeans':
                self.out_quantization_outer = KmeansQuantization(self.bits_out_outer, max_iter=3)
            else:
                self.out_quantization_outer = self.out_quantization_outer_default

    def verify_initialized(self, quantization_handle, tensor, init_fn):
        if quantization_handle is None:
            init_fn(tensor)

    def log_state(self, step, ml_logger):
        if self.__enabled__():
            # TODO: make more generic
            if hasattr(self, 'lsq_alpha'):
                ml_logger.log_metric(self.name + '.lsq_alpha', self.alpha.item(),  step='auto')

            if self.out_quantization_outer is not None:
                for n, p in self.out_quantization_outer.named_parameters():
                    if p.numel() == 1:
                        ml_logger.log_metric(self.name + '.' + n, p.item(),  step='auto')
                    else:
                        for i, e in enumerate(p):
                            ml_logger.log_metric(self.name + '.' + n + '.' + str(i), e.item(),  step='auto')
