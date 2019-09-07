import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from quantization.methods.uniform import UniformQuantization
from quantization.methods.clipped_uniform import LearnedStepSizeQuantization
from quantization.methods.non_uniform import LearnableDifferentiableQuantization, KmeansQuantization, LearnedCentroidsQuantization
from utils.absorb_bn import is_absorbing, is_bn


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


class Conv2dFunctor:
    def __init__(self, conv2d):
        self.conv2d = conv2d

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.conv2d(*input, weight, bias, self.conv2d.stride, self.conv2d.padding,
                                         self.conv2d.dilation, self.conv2d.groups)
        return res


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


class QuantizationScheduler(object):
    _iter_counter = count(0)

    def __init__(self, model, optimizer, grad_rate, enable=True):
        self.quantizations = []
        self.optimizer = optimizer
        self.grad_rate = grad_rate
        self.scheduling_enabled = enable

        model.register_forward_hook(lambda m, inp, out: self.step(m))

    def register_module_quantization(self, qwrapper):
        self.quantizations.append(qwrapper)
        if len(self.quantizations) == 1 or not self.scheduling_enabled:
            qwrapper.enabled = True
        else:
            qwrapper.enabled = False

    def step(self, model):
        if model.training:
            step = next(QuantizationScheduler._iter_counter)
        if model.training and step > 0:
            if step % 1001 == 0:
                for q in self.quantizations:
                    q.rho.data = 0.934 * q.rho
                    q.temperature.data = q.temperature + 75
                # print("Updated rho {}".format(self.quantizations[0].rho.item()))

        if self.scheduling_enabled and model.training:
            if step % self.grad_rate == 0:
                i = int(step / self.grad_rate)
                if i < len(self.quantizations):
                    self.quantizations[i].enabled = True

    def add_quantization_params(self, all_quant_params):
        opt = 'SGD'
        dataset = 'imagenet'
        key = opt + '_' + dataset
        if key in all_quant_params:
            quant_params = all_quant_params[key]
            for group in quant_params:
                self.optimizer.add_param_group(group)


class ModelQuantizer:
    def __init__(self, model, args, quantizable_layers, replacement_factory, quantization_scheduler=None):
        self.model = model
        self.args = args
        self.bit_weights = args.bit_weights
        self.bit_act = args.bit_act
        self.functor_map = {nn.Conv2d: Conv2dFunctor}
        self.replacement_factory = replacement_factory

        self.quantization_scheduler = quantization_scheduler

        self.quantization_wrappers = []
        self.quantizable_modules = []
        self.quantizable_layers = quantizable_layers
        self._pre_process_container(model)
        self._create_quantization_wrappers()

        # TODO: fix freeze problem
        self.quantization_params = LearnedStepSizeQuantization.learned_parameters() + \
                                   LearnedCentroidsQuantization.learned_parameters()

    def freeze(self):
        for n, p in self.model.named_parameters():
            if not np.any([qp in n for qp in self.quantization_params]):
                p.requires_grad = False

        for n, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0

    @staticmethod
    def has_children(module):
        try:
            next(module.children())
            return True
        except StopIteration:
            return False

    def _create_quantization_wrappers(self):
        for qm in self.quantizable_modules:
            # replace module by it's wrapper
            fn = self.functor_map[type(qm.module)](qm.module) if type(qm.module) in self.functor_map else None
            args = {"bits_out": self.bit_act, "bits_weight": self.bit_weights, "forward_functor": fn}
            args.update(vars(self.args))
            if hasattr(qm, 'bn'):
                args['bn'] = qm.bn
            module_wrapper = self.replacement_factory[type(qm.module)](qm.full_name, qm.module, self.quantization_scheduler,
                                                                    **args)
            setattr(qm.container, qm.name, module_wrapper)
            self.quantization_wrappers.append((qm.full_name, module_wrapper))

    def _pre_process_container(self, container, prefix=''):
        prev, prev_name = None, None
        for name, module in container.named_children():
            if is_bn(module) and is_absorbing(prev) and prev_name in self.quantizable_layers:
                # Pass BN module to prev module quantization wrapper for BN folding/unfolding
                self.quantizable_modules[-1].bn = module

            full_name = prefix + name
            if full_name in self.quantizable_layers:
                self.quantizable_modules.append(
                    type('', (object,), {'name': name, 'full_name': full_name, 'module': module, 'container': container})()
                )

            if self.has_children(module):
                # For container we call recursively
                self._pre_process_container(module, full_name + '.')

            prev = module
            prev_name = full_name

    def quantizer_parameters(self):
        weight_binning_params = [param for name, param in self.model.named_parameters()
                                 if LearnableDifferentiableQuantization.binning_param_name in name]
        return weight_binning_params

    def log_weights_quantizer_state(self, ml_logger, step, name, qwrapper):
        qwrapper.log_state(step, ml_logger)

    def log_act_quantizer_state(self, ml_logger, step, name, qwrapper):
        qwrapper.log_state(step, ml_logger)

    def log_quantizer_state(self, ml_logger, step):
        with torch.no_grad():
            for name, qwrapper in self.quantization_wrappers:
                if self.bit_weights is not None:
                    self.log_weights_quantizer_state(ml_logger, step, name, qwrapper)
                if self.bit_act is not None:
                    self.log_act_quantizer_state(ml_logger, step, name, qwrapper)

    class QuantMethod:
        def __init__(self, quantization_wrappers, method):
            self.quantization_wrappers = quantization_wrappers
            self.method = method

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_method(self.method)

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_method()

    class QuantMode:
        def __init__(self, quantization_wrappers, mode):
            self.quantization_wrappers = quantization_wrappers
            self.mode = mode

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_mode(self.mode)

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_mode()

    class DisableQuantizer:
        def __init__(self, quantization_wrappers):
            self.quantization_wrappers = quantization_wrappers

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.active = False

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.active = True

    def quantization_method(self, method):
        return ModelQuantizer.QuantMethod(self.quantization_wrappers, method)

    def quantization_mode(self, mode):
        return ModelQuantizer.QuantMode(self.quantization_wrappers, mode)

    def disable(self):
        return ModelQuantizer.DisableQuantizer(self.quantization_wrappers)
