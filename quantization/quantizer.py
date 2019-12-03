import torch
import torch.nn as nn
import numpy as np
from itertools import count
from quantization.methods.clipped_uniform import LearnedStepSizeQuantization
from quantization.methods.non_uniform import LearnableDifferentiableQuantization, LearnedCentroidsQuantization
from quantization.methods.clipped_uniform import FixedClipValueQuantization
# from utils.absorb_bn import is_absorbing, is_bn


class Conv2dFunctor:
    def __init__(self, conv2d):
        self.conv2d = conv2d

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.conv2d(*input, weight, bias, self.conv2d.stride, self.conv2d.padding,
                                         self.conv2d.dilation, self.conv2d.groups)
        return res


class LinearFunctor:
    def __init__(self, linear):
        self.linear = linear

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.linear(*input, weight, bias)
        return res


class EmbeddingFunctor:
    def __init__(self, embedding):
        self.embedding = embedding

    def __call__(self, *input, weight, bias=None):
        res = torch.nn.functional.embedding(
            *input, weight, self.embedding.padding_idx, self.embedding.max_norm,
            self.embedding.norm_type, self.embedding.scale_grad_by_freq, self.embedding.sparse)
        return res


# class QuantizationScheduler(object):
#     _iter_counter = count(0)
#
#     def __init__(self, model, optimizer, grad_rate, enable=True):
#         self.quantizations = []
#         self.optimizer = optimizer
#         self.grad_rate = grad_rate
#         self.scheduling_enabled = enable
#
#         model.register_forward_hook(lambda m, inp, out: self.step(m))
#
#     def register_module_quantization(self, qwrapper):
#         self.quantizations.append(qwrapper)
#         if len(self.quantizations) == 1 or not self.scheduling_enabled:
#             qwrapper.enabled = True
#         else:
#             qwrapper.enabled = False
#
#     def step(self, model):
#         if model.training:
#             step = next(QuantizationScheduler._iter_counter)
#
#         if self.scheduling_enabled and model.training:
#             if step % self.grad_rate == 0:
#                 i = int(step / self.grad_rate)
#                 if i < len(self.quantizations):
#                     self.quantizations[i].enabled = True


class OptimizerBridge(object):
    def __init__(self, optimizer, settings={'algo': 'SGD', 'dataset': 'imagenet'}):
        self.optimizer = optimizer
        self.settings = settings

    def add_quantization_params(self, all_quant_params):
        key = self.settings['algo'] + '_' + self.settings['dataset']
        if key in all_quant_params:
            quant_params = all_quant_params[key]
            for group in quant_params:
                self.optimizer.add_param_group(group)


class ModelQuantizer:
    def __init__(self, model, args, quantizable_layers, replacement_factory, optimizer_bridge=None):
        self.model = model
        self.args = args
        self.bit_weights = args.bit_weights
        self.bit_act = args.bit_act
        self.post_relu = True
        self.functor_map = {nn.Conv2d: Conv2dFunctor, nn.Linear: LinearFunctor, nn.Embedding: EmbeddingFunctor}
        self.replacement_factory = replacement_factory

        self.optimizer_bridge = optimizer_bridge

        self.quantization_wrappers = []
        self.quantizable_modules = []
        self.quantizable_layers = quantizable_layers
        self._pre_process_container(model)
        self._create_quantization_wrappers()

        # TODO: hack, make it generic
        self.quantization_params = LearnedStepSizeQuantization.learned_parameters()

    def load_state_dict(self, state_dict):
        for name, qwrapper in self.quantization_wrappers:
            qwrapper.load_state_dict(state_dict)

    def freeze(self):
        for n, p in self.model.named_parameters():
            # TODO: hack, make it more robust
            if not np.any([qp in n for qp in self.quantization_params]):
                p.requires_grad = False

        # for n, p in self.model.named_parameters():
        #     if not ('conv' in n or 'downsample.0' in n):
        #         p.requires_grad = False

        # for n, p in self.model.named_parameters():
        #     if not ('bn' in n or 'downsample.1' in n):
        #         p.requires_grad = False

        # for n, m in self.model.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.momentum = 0

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
            args = {"bits_out": self.bit_act, "bits_weight": self.bit_weights, "forward_functor": fn,
                    "post_relu": self.post_relu, "optim_bridge": self.optimizer_bridge}
            args.update(vars(self.args))
            if hasattr(qm, 'bn'):
                args['bn'] = qm.bn
            module_wrapper = self.replacement_factory[type(qm.module)](qm.full_name, qm.module,
                                                                    **args)
            setattr(qm.container, qm.name, module_wrapper)
            self.quantization_wrappers.append((qm.full_name, module_wrapper))

    def _pre_process_container(self, container, prefix=''):
        prev, prev_name = None, None
        for name, module in container.named_children():
            # if is_bn(module) and is_absorbing(prev) and prev_name in self.quantizable_layers:
            #     # Pass BN module to prev module quantization wrapper for BN folding/unfolding
            #     self.quantizable_modules[-1].bn = module

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

    def log_quantizer_state(self, ml_logger, step):
        if self.bit_weights is not None or self.bit_act is not None:
            with torch.no_grad():
                for name, qwrapper in self.quantization_wrappers:
                    qwrapper.log_state(step, ml_logger)

    def get_qwrappers(self):
        return [qwrapper for (name, qwrapper) in self.quantization_wrappers if qwrapper.__enabled__()]

    def set_clipping(self, clipping, device):  # TODO: handle device internally somehow
        qwrappers = self.get_qwrappers()
        for i, qwrapper in enumerate(qwrappers):
            qwrapper.set_quantization(FixedClipValueQuantization,
                                      {'clip_value': clipping[i], 'device': device})

    def get_clipping(self):
        clipping = []
        qwrappers = self.get_qwrappers()
        for i, qwrapper in enumerate(qwrappers):
            q = qwrapper.get_quantization()
            clip_value = getattr(q, 'alpha')
            clipping.append(clip_value.item())

        return qwrappers[0].get_quantization().alpha.new_tensor(clipping)

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
