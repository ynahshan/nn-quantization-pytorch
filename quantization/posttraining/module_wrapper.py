import os
import torch.nn as nn
import torch
import torch.nn as nn

from quantization.methods.clipped_uniform import AngDistanceQuantization, L3NormQuantization, L2NormQuantization, \
    LpNormQuantization, L1NormQuantization
from quantization.methods.clipped_uniform import MaxAbsStaticQuantization, AciqLaplaceQuantization, \
    AciqGausQuantization, LogLikeQuantization
from quantization.methods.clipped_uniform import MseNoPriorQuantization, MseUniformPriorQuantization
from quantization.methods.non_uniform import KmeansQuantization

quantization_mapping = {'max_static': MaxAbsStaticQuantization,
                        'aciq_laplace': AciqLaplaceQuantization,
                        'aciq_gaus': AciqGausQuantization,
                        'mse_uniform_prior': MseUniformPriorQuantization,
                        'mse_no_prior': MseNoPriorQuantization,
                        'ang_dis': AngDistanceQuantization,
                        'l3_norm': L3NormQuantization,
                        'l2_norm': L2NormQuantization,
                        'l1_norm': L1NormQuantization,
                        'lp_norm': LpNormQuantization,
                        'log_like': LogLikeQuantization
                        }


def is_positive(module):
    return isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6)


class ActivationModuleWrapperPost(nn.Module):
    def __init__(self, name, wrapped_module, **kwargs):
        super(ActivationModuleWrapperPost, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.post_relu = True
        self.enabled = True
        self.active = True

        if self.bits_out is not None:
            self.out_quantization = self.out_quantization_default = None

            def __init_out_quantization__(tensor):
                self.out_quantization_default = quantization_mapping[self.qtype](self, tensor, self.bits_out,
                                                                                 symmetric=(not is_positive(wrapped_module)),
                                                                                 uint=True, kwargs=kwargs)
                self.out_quantization = self.out_quantization_default
                print("ActivationModuleWrapperPost - {} | {} | {}".format(self.name, str(self.out_quantization), str(tensor.device)))

            self.out_quantization_init_fn = __init_out_quantization__

    def __enabled__(self):
        return self.enabled and self.active and self.bits_out is not None

    def forward(self, *input):
        # Uncomment to enable dump
        # torch.save(*input, os.path.join('dump', self.name + '_in' + '.pt'))

        if self.post_relu:
            out = self.wrapped_module(*input)

            # Quantize output
            if self.__enabled__():
                self.verify_initialized(self.out_quantization, out, self.out_quantization_init_fn)
                out = self.out_quantization(out)
        else:
            # Quantize output
            if self.__enabled__():
                self.verify_initialized(self.out_quantization, *input, self.out_quantization_init_fn)
                out = self.out_quantization(*input)
            else:
                out = self.wrapped_module(*input)

        # Uncomment to enable dump
        # torch.save(out, os.path.join('dump', self.name + '_out' + '.pt'))

        return out

    def get_quantization(self):
        return self.out_quantization

    def set_quantization(self, qtypy, kwargs, verbose=False):
        self.out_quantization = qtypy(self, self.bits_out, symmetric=(not is_positive(self.wrapped_module)),
                                      uint=True, kwargs=kwargs)
        if verbose:
            print("ActivationModuleWrapperPost - {} | {} | {}".format(self.name, str(self.out_quantization),
                                                                      str(kwargs['device'])))

    def set_quant_method(self, method=None):
        if self.bits_out is not None:
            if method == 'kmeans':
                self.out_quantization = KmeansQuantization(self.bits_out)
            else:
                self.out_quantization = self.out_quantization_default

    @staticmethod
    def verify_initialized(quantization_handle, tensor, init_fn):
        if quantization_handle is None:
            init_fn(tensor)

    def log_state(self, step, ml_logger):
        if self.__enabled__():
            if self.out_quantization is not None:
                for n, p in self.out_quantization.named_parameters():
                    if p.numel() == 1:
                        ml_logger.log_metric(self.name + '.' + n, p.item(),  step='auto')
                    else:
                        for i, e in enumerate(p):
                            ml_logger.log_metric(self.name + '.' + n + '.' + str(i), e.item(),  step='auto')


class ParameterModuleWrapperPost(nn.Module):
    def __init__(self, name, wrapped_module, **kwargs):
        super(ParameterModuleWrapperPost, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.forward_functor = kwargs['forward_functor']
        self.bit_weights = kwargs['bits_weight']
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.enabled = True
        self.active = True
        self.centroids_hist = {}
        self.log_weights_hist = False
        self.log_weights_mse = False
        self.log_clustering = False
        self.bn = kwargs['bn'] if 'bn' in kwargs else None
        self.dynamic_weight_quantization = True
        self.bcorr_w = kwargs['bcorr_w']

        setattr(self, 'weight', wrapped_module.weight)
        delattr(wrapped_module, 'weight')
        if hasattr(wrapped_module, 'bias'):
            setattr(self, 'bias', wrapped_module.bias)
            delattr(wrapped_module, 'bias')

        if self.bit_weights is not None:
            self.weight_quantization_default = quantization_mapping[self.qtype](self, self.weight, self.bit_weights,
                                                                             symmetric=True, uint=True, kwargs=kwargs)
            self.weight_quantization = self.weight_quantization_default
            if not self.dynamic_weight_quantization:
                self.weight_q = self.weight_quantization(self.weight)
                self.weight_mse = torch.mean((self.weight_q - self.weight)**2).item()
            print("ParameterModuleWrapperPost - {} | {} | {}".format(self.name, str(self.weight_quantization),
                                                                      str(self.weight.device)))

    def __enabled__(self):
        return self.enabled and self.active and self.bit_weights is not None

    def forward(self, *input):
        w = self.weight
        if self.__enabled__():
            # Quantize weights
            if self.dynamic_weight_quantization:
                w = self.weight_quantization(self.weight)

                if self.bcorr_w:
                    bias_q = w.view(w.shape[0], -1).mean(-1)
                    bias_q = bias_q.view(bias_q.numel(), 1, 1, 1) if len(w.shape) == 4 else bias_q.view(bias_q.numel(), 1)

                    bias_orig = self.weight.view(self.weight.shape[0], -1).mean(-1)
                    bias_orig = bias_orig.view(bias_orig.numel(), 1, 1, 1) if len(self.weight.shape) == 4 else bias_orig.view(
                        bias_orig.numel(), 1)

                    w = w - bias_q + bias_orig
            else:
                w = self.weight_q

        out = self.forward_functor(*input, weight=w, bias=(self.bias if hasattr(self, 'bias') else None))

        return out

    def get_quantization(self):
        return self.weight_quantization

    def set_quantization(self, qtypy, kwargs, verbose=False):
        self.weight_quantization = qtypy(self, self.bit_weights, symmetric=True, uint=True, kwargs=kwargs)
        if verbose:
            print("ParameterModuleWrapperPost - {} | {} | {}".format(self.name, str(self.weight_quantization),
                                                                      str(kwargs['device'])))

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
            if self.weight_quantization is not None:
                for n, p in self.weight_quantization.loggable_parameters():
                    if p.numel() == 1:
                        ml_logger.log_metric(self.name + '.' + n, p.item(),  step='auto')
                    else:
                        for i, e in enumerate(p):
                            ml_logger.log_metric(self.name + '.' + n + '.' + str(i), e.item(),  step='auto')

            if self.log_weights_hist:
                ml_logger.tf_logger.add_histogram(self.name + '.weight', self.weight.cpu().flatten(),  step='auto')

            if self.log_weights_mse:
                ml_logger.log_metric(self.name + '.mse_q', self.weight_mse,  step='auto')
