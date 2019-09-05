import torch
import torch.nn as nn
from itertools import count
from quantization.methods import KmeansQuantization, MinMaxQuantization


class ActivationModuleWrapperPost(nn.Module):
    def __init__(self, name, wrapped_module, quantization_scheduler, **kwargs):
        super(ActivationModuleWrapperPost, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.quantization_scheduler = quantization_scheduler
        self.bits_out = kwargs['bits_out']
        self.enabled = True
        self.active = True

        if self.bits_out is not None:

            self.out_quantization = self.out_quantization_default = None

            def __init_out_quantization__(tensor):
                self.out_quantization_default = MinMaxQuantization(self, self.bits_out, symmetric=False)
                self.out_quantization = self.out_quantization_default

                if self.quantization_scheduler is not None:
                    self.quantization_scheduler.add_quantization_params(self.out_quantization.optim_parameters())

            self.out_quantization_init_fn = __init_out_quantization__

            if self.quantization_scheduler is not None:
                self.quantization_scheduler.register_module_quantization(self)

    def __enabled__(self):
        return self.enabled and self.active and self.bits_out is not None

    def forward(self, *input):
        out = self.wrapped_module(*input)

        # Quantize output
        if self.__enabled__():
            self.verify_initialized(self.out_quantization, out, self.out_quantization_init_fn)
            out = self.out_quantization(out)

        return out

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
