import torch
import torch.nn as nn


class Noise(object):
    def __init__(self, module, tensor, *args, **kwargs):
        self.dist = torch.distributions.uniform.Uniform(-0.5, 0.5)
        s = 1./2 # TODO: get from kwargs
        with torch.no_grad():
            self.amp = (tensor.max() - tensor.min()) * s

    def __call__(self, tensor):
        noise = self.dist.sample(sample_shape=tensor.shape).to(tensor.device) * self.amp
        return tensor + noise

    def loggable_parameters(self):
        return []
