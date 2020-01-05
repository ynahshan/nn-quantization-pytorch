import torch
import torch.nn as nn
from functools import reduce


class KurtosisLoss(object):
    def __init__(self, kurt_target, kurt_lambda, model, criterion, params_func):
        self.kurt_target = kurt_target
        self.kurt_lambda = kurt_lambda
        self.model = model
        self.criterion = criterion
        self.params_func = params_func

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        params = self.params_func(self.model)
        K = [KurtosisLoss.kurtosis(p) for p in params]
        kurt_loss = reduce(lambda a, b: a + b, [(k - self.kurt_target)**2 for k in K]) / len(K)
        orig_loss = self.criterion(input, target) + self.kurt_lambda * kurt_loss
        return orig_loss + kurt_loss

    @staticmethod
    def kurtosis(x):
        return torch.mean(((x - x.mean()) / x.std()) ** 4)
