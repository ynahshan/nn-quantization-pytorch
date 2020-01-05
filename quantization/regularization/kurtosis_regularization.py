import torch
from functools import reduce


class KurtosisLoss(object):
    def __init__(self, kurt_target, kurt_lambda, model, criterion, layers):
        self.kurt_target = kurt_target
        self.kurt_lambda = kurt_lambda
        self.model = model
        self.criterion = criterion
        self.layers = layers

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        params = [p for n, p in self.model.named_parameters() if 'weight' in n and n.replace('.weight', '') in self.layers]
        assert len(params) == len(self.layers)
        K = [KurtosisLoss.kurtosis(p) for p in params]
        kurt_loss = reduce(lambda a, b: a + b, [(k - self.kurt_target)**2 for k in K]) / len(K)
        orig_loss = self.criterion(input, target) + self.kurt_lambda * kurt_loss
        return orig_loss + kurt_loss

    @staticmethod
    def kurtosis(x):
        return torch.mean(((x - x.mean()) / x.std()) ** 4)
