import torch
import torch.nn as nn
from clustering.kmeans import lloyd1d
from .uniform import QuantizationBase


class ArgmaxMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        output = input.new_full(input.shape, 0., requires_grad=True)
        output[torch.arange(output.shape[0]), input.argmax(dim=dim)] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class StepQuantizationSte(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, centroids):
        b = (centroids[1:] + centroids[:-1]) / 2
        i = (tensor.view(-1, 1) > b).sum(1)
        return centroids[i].view(tensor.shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class KmeansQuantization(object):
    def __init__(self, num_bits, max_iter=100, rho=0.5, uniform_init=False):
        self.num_bits = num_bits
        self.num_bins = int(2 ** num_bits)
        self.max_iter = max_iter
        self.rho =rho
        self.uniform_init = uniform_init

    def clustering(self, tensor):
        if self.uniform_init:
            # Initialize k-means centroids uniformaly
            rho = 0.5
            bin_size = (tensor.max() - tensor.min()) / self.num_bins
            zp = torch.round(tensor.min() / bin_size)
            x = (torch.arange(self.num_bins, device=tensor.device, dtype=tensor.dtype) + zp) * bin_size
            init = rho * torch.where(x != 0, x + bin_size / 2, x)
        else:
            init = None

        with torch.no_grad():
            cluster_idx, centers = lloyd1d(tensor.flatten(), self.num_bins, max_iter=self.max_iter, init_state=init, tol=1e-5)

        # workaround for out of memory issue
        torch.cuda.empty_cache()

        B = torch.zeros(tensor.numel(), self.num_bins, device=tensor.device)
        B[torch.arange(B.shape[0]), cluster_idx.long()] = 1
        v = centers.flatten()
        return B, v

    def __call__(self, tensor):
        B, v = self.clustering(tensor)
        tensor_q = torch.matmul(B, v).view(tensor.shape)

        return tensor_q


class CentroidsQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, c, T):
        b = (c[1:] + c[:-1]) / 2
        s = c[1:] - c[:-1]
        # TODO: add not symmetric (Relu) case
        g = torch.sigmoid(T * (tensor.view(-1, 1) - b))

        ctx.save_for_backward(b, s, g, T)

        y = torch.sum(s * g, dim=1) + c[0]
        return y.view(tensor.shape)

    @staticmethod
    def backward(ctx, grad_output):
        b, s, g, T = ctx.saved_tensors

        t = s * g * (1 - g)

        grad_x = T * torch.sum(t, dim=1).view(grad_output.shape)
        grad_x.mul_(grad_output)
        grad_x.clamp_(-100, 100)

        # TODO: check dimensions to handle cases where dim=1 less than 4
        grad_ck = g[:, -1].clone()
        grad_ck.sub_((T / 2) * t[:, -1])
        grad_ck.mul_(grad_output.flatten())
        grad_ck = torch.sum(grad_ck, dim=0).view(-1)

        grad_cj = g[:, :-1] - g[:, 1:]
        grad_cj.sub_((T / 2) * (t[:, 1:] + t[:, :-1]))
        grad_cj.mul_(grad_output.view(-1, 1))
        grad_cj = torch.sum(grad_cj, dim=0).view(-1)

        grad_c0 = -g[:, 0].clone()
        grad_c0.sub_((T / 2) * t[:, 0])
        grad_c0.add_(1.)
        grad_c0.mul_(grad_output.flatten())
        grad_c0 = torch.sum(grad_c0, dim=0).view(-1)

        grad_c = torch.clamp(torch.cat((grad_c0, grad_cj, grad_ck)), -1, 1)

        return grad_x, grad_c, None


class LearnedCentroidsQuantization(QuantizationBase):
    c_param_name = 'c'

    def __init__(self, module, num_bits, symmetric, tensor, temperature):
        super(LearnedCentroidsQuantization, self).__init__(module, num_bits, symmetric)
        self.soft_quant = True
        self.hard_quant = False

        with torch.no_grad():
            # Initialize k-means centroids uniformaly
            _, centers = KmeansQuantization(num_bits).clustering(tensor)
            edges, _ = centers.flatten().sort()

        self.register_buffer('centroids', edges)
        self.register_buffer('temperature', tensor.new_tensor([temperature]))
        self.register_buffer('rho', tensor.new_tensor([1.]))

        self.register_parameter(self.c_param_name, edges.new_ones(edges.shape))

        self.__create_optim_params__()

    def __call__(self, tensor):
        c = self.centroids * self.c

        # Soft quantization
        if self.soft_quant:
            T = self.temperature
            # b = (c[1:] + c[:-1]) / 2
            # s = c[1:] - c[:-1]
            #
            # y = torch.sum(s * torch.sigmoid(T * (tensor.view(-1, 1) - b)), dim=1).view(tensor.shape)
            # if self.symmetric:
            #     # In symmetric case shift central bin to 0
            #     tensor_soft = y + c[0]
            # else:
            #     # Asymmetric case after relu, avoid changing zeros in the tensor
            #     tensor_soft = torch.where(tensor > 0, y, tensor)

            tensor_soft = CentroidsQuantization.apply(tensor, c, T)

        # Hard quantization
        if self.hard_quant:
            tensor_hard = StepQuantizationSte().apply(tensor, c)

        if self.soft_quant and self.hard_quant:
            print('Hello')
            assert False
            # tensor_q = self.rho * tensor_soft + (1 - self.rho) * tensor_hard
        elif self.soft_quant:
            tensor_q = tensor_soft
        elif self.hard_quant:
            tensor_q = tensor_hard
        else:
            raise RuntimeError('quantization not defined!!!')

        return tensor_q

    def loggable_parameters(self):
        lp = super(LearnedCentroidsQuantization, self).loggable_parameters()
        return lp + [('ctr', self.centroids * self.c),
                     ('rho', self.rho),
                     ('temperature', self.temperature)]

    def __create_optim_params__(self):
        # 3 bit width configuration. Need to see how to find those parameters more easily per bit width/model/dataset
        # self.__add_optim_params__('SGD', 'imagenet', [
        #     (self.c_param_name, {'params': [self.c], 'lr': 1e-4, 'momentum': 0, 'weight_decay': 0}),
        #     (self.T_param_name, {'params': [self.T], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0})
        # ])
        self.__add_optim_params__('SGD', 'imagenet', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0}),
            # (self.T_param_name, {'params': [self.T], 'lr': 1e-1, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('Adam', 'imagenet', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0}),
            # (self.T_param_name, {'params': [self.T], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('SGD', 'cifar10', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0}),
            # (self.T_param_name, {'params': [self.T], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('Adam', 'cifar10', [
            (self.c_param_name, {'params': [self.c], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0}),
            # (self.T_param_name, {'params': [self.T], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])

    @staticmethod
    def learned_parameters():
        # Use this function to control what parameters to learn
        return [
                LearnedCentroidsQuantization.c_param_name
                ]

    def clustering(self, tensor):
        tq = self.quantize(tensor)
        B_tag = torch.abs(tq.view(-1, 1) - self.c.view(1, -1))
        B = B_tag.new_zeros(B_tag.shape)
        B[torch.arange(B.shape[0]), B_tag.argmin(dim=1)] = 1

        return B, self.c


class LearnedSigmoidQuantization(QuantizationBase):
    alpha_param_name = 'alpha'
    beta_param_name = 'beta'
    b_param_name = 'b'

    def __init__(self, module, num_bits, symmetric, tensor, temperature):
        super(LearnedCentroidsQuantization, self).__init__(module, num_bits, symmetric)
        self.temperature = temperature
        # TODO: change to symmetric
        self.Y = tensor.new_tensor([0, 1, 2, 4])

        with torch.no_grad():
            b = (self.Y[1:] + self.Y[:-1]) / 2

        self.register_parameter(self.beta_param_name, tensor.new_tensor([self.num_bins / tensor.abs().max()]))
        self.register_parameter(self.alpha_param_name, 1 / self.beta)
        self.register_parameter(self.b_param_name, b)

        self.__create_optim_params__()

    def __call__(self, tensor):
        T = self.temperature
        s = self.Y[1:] - self.Y[:-1]

        temp = torch.clamp(T * (self.beta * tensor.view(-1, 1) - self.b), -10, 10)
        # Assume relu, handle zeros issue
        tensor_q = torch.where(tensor > 0,
                               self.alpha * torch.sum(s * torch.sigmoid(temp), dim=1).view(tensor.shape),
                               tensor)
        # tensor_q = LearnableSigmoidQuantization.SigmoidQuantSte().apply(tensor_q, self.alpha, self.beta, s, self.b, 1e10)

        # Hard quantization
        tensor_q = StepQuantizationSte().apply(tensor_q, self.c)

        return tensor_q

    def __create_optim_params__(self):
        self.__add_optim_params__('SGD', 'imagenet', [
            (self.alpha_param_name, {'params': [self.alpha], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0}),
            (self.beta_param_name, {'params': [self.beta], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0}),
            (self.b_param_name, {'params': [self.b], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])

    @staticmethod
    def learned_parameters():
        return [
                LearnedSigmoidQuantization.alpha_param_name,
                LearnedSigmoidQuantization.beta_param_name,
                LearnedSigmoidQuantization.b_param_name
                ]


class LearnableDifferentiableQuantization(object):
    binning_param_name = 'tensor_binning'
    T_param_name = 'lsiq_T'

    def __init__(self, module, tensor, num_bits):
        self.num_bits = num_bits
        self.temperature = 1
        self.num_bins = int(2 ** num_bits)
        self.c = None

        with torch.no_grad():
            # Initialize B with k-means quantization
            B, _ = KmeansQuantization(num_bits).clustering(tensor)

        module.register_parameter(self.binning_param_name, nn.Parameter(B))
        self.tensor_binning = getattr(module, self.binning_param_name)

        module.register_parameter(self.T_param_name, nn.Parameter(tensor.new_tensor([1 / self.temperature])))
        self.T = getattr(module, self.T_param_name)

        self.sm = nn.Softmax(dim=1)

    def clustering(self, tensor):
        B = ArgmaxMaskSTE().apply(self.tensor_binning)
        v = torch.matmul(tensor.view(-1), B) / B.sum(dim=0)
        return B, v

    def quantize(self, tensor):
        T = 1 / self.T.abs()

        # Soft quantization
        B = self.sm(T * self.tensor_binning)
        v = torch.matmul(tensor.view(-1), B) / B.sum(dim=0)
        tensor_q = torch.matmul(B, v).view(tensor.shape)

        # Hard quantization
        tensor_q = StepQuantizationSte().apply(tensor_q, v)

        self.c = v.detach().cpu()

        return tensor_q

    def loggable_parameters(self):
        return [('c', self.c), (LearnableDifferentiableQuantization.T_param_name, 1 / self.T.detach())]

    def named_parameters(self):
        np = [
              (LearnableDifferentiableQuantization.binning_param_name, self.tensor_binning),
              (LearnableDifferentiableQuantization.T_param_name, self.T),
              ]

        named_params = [(n, p) for n, p in np if n in self.learnable_parameters_names()]
        return named_params

    def optim_parameters(self):
        pdict = [
            (LearnableDifferentiableQuantization.binning_param_name, {'params': [self.tensor_binning], 'lr': 1e-1, 'momentum': 0, 'weight_decay': 0}),
            (LearnableDifferentiableQuantization.T_param_name, {'params': [self.T], 'lr': 1e-2, 'momentum': 0, 'weight_decay': 0}),
        ]

        params = [d for n, d in pdict if n in self.learnable_parameters_names()]
        return params

    @staticmethod
    def learnable_parameters_names():
        return [
                LearnableDifferentiableQuantization.binning_param_name,
                LearnableDifferentiableQuantization.T_param_name
                ]
