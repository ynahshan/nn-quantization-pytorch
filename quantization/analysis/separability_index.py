import os, sys, time, random
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import argparse
import torch
import torchvision.models as models
import scipy.optimize as opt
from pathlib import Path
import numpy as np
import torch.nn as nn
from itertools import count
import torch.backends.cudnn as cudnn
from quantization.quantizer import ModelQuantizer
from quantization.posttraining.module_wrapper import ActivationModuleWrapperPost, ParameterModuleWrapperPost
from quantization.methods.clipped_uniform import FixedClipValueQuantization
from utils.mllog import MLlogger
from quantization.posttraining.cnn_classifier import CnnModel
from tqdm import tqdm


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

home = str(Path.home())
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name')
parser.add_argument('--datapath', metavar='DATAPATH', type=str, default=None,
                    help='dataset folder')
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-cb', '--cal-batch-size', default=64, type=int, help='Batch size for calibration')
parser.add_argument('-cs', '--cal-set-size', default=64, type=int, help='Batch size for calibration')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--custom_resnet', action='store_true', help='use custom resnet implementation')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu_ids', default=[0], type=int, nargs='+',
                    help='GPU ids to use (e.g 0 1 2 3)')
parser.add_argument('--shuffle', '-sh', action='store_true', help='shuffle data')

parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
parser.add_argument('--bit_weights', '-bw', type=int, help='Number of bits for weights', default=None)
parser.add_argument('--bit_act', '-ba', type=int, help='Number of bits for activations', default=None)
parser.add_argument('--pre_relu', dest='pre_relu', action='store_true', help='use pre-ReLU quantization')
parser.add_argument('--qtype', default='max_static', help='Type of quantization method')
parser.add_argument('--dont_fix_np_seed', '-dfns', action='store_true', help='Do not fix np seed even if seed specified')

parser.add_argument('--num_iter', '-i', type=int, help='Number of bits for activations', default=3)
parser.add_argument('--num_points', '-n', type=int, help='Number of bits for activations', default=100)


def separability_index(f, m, n, k=1, gpu=False, status_callback=None):
    g = None
    max_ = None
    gamma_ = []
    T_ = []
    for j in range(k):
        x, z = torch.tensor(np.random.uniform(0, 1, size=(n, m))).float(), torch.tensor(
            np.random.uniform(0, 1, size=(n, m))).float()
        if gpu:
            x = x.cuda()
            z = z.cuda()

        print("Calaculate f(x)")
        fx = f(x).double()
        if max_ is None:
            max_ = fx.max()
        else:
            max_ = torch.max(max_, fx.max())
        print("Calaculate f(z)")
        fz = f(z).double()
        max_ = torch.max(max_, fz.max())

        t1 = fx + (m - 1) * fz

        print("Calaculate f(xj,zj')")
        t2 = t1.new_zeros((n,), dtype=torch.float64)
        for i in range(m):
            y = z.clone()
            y[:, i] = x[:, i]
            fy = f(y).double()
            max_ = torch.max(max_, fy.max())
            t2 += fy

        g_ = fx * (t1 - t2)
        if g is None:
            g = g_
        else:
            g = torch.cat([g, g_], dim=0)

        gamma = g.mean()
        gamma_.append(gamma.cpu().item())

        s = torch.sqrt(torch.sum((g - gamma) ** 2) / (g.numel() - 1))
        T = np.sqrt(g.numel()) * gamma / max(s, 1e-2)
        print(s)
        T_.append(T.cpu().item())

        if status_callback is not None:
            status_callback(j, gamma, T, max_)

    return gamma_, T_, max_

# assum x is matrix (n,m) in range [0,1]
# n - number of sumples
# m - number of variables
def model_func(x, scales, inf_model, mq, a, b):
    loss = x.new_empty(x.shape[0])
    for i in tqdm(range(x.shape[0])):
        # in general do transformation X: [0, 1] => [a, b] where [a, b] is region of interest
        # e.g. region around point that minimizes some metric
        # We can do simple linear transformation (x + alpha) / beta where
        alpha = a / (b - a)
        beta = 1 / (b - a)
        r = (x[i] + alpha) / beta
        r = torch.min(r, scales)
        r = torch.max(r, r.new_zeros(1))
        mq.set_clipping(r, inf_model.device)

        # evaluate with clipping
        loss[i] = inf_model.evaluate_calibration()
    return loss


def main(args, ml_logger):
    # Fix the seed
    random.seed(args.seed)
    if not args.dont_fix_np_seed:
        np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.qtype = 'max_static'
    # create model
    # Always enable shuffling to avoid issues where we get bad results due to weak statistics
    inf_model = CnnModel(args.arch, args.custom_resnet, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                         batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                         cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size)

    all_layers = []
    if args.bit_weights is not None:
        all_layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.Conv2d)][1:-1]
    if args.bit_act is not None:
        all_layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU)][1:-1]
    if args.bit_act is not None and 'mobilenet' in args.arch:
        all_layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU6)][1:-1]

    replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                           nn.ReLU6: ActivationModuleWrapperPost,
                           nn.Conv2d: ParameterModuleWrapperPost}
    mq = ModelQuantizer(inf_model.model, args, all_layers, replacement_factory)

    loss = inf_model.evaluate_calibration()
    print("loss: {:.4f}".format(loss.item()))
    ml_logger.log_metric('loss', loss.item(), step='auto')

    # get clipping values
    p_max = mq.get_clipping()
    # print(init)

    args.qtype = 'l2_norm'
    inf_model = CnnModel(args.arch, args.custom_resnet, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                         batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                         cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size)
    mq = ModelQuantizer(inf_model.model, args, all_layers, replacement_factory)
    loss = inf_model.evaluate_calibration()
    print("loss l2: {:.4f}".format(loss.item()))
    p_l2 = mq.get_clipping()

    args.qtype = 'l3_norm'
    inf_model = CnnModel(args.arch, args.custom_resnet, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                         batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                         cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size)
    mq = ModelQuantizer(inf_model.model, args, all_layers, replacement_factory)
    loss = inf_model.evaluate_calibration()
    print("loss l2: {:.4f}".format(loss.item()))
    p_l3 = mq.get_clipping()

    # gamma_avg = 0
    # T_avg = 0
    num_iter = args.num_iter
    n = args.num_points

    def status_callback(i, gamma, T, f_max):
        T = T.item()
        gamma = gamma.item()
        f_max = f_max.item()

        print("gamma^2: {}, T: {}, max: {}".format(gamma, T, f_max))
        ml_logger.log_metric('gamma', gamma, step='auto')
        ml_logger.log_metric('T', T, step='auto')
        ml_logger.log_metric('f_max', f_max, step='auto')
        T_norm = T / np.sqrt(i+1)
        ml_logger.log_metric('T_norm', T_norm, step='auto')
        gamma_norm = gamma / f_max**2
        ml_logger.log_metric('gamma_norm', gamma_norm, step='auto')

    gamma_, T_, f_max = separability_index(lambda x: model_func(x, p_max, inf_model, mq, p_l2, p_l3), len(p_max), n, num_iter,
                                  gpu=True, status_callback=status_callback)

    gamma_norm = np.mean(np.array(gamma_) / f_max.item()**2)
    T_norm = np.mean(np.array(T_) / np.sqrt(np.arange(1, num_iter + 1)))

    print("gamma^2 norm: {}, T norm: {}".format(gamma_norm, T_norm))
    ml_logger.log_metric('gamma_tot', gamma_norm, step='auto')
    ml_logger.log_metric('T_tot', T_norm, step='auto')


if __name__ == '__main__':
    args = parser.parse_args()
    with MLlogger(os.path.join(home, 'mxt-sim/mllog_runs'), args.experiment, args,
                  name_args=[args.arch, args.dataset, "W{}A{}".format(args.bit_weights, args.bit_act)]) as ml_logger:
        main(args, ml_logger)
