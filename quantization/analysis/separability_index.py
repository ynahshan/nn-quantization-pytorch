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


def separability_index(f, m, n, gpu=False, gamma_expecation=True):
    x, z = torch.tensor(np.random.uniform(0, 1, size=(n, m))).float(), torch.tensor(np.random.uniform(0, 1, size=(n, m))).float()
    if gpu:
        x = x.cuda()
        z = z.cuda()

    print("Calaculate f(x)")
    t0 = f(x)
    print("Calaculate f(z)")
    t1 = t0 + (m - 1)*f(z)

    print("Calaculate f(xj,zj')")
    t2 = t1.new_zeros((n,))
    for i in range(m):
        y = z.clone()
        y[:,i] = x[:,i]
        t2 += f(y)

    g = t0 * (t1 - t2)
    gamma = g.mean()

    s = torch.sqrt(torch.sum((g - gamma)**2) / (n - 1))
    T = np.sqrt(n) * gamma / max(s, 1e-8)

    if gamma_expecation:
        return gamma, T
    else:
        return g, T

# assum x is matrix (n,m) in range [0,1]
# n - number of sumples
# m - number of variables
def model_func(x, scales, inf_model, mq):
    loss = x.new_empty(x.shape[0])
    for i in tqdm(range(x.shape[0])):
        # set clip value. rescale to [0.5,1] to avoid radical saturation
        r = (x[i] + 1) / 2
        mq.set_clipping(r*scales, inf_model.device)

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
    init = mq.get_clipping()
    # print(init)

    gamma_avg = 0
    T_avg = 0
    num_iter = args.num_iter
    n = args.num_points
    for i in range(num_iter):
        gamma, T = separability_index(lambda x: model_func(x, init, inf_model, mq), len(init), n, gpu=True, gamma_expecation=True)
        gamma_avg += gamma.abs()
        T_avg += T.abs()

        print("gamma^2: {}, T: {}".format(gamma.abs(), T))
        ml_logger.log_metric('gamma_sample', gamma.abs().cpu().item(), step='auto')
        ml_logger.log_metric('gamma_avg', gamma_avg.cpu().item() / (i + 1), step='auto')

    gamma = gamma_avg.cpu().item() / num_iter
    T = T_avg.cpu().item() / num_iter

    print("gamma^2: {}, T: {}".format(gamma, T))
    ml_logger.log_metric('gamma', gamma, step='auto')
    ml_logger.log_metric('T', T, step='auto')


if __name__ == '__main__':
    args = parser.parse_args()
    with MLlogger(os.path.join(home, 'mxt-sim/mllog_runs'), args.experiment, args,
                  name_args=[args.arch, args.dataset, "W{}A{}".format(args.bit_weights, args.bit_act)]) as ml_logger:
        main(args, ml_logger)
