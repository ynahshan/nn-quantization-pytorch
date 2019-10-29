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
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-cb', '--cal-batch-size', default=None, type=int, help='Batch size for calibration')
parser.add_argument('-cs', '--cal-set-size', default=None, type=int, help='Batch size for calibration')
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
parser.add_argument('--qtype', default='aciq_laplace', help='Type of quantization method')
parser.add_argument('-lp', type=float, help='p parameter of Lp norm', default=3.)

parser.add_argument('--min_method', '-mm', help='Minimization method to use [Nelder-Mead, Powell, COBYLA]', default='Powell')
parser.add_argument('--maxiter', '-maxi', type=int, help='Maximum number of iterations to minimize algo', default=None)
parser.add_argument('--maxfev', '-maxf', type=int, help='Maximum number of function evaluations of minimize algo', default=None)

parser.add_argument('--init_method', default='static',
                    help='Scale initialization method [static, dynamic, random], default=static')
parser.add_argument('-siv', type=float, help='Value for static initialization', default=1.)

parser.add_argument('--dont_fix_np_seed', '-dfns', action='store_true', help='Do not fix np seed even if seed specified')


# TODO: refactor this
_eval_count = count(0)
_min_loss = 1e6


def run_inference_on_batch(scales, model, mq):
    global _eval_count, _min_loss
    eval_count = next(_eval_count)

    qwrappers = mq.get_qwrappers()
    for i, qwrapper in enumerate(qwrappers):
        if i < len(scales):
            qwrapper.set_quantization(FixedClipValueQuantization, {'clip_value': scales[i], 'device': model.device},
                                      verbose=(eval_count % 300 == 0))
    loss = model.evaluate_calibration().item()

    if loss < _min_loss:
        _min_loss = loss

    print_freq = 20
    if eval_count % 20 == 0:
        print("func eval iteration: {}, minimum loss of last {} iterations: {:.4f}".format(
            eval_count, print_freq, _min_loss))

    return loss


def coord_descent(fun, init, args, **kwargs):
    maxiter = kwargs['maxiter']
    x = init.copy()

    def coord_opt(alpha, scales, i):
        if alpha < 0:
            result = 1e6
        else:
            scales[i] = alpha
            result = fun(scales)

        return result

    nfev = 0
    for j in range(maxiter):
        for i in range(len(x)):
            print("Optimizing variable {}".format(i))
            r = opt.minimize_scalar(lambda alpha: coord_opt(alpha, x, i))
            nfev += r.nfev
            opt_alpha = r.x
            x[i] = opt_alpha

    res = opt.OptimizeResult()
    res.x = x
    res.nit = maxiter
    res.nfev = nfev
    res.success = True

    return res


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

    layers = []
    # TODO: make it more generic
    if args.bit_weights is not None:
        layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.Conv2d)][1:-1]
    if args.bit_act is not None:
        layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU)][1:-1]
    if args.bit_act is not None and 'mobilenet' in args.arch:
        layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU6)][1:-1]

    replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                           nn.ReLU6: ActivationModuleWrapperPost,
                           nn.Conv2d: ParameterModuleWrapperPost}
    mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)
    # mq.log_quantizer_state(ml_logger, -1)

    print("init_method: {}, qtype {}".format(args.init_method, args.qtype))
    # initialize scales
    if args.init_method == 'dynamic':
        # evaluate to initialize dynamic clipping
        loss = inf_model.evaluate_calibration()
        print("Initial loss: {:.4f}".format(loss.item()))

        # get clipping values
        init = mq.get_clipping()
    else:
        if args.init_method == 'static':
            init = np.array([args.siv] * len(layers))
        elif args.init_method == 'random':
            init = np.random.uniform(0.5, 1., size=len(layers))  # TODO: pass range by argument
        else:
            raise RuntimeError("Invalid argument init_method {}".format(args.init_method))

        # set clip value to qwrappers
        mq.set_clipping(init, inf_model.device)
        print("scales initialization: {}".format(str(init)))

        # evaluate with clipping
        loss = inf_model.evaluate_calibration()
        print("Initial loss: {:.4f}".format(loss.item()))

    global _min_loss
    _min_loss = loss.item()

    # evaluate
    # acc = inf_model.validate()
    # ml_logger.log_metric('Acc init', acc, step='auto')

    # run optimizer
    min_options = {}
    if args.maxiter is not None:
        min_options['maxiter'] = args.maxiter
    if args.maxfev is not None:
        min_options['maxfev'] = args.maxfev

    _iter = count(0)

    def local_search_callback(x):
        it = next(_iter)
        loss = run_inference_on_batch(x, inf_model, mq)
        print("\n[{}]: Local search callback".format(it))
        print("loss: {:.4f}\n".format(loss))

    method = coord_descent if args.min_method == 'CD' else args.min_method
    res = opt.minimize(lambda scales: run_inference_on_batch(scales, inf_model, mq), init.cpu().numpy(),
                       method=method, options=min_options, callback=local_search_callback)

    print(res)
    scales = res.x

    mq.set_clipping(scales, inf_model.device)

    # evaluate
    acc = inf_model.validate()
    ml_logger.log_metric('Acc {}'.format(args.min_method), acc, step='auto')
    # save scales


if __name__ == '__main__':
    args = parser.parse_args()
    if args.cal_batch_size is None:
        args.cal_batch_size = args.batch_size
    if args.cal_batch_size > args.batch_size:
        print("Changing cal_batch_size parameter from {} to {}".format(args.cal_batch_size, args.batch_size))
        args.cal_batch_size = args.batch_size
    if args.cal_set_size is None:
        args.cal_set_size = args.batch_size

    with MLlogger(os.path.join(home, 'mxt-sim/mllog_runs'), args.experiment, args,
                  name_args=[args.arch, args.dataset, "W{}A{}".format(args.bit_weights, args.bit_act)]) as ml_logger:
        main(args, ml_logger)
