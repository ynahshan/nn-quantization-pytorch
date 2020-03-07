import os, sys, time, random
proj_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(proj_root_dir)
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
import pickle
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
parser.add_argument('--custom_inception', action='store_true', help='use custom inception implementation')
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
parser.add_argument('-lp', type=float, help='p parameter of Lp norm', default=3.)

parser.add_argument('--min_method', '-mm', help='Minimization method to use [Nelder-Mead, Powell, COBYLA]', default='Powell')
parser.add_argument('--maxiter', '-maxi', type=int, help='Maximum number of iterations to minimize algo', default=None)
parser.add_argument('--maxfev', '-maxf', type=int, help='Maximum number of function evaluations of minimize algo', default=None)

parser.add_argument('--init_method', default='static',
                    help='Scale initialization method [static, dynamic, random], default=static')
parser.add_argument('-siv', type=float, help='Value for static initialization', default=1.)

parser.add_argument('--dont_fix_np_seed', '-dfns', action='store_true', help='Do not fix np seed even if seed specified')
parser.add_argument('--bcorr_w', '-bcw', action='store_true', help='Bias correction for weights', default=False)
parser.add_argument('--tag', help='Tag for logging purposes', default='n/a')
parser.add_argument('--bn_folding', '-bnf', action='store_true', help='Apply Batch Norm folding', default=False)


# TODO: refactor this
_eval_count = count(0)
_min_loss = 1e6


def evaluate_calibration_clipped(scales, model, mq):
    global _eval_count, _min_loss
    eval_count = next(_eval_count)

    mq.set_clipping(scales, model.device)
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

        if 'callback' in kwargs:
            kwargs['callback'](x)

    res = opt.OptimizeResult()
    res.x = x
    res.nit = maxiter
    res.nfev = nfev
    res.fun = np.array([r.fun])
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

    if args.tag is not None:
        ml_logger.mlflow.log_param('tag', args.tag)

    enable_bcorr = False
    if args.bcorr_w:
        args.bcorr_w = False
        enable_bcorr = True

    args.qtype = 'max_static'
    # create model
    # Always enable shuffling to avoid issues where we get bad results due to weak statistics
    custom_resnet = True
    custom_inception = True
    inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                         batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                         cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)

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
    maxabs_loss = inf_model.evaluate_calibration()
    print("max loss: {:.4f}".format(maxabs_loss.item()))
    max_point = mq.get_clipping()
    ml_logger.log_metric('Loss max', maxabs_loss.item(), step='auto')

    # evaluate
    maxabs_acc = 0#inf_model.validate()
    ml_logger.log_metric('Acc maxabs', maxabs_acc, step='auto')
    data = {'max': {'alpha': max_point.cpu().numpy(), 'loss': maxabs_loss.item(), 'acc': maxabs_acc}}

    del inf_model
    del mq

    def eval_pnorm(p):
        args.qtype = 'lp_norm'
        args.lp = p
        # Fix the seed
        random.seed(args.seed)
        if not args.dont_fix_np_seed:
            np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                             batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                             cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)

        mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)
        loss = inf_model.evaluate_calibration()
        point = mq.get_clipping()

        # evaluate
        acc = inf_model.validate()

        del inf_model
        del mq

        return point, loss, acc

    def eval_pnorm_on_calibration(p):
        args.qtype = 'lp_norm'
        args.lp = p
        # Fix the seed
        random.seed(args.seed)
        if not args.dont_fix_np_seed:
            np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                             batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                             cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)

        mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)
        loss = inf_model.evaluate_calibration()
        point = mq.get_clipping()

        del inf_model
        del mq

        return point, loss

    # l2_point, l2_loss = eval_pnorm_on_calibration(2)
    # print("loss l2: {:.4f}".format(l2_loss.item()))
    # ml_logger.log_metric('Loss l2', l2_loss.item(), step='auto')
    # # l4_point, l4_loss = eval_pnorm_on_calibration(4)
    # print("loss l4: {:.4f}".format(l4_loss.item()))
    # ml_logger.log_metric('Loss l4', l4_loss.item(), step='auto')

    # args.qtype = 'lp_norm'
    # args.lp = p_intr
    # Fix the seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                         batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                         cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)

    mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)

    opt_point = np.array([0.42811054, 1.27721779, 0.53149996, 1.51492159, 0.91115569,
       1.17987683, 1.13352566, 1.5227828 , 0.67026185, 0.75535328,
       0.54173654, 0.70824616, 0.44899457, 1.25257411, 0.68778409])
    start_point = 0.8*opt_point
    end_point = 1.5*opt_point
    k = 100
    step = (end_point - start_point) / k
    print("start")
    print(start_point)
    print("end")
    print(end_point)

    losses = []
    points = []
    for i in range(k+1):
        point = start_point + i * step
        mq.set_clipping(point, inf_model.device)
        loss = inf_model.evaluate_calibration()
        losses.append(loss.item())
        points.append(point)
        print("({}: loss) - {}".format(i, loss.item()))

    data = {'opt': opt_point, 'points': points, 'loss': losses}

    # save scales
    f_name = "quadratic_loss_{}_W{}A{}.pkl".format(args.arch, args.bit_weights, args.bit_act)
    f = open(os.path.join(proj_root_dir, 'data', f_name), 'wb')
    pickle.dump(data, f)
    f.close()
    print("Data saved to {}".format(f_name))

    # del inf_model
    # del mq
    #
    # # Interpolate optimal p
    # z = np.polyfit(np.arange(len(losses)), losses, 2)
    # y = np.poly1d(z)
    # p_opt = y.deriv().roots[0]
    # opt_point = start_point + p_opt * step
    # loss_opt = y(p_opt)
    #
    # print("loss p intr: {:.4f}".format(loss_opt.item()))
    # ml_logger.log_metric('Loss p intr', loss_opt.item(), step='auto')
    #
    # global _eval_count, _min_loss
    # _min_loss = loss_opt.item()
    #
    # init = opt_point
    #
    # if enable_bcorr:
    #     args.bcorr_w = True
    # random.seed(args.seed)
    # if not args.dont_fix_np_seed:
    #     np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    #
    # inf_model = CnnModel(args.arch, custom_resnet, custom_inception, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
    #                      batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
    #                      cal_batch_size=args.cal_batch_size, cal_set_size=args.cal_set_size, args=args)
    #
    # mq = ModelQuantizer(inf_model.model, args, layers, replacement_factory)
    #
    # # run optimizer
    # min_options = {}
    # if args.maxiter is not None:
    #     min_options['maxiter'] = args.maxiter
    # if args.maxfev is not None:
    #     min_options['maxfev'] = args.maxfev
    #
    # _iter = count(0)
    #
    # def local_search_callback(x):
    #     it = next(_iter)
    #     mq.set_clipping(x, inf_model.device)
    #     loss = inf_model.evaluate_calibration()
    #     print("\n[{}]: Local search callback".format(it))
    #     print("loss: {:.4f}\n".format(loss.item()))
    #     print(x)
    #     ml_logger.log_metric('Loss {}'.format(args.min_method), loss.item(), step='auto')
    #
    #     # evaluate
    #     acc = inf_model.validate()
    #     ml_logger.log_metric('Acc {}'.format(args.min_method), acc, step='auto')
    #
    # args.min_method = "Powell"
    # method = coord_descent if args.min_method == 'CD' else args.min_method
    # res = opt.minimize(lambda scales: evaluate_calibration_clipped(scales, inf_model, mq), init.cpu().numpy(),
    #                    method=method, options=min_options, callback=local_search_callback)
    #
    # print(res)
    #
    # scales = res.x
    # mq.set_clipping(scales, inf_model.device)
    # loss = inf_model.evaluate_calibration()
    # ml_logger.log_metric('Loss {}'.format(args.min_method), loss.item(), step='auto')
    #
    # # evaluate
    # acc = inf_model.validate()
    # ml_logger.log_metric('Acc {}'.format(args.min_method), acc, step='auto')
    # data['powell'] = {'alpha': scales, 'loss': loss.item(), 'acc': acc}
    #
    # # save scales
    # f_name = "scales_{}_W{}A{}.pkl".format(args.arch, args.bit_weights, args.bit_act)
    # f = open(os.path.join(proj_root_dir, 'data', f_name), 'wb')
    # pickle.dump(data, f)
    # f.close()
    # print("Data saved to {}".format(f_name))


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
