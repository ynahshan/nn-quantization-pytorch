import os, sys, time, random
proj_root_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
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
from tqdm import tqdm
import pickle


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
parser.add_argument('-cb', '--cal-batch-size', default=256, type=int, help='Batch size for calibration')
parser.add_argument('-cs', '--cal-set-size', default=256, type=int, help='Batch size for calibration')
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

parser.add_argument('--bn_folding', '-bnf', action='store_true', help='Apply Batch Norm folding', default=False)
parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
parser.add_argument('--bit_weights', '-bw', type=int, help='Number of bits for weights', default=None)
parser.add_argument('--bit_act', '-ba', type=int, help='Number of bits for activations', default=None)
parser.add_argument('--pre_relu', dest='pre_relu', action='store_true', help='use pre-ReLU quantization')
parser.add_argument('--qtype', default='max_static', help='Type of quantization method')
parser.add_argument('-lp', type=float, help='p parameter of Lp norm', default=2.)
parser.add_argument('--dont_fix_np_seed', '-dfns', action='store_true', help='Do not fix np seed even if seed specified')

parser.add_argument('--grid_resolution', '-gr', type=int, help='Number of intervals in the grid, one coordinate.', default=11)


def main(args):
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
    custom_resnet = True
    custom_inception = True
    replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                           nn.ReLU6: ActivationModuleWrapperPost,
                           nn.Conv2d: ParameterModuleWrapperPost}

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

        all_layers = []
        if args.bit_weights is not None:
            all_layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.Conv2d)][1:-1]
        if args.bit_act is not None:
            all_layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU)][1:-1]
        if args.bit_act is not None and 'mobilenet' in args.arch:
            all_layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU6)][1:-1]

        mq = ModelQuantizer(inf_model.model, args, all_layers, replacement_factory)
        loss = inf_model.evaluate_calibration()
        point = mq.get_clipping()

        # evaluate
        # acc = inf_model.validate()

        del inf_model
        del mq

        return point, loss

    ps = np.linspace(2, 4.25, 100)
    losses = []
    for p in tqdm(ps):
        point, loss = eval_pnorm(p)
        losses.append(loss.item())
        print("(p, loss) - ({}, {})".format(p, loss.item()))

    f_name = "{}_W{}A{}_loss_vs_p.pkl".format(args.arch, args.bit_weights, args.bit_act)
    f = open(os.path.join(proj_root_dir, 'data', f_name), 'wb')
    data = {'p': ps, 'loss': losses}
    pickle.dump(data, f)
    f.close()
    print("Data saved to {}".format(f_name))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
