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
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.meters import AverageMeter, ProgressMeter, accuracy
from torch.utils.data import RandomSampler
import torch.backends.cudnn as cudnn
from quantization.quantizer import ModelQuantizer
from quantization.posttraining.module_wrapper import ActivationModuleWrapperPost, ParameterModuleWrapperPost
from models.resnet import resnet as custom_resnet
from quantization.methods.clipped_uniform import FixedClipValueQuantization


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
parser.add_argument('-cb', '--cal-batch-size', default=64, type=int, help='Batch size for calibration')
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

parser.add_argument('--min_method', '-mm', help='Minimization method to use [Nelder-Mead, Powell, COBYLA]', default='COBYLA')
parser.add_argument('--maxiter', '-maxi', type=int, help='Maximum number of iterations to minimize algo', default=None)
parser.add_argument('--maxfev', '-maxf', type=int, help='Maximum number of function evaluations of minimize algo', default=None)

parser.add_argument('--init_method', default='static',
                    help='Scale initialization method [static, dynamic, random], default=static')
parser.add_argument('-siv', type=float, help='Value for static initialization', default=1.)


class CnnModel(object):
    def __init__(self, arch, use_custom_resnet, pretrained, dataset, gpu_ids, datapath, batch_size, shuffle, workers,
                 print_freq, cal_batch_size):
        self.arch = arch
        self.use_custom_resnet = use_custom_resnet
        self.pretrained = pretrained
        self.dataset = dataset
        self.gpu_ids = gpu_ids
        self.datapath = datapath
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        self.print_freq = print_freq
        self.cal_batch_size = cal_batch_size

        # create model
        if 'resnet' in arch and use_custom_resnet:
            model = custom_resnet(arch=arch, pretrained=pretrained, depth=self.__arch2depth__(arch),
                                  dataset=dataset)
        elif pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

        torch.cuda.set_device(gpu_ids[0])
        model = model.to(self.device)

        if len(gpu_ids) > 1:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, gpu_ids)
            else:
                model = torch.nn.DataParallel(model, gpu_ids)

        self.model = model

        # define loss function (criterion) and optimizer
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        val_data = get_dataset(dataset, 'val', get_transform(dataset, augment=False),
                               datasets_path=datapath)
        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True)

        self.cal_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.cal_batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True)

    @staticmethod
    def __arch2depth__(arch):
        depth = None
        if 'resnet18' in arch:
            depth = 18
        elif 'resnet34' in arch:
            depth = 34
        elif 'resnet50' in arch:
            depth = 50
        elif 'resnet101' in arch:
            depth = 101

        return depth

    def evaluate_calibration(self):
        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            if not hasattr(self, 'cal_set'):
                self.cal_set = []
                # TODO: Workaround, refactor this later
                for i, (images, target) in enumerate(self.cal_loader):
                    if i * self.cal_batch_size >= 512:
                        break
                    images = images.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    self.cal_set.append((images, target))

            res = torch.tensor([0.]).to(self.device)
            for i in range(len(self.cal_set)):
                images, target = self.cal_set[i]
                # compute output
                output = self.model(images)
                loss = self.criterion(output, target)
                res += loss

            return res / len(self.cal_set)

    def validate(self):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(self.val_loader), batch_time, losses, top1, top5,
                                 prefix='Test: ')

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # compute output
                output = self.model(images)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    progress.print(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        return top1.avg


# TODO: refactor this
_eval_count = count(0)
_min_loss = 1e6


def run_inference_on_batch(scales, model, mq):
    global _eval_count, _min_loss
    eval_count = next(_eval_count)

    qwrappers = [qwrapper for (name, qwrapper) in mq.quantization_wrappers if isinstance(qwrapper, ActivationModuleWrapperPost)]
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


def set_clipping(mq, clipping, device):
    qwrappers = [qwrapper for (name, qwrapper) in mq.quantization_wrappers if
                 isinstance(qwrapper, ActivationModuleWrapperPost)]
    for i, qwrapper in enumerate(qwrappers):
        qwrapper.set_quantization(FixedClipValueQuantization,
                                  {'clip_value': clipping[i], 'device': device})


def get_clipping(mq):
    clipping = []
    qwrappers = [qwrapper for (name, qwrapper) in mq.quantization_wrappers if
                 isinstance(qwrapper, ActivationModuleWrapperPost)]
    for i, qwrapper in enumerate(qwrappers):
        q = qwrapper.get_quantization()
        clip_value = getattr(q, 'alpha')
        clipping.append(clip_value)

    return np.array(clipping)


def main():
    args = parser.parse_args()

    # Fix the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create model
    # Always enable shuffling to avoid issues where we get bad results due to weak statistics
    inf_model = CnnModel(args.arch, args.custom_resnet, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                         batch_size=args.batch_size, shuffle=True, workers=args.workers, print_freq=args.print_freq,
                         cal_batch_size=args.cal_batch_size)

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
        init = get_clipping(mq)
    else:
        if args.init_method == 'static':
            init = np.array([args.siv] * len(layers))
        elif args.init_method == 'random':
            init = np.random.uniform(0.5, 1.5, size=len(layers))  # TODO: pass range by argument
        else:
            raise RuntimeError("Invalid argument init_method {}".format(args.init_method))

        # set clip value to qwrappers
        set_clipping(mq, init, inf_model.device)
        print("scales initialization: {}".format(str(init)))

        # evaluate with clipping
        loss = inf_model.evaluate_calibration()
        print("Initial loss: {:.4f}".format(loss.item()))

    # evaluate
    inf_model.validate()

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

    res = opt.minimize(lambda scales: run_inference_on_batch(scales, inf_model, mq), np.array(init),
                       method=args.min_method, options=min_options, callback=local_search_callback)

    # def annealing_callback(x, f, context):
    #     print("Annealing callback")
    #     print(x)
    #     print("loss: {:.4f}".format(f))
    #     print("context: {}".format(context))
    # res = opt.dual_annealing(lambda scales: run_inference_on_batch(scales, inf_model, mq), maxiter=10, visit=2,
    #                          bounds=[(0., 2.)]*len(layers), x0=init, callback=annealing_callback, accept=1e3,
    #                          local_search_options={'method': 'Powell', 'options': {'maxiter': 1, 'disp': True},
    #                                                'callback': local_search_callback})
    print(res)
    scales = res.x
    qwrappers = [qwrapper for (name, qwrapper) in mq.quantization_wrappers if isinstance(qwrapper, ActivationModuleWrapperPost)]
    for i, qwrapper in enumerate(qwrappers):
        if i < len(scales):
            qwrapper.set_quantization(FixedClipValueQuantization, {'clip_value': scales[i], 'device': inf_model.device})
    # evaluate
    inf_model.validate()
    # save scales


if __name__ == '__main__':
    main()