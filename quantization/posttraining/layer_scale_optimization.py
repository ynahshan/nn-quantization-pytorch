import os, sys, time
import argparse
import torch
import torchvision.models as models
import scipy.optimize as opt
from pathlib import Path
import numpy as np
import torch.nn as nn
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.meters import AverageMeter, ProgressMeter, accuracy
from torch.utils.data import RandomSampler
from quantization.quantizer import ModelQuantizer
from quantization.posttraining.module_wrapper import ActivationModuleWrapperPost, ParameterModuleWrapperPost
from models.resnet import resnet as custom_resnet
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


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
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--custom_resnet', action='store_true', help='use custom resnet implementation')
parser.add_argument('--seed', default=12345, type=int,
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


class CnnModel(object):
    def __init__(self, arch, use_custom_resnet, pretrained, dataset, gpu_ids, datapath, batch_size, shuffle, workers,
                 print_freq):
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
            batch_size=batch_size, shuffle=shuffle,
            sampler=RandomSampler(val_data),
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
            if not hasattr(self, 'cal_batch'):
                # TODO: Workaround, refactor this later
                # TODO: Make it deterministic
                images, target = next(self.cal_loader.__iter__())
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                self.cal_batch = (images, target)
            images, target = self.cal_batch

            # compute output
            output = self.model(images)
            loss = self.criterion(output, target)
            return loss

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

iter = 0
def run_inference_on_batch(scales, model, mq):
    global iter

    from quantization.methods.clipped_uniform import FixedClipValueQuantization
    qwrappers = [qwrapper for (name, qwrapper) in mq.quantization_wrappers if isinstance(qwrapper, ActivationModuleWrapperPost)]
    for i, qwrapper in enumerate(qwrappers):
        if i < len(scales):
            qwrapper.set_quantization(FixedClipValueQuantization, {'clip_value': scales[i], 'device': model.device},
                                      verbose=(iter % 20 == 0))
    loss = model.evaluate_calibration()
    print("iter: {}, loss: {}".format(iter, loss.item()))
    iter += 1
    return loss.item()


def main():
    args = parser.parse_args()

    # create model
    inf_model = CnnModel(args.arch, args.custom_resnet, args.pretrained, args.dataset, args.gpu_ids, args.datapath,
                     args.batch_size, args.shuffle, args.workers, args.print_freq)

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

    # initialize scales
    init = np.array([1.]*len(layers))  # TODO: allow better initialization like mse based or aciq
    # run optimizer
    min_options = {}
    if args.maxiter is not None:
        min_options['maxiter'] = args.maxiter
    if args.maxfev is not None:
        min_options['maxfev'] = args.maxfev
    res = opt.minimize(lambda scales: run_inference_on_batch(scales, inf_model, mq), np.array(init),
                       method=args.min_method, options=min_options)
    print(res)
    scales = res.x
    # evaluate
    from quantization.methods.clipped_uniform import FixedClipValueQuantization
    qwrappers = [qwrapper for (name, qwrapper) in mq.quantization_wrappers if isinstance(qwrapper, ActivationModuleWrapperPost)]
    for i, qwrapper in enumerate(qwrappers):
        if i < len(scales):
            qwrapper.set_quantization(FixedClipValueQuantization, {'clip_value': scales[i], 'device': inf_model.device})
    acc = inf_model.validate()
    print("Val accuracy: {}".format(acc))
    # save scales


if __name__ == '__main__':
    main()
