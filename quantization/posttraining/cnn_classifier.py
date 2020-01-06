import os, sys, time, random
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
from models.resnet import resnet as custom_resnet
from models.inception import inception_v3 as custom_inception
from utils.misc import normalize_module_name, arch2depth


class CnnModel(object):
    def __init__(self, arch, use_custom_resnet, use_custom_inception, pretrained, dataset, gpu_ids, datapath, batch_size, shuffle, workers,
                 print_freq, cal_batch_size, cal_set_size, args):
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
        self.cal_set_size = cal_set_size  # TODO: pass it as cmd line argument

        # create model
        if 'resnet' in arch and use_custom_resnet:
            model = custom_resnet(arch=arch, pretrained=pretrained, depth=arch2depth(arch),
                                  dataset=dataset)
        elif 'inception_v3' in arch and use_custom_inception:
            model = custom_inception(pretrained=pretrained)
        else:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=pretrained)

        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

        torch.cuda.set_device(gpu_ids[0])
        model = model.to(self.device)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, self.device)
                args.start_epoch = checkpoint['epoch']
                checkpoint['state_dict'] = {normalize_module_name(k): v for k, v in checkpoint['state_dict'].items()}
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        if len(gpu_ids) > 1:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, gpu_ids)
            else:
                model = torch.nn.DataParallel(model, gpu_ids)

        self.model = model

        if args.bn_folding:
            print("Applying batch-norm folding ahead of post-training quantization")
            from utils.absorb_bn import search_absorbe_bn
            search_absorbe_bn(model)

        # define loss function (criterion) and optimizer
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        val_data = get_dataset(dataset, 'val', get_transform(dataset, augment=False, scale_size=299 if 'inception' in arch else None,
                               input_size=299 if 'inception' in arch else None),
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
                    if i * self.cal_batch_size >= self.cal_set_size:
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
