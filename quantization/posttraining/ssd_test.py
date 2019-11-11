import argparse
import logging
import os
import sys
from itertools import count

import numpy as np
import scipy.optimize as opt
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader, Subset

from quantization.posttraining.module_wrapper import ActivationModuleWrapperPost, ParameterModuleWrapperPost
from quantization.quantizer import ModelQuantizer

p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ssd')
print(p)
sys.path.append(p)  # dirty
from ssd.config import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import build_detection_model
from ssd.utils import dist_util
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger

from ssd.data.datasets import build_dataset
from ssd.data.transforms import build_transforms, build_target_transform
from ssd.data import samplers
from ssd.data.build import BatchCollator

# TODO: refactor this
_eval_count = count(0)
_min_loss = 1e6


def make_cal_data_loader(cfg, distributed=False, max_iter=None, start_iter=0, size=256):
    train_transform = build_transforms(cfg, is_train=False)
    target_transform = build_target_transform(cfg)
    dataset_list = cfg.DATASETS.TRAIN
    datasets = build_dataset(dataset_list, transform=train_transform, target_transform=target_transform, is_train=True)

    shuffle = True
    datasets[0] = Subset(datasets[0], np.arange(size))
    data_loaders = []

    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter,
                                                                start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(True))
        data_loaders.append(data_loader)

    assert len(data_loaders) == 1
    return data_loaders[0]


def evaluate_calibration(model, cal_set, device):
    # switch to evaluate mode
    model.eval()
    model.get_loss = True

    with torch.no_grad():
        res = 0.
        for i, (images, target, _) in enumerate(cal_set):
            images, target = images.to(device), target.to(device)
            # compute output
            loss_dict = model(images, target)
            loss = loss_dict['reg_loss']+25*loss_dict['cls_loss']
            print(loss_dict)
            res += loss.item()

    model.get_loss = False
    return res / len(cal_set)


def evaluate_calibration_clipped(scales, model, mq, cal_set, device):
    global _eval_count, _min_loss
    eval_count = next(_eval_count)

    mq.set_clipping(scales, device)
    loss = evaluate_calibration(model, cal_set, device)

    if loss < _min_loss:
        _min_loss = loss

    print_freq = 20
    if eval_count % 20 == 0:
        print("func eval iteration: {}, minimum loss of last {} iterations: {:.4f}".format(
            eval_count, print_freq, _min_loss))

    return loss


def evaluation(cfg, ckpt, distributed, args):
    logger = logging.getLogger("SSD.inference")

    model = build_detection_model(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)

    if True:
        all_convs = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        all_relu = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
        all_relu6 = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU6)]
        layers = all_relu[1:-1] + all_relu6[1:-1] + all_convs[1:-1]
        replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                               nn.ReLU6: ActivationModuleWrapperPost,
                               nn.Conv2d: ParameterModuleWrapperPost}

        mq = ModelQuantizer(model, args, layers, replacement_factory)

    # evaluate
    do_evaluation(cfg, model, distributed)
    init = mq.get_clipping()

    _iter = count(0)

    def local_search_callback(x):
        it = next(_iter)
        mq.set_clipping(x, device)
        # loss = inf_model.evaluate_calibration()
        # print("\n[{}]: Local search callback".format(it))
        # print("loss: {:.4f}\n".format(loss.item()))
        print(x)

        # evaluate
        # acc = inf_model.validate()

    print(cfg)
    cal_set = make_cal_data_loader(cfg, size=cfg.TEST.BATCH_SIZE)
    res = opt.minimize(lambda scales: evaluate_calibration_clipped(scales, model, mq, cal_set, device),
                       init.cpu().numpy(),
                       method='Powell', options={'maxiter': 2}, callback=local_search_callback)

    print(res)
    scales = res.x
    mq.set_clipping(scales, device)

    # evaluate
    do_evaluation(cfg, model, distributed)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument("--output_dir", default="eval_results", type=str,
                        help="The directory to store evaluation results.")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--bit_weights', '-bw', type=int, help='Number of bits for weights', default=None)
    parser.add_argument('--bit_act', '-ba', type=int, help='Number of bits for activations', default=None)
    parser.add_argument('--pre_relu', dest='pre_relu', action='store_true', help='use pre-ReLU quantization')
    parser.add_argument('--qtype', default='l2_norm', help='Type of quantization method')

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    evaluation(cfg, ckpt=args.ckpt, distributed=distributed, args=args)


if __name__ == '__main__':
    main()
