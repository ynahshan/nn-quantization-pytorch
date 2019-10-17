from subprocess import run
import mlflow
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--act_or_weights', '-aow', help='Quantize activations or weights [a, w]', default='a')
parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
args = parser.parse_args()

models_set = [
    # {'model': 'vgg16', 'bs': 128, 'dev': [5]},
    # {'model': 'vgg16_bn', 'bs': 128, 'dev': [5]},
    # {'model': 'inception_v3', 'bs': 256, 'dev': [5]},
    # {'model': 'mobilenet_v2', 'bs': 128, 'dev': [5]},
    {'model': 'resnet18', 'bs': 64, 'dev': [0]},
    # {'model': 'resnet50', 'bs': 128, 'dev': [5]},
    # {'model': 'resnet101', 'bs': 512, 'dev': [5]}
]

layer_type = '-ba' if args.act_or_weights == 'a' else '-bw'
for mset in models_set:
    for bits in [2, 3, 4, 5, 6, 7, 8]:
        run(["python", "quantization/analysis/separability_index.py"] + ['-a', mset['model']] + ['-b', str(mset['bs'])]
            + ['--dataset', 'imagenet'] + ['--gpu_ids'] + " ".join(map(str, mset['dev'])).split(" ")
            + "--pretrained --custom_resnet".split(" ") + ['-exp', args.experiment]
            + [layer_type, str(bits)] + "-i 5 -n 100".split(" ")
            )
