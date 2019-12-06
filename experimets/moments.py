from subprocess import run
import mlflow
import numpy as np

models_set = [
    # {'model': 'vgg16', 'bs': 128, 'dev': [5]},
    # {'model': 'vgg16_bn', 'bs': 128, 'dev': [5]},
    # {'model': 'inception_v3', 'bs': 256, 'dev': [5]},
    # {'model': 'mobilenet_v2', 'bs': 128, 'dev': [5]},
    {'model': 'resnet18', 'bs': 256, 'dev': [0]},
    # {'model': 'resnet50', 'bs': 128, 'dev': [5]},
    # {'model': 'resnet101', 'bs': 512, 'dev': [5]}
]

exp_name = 'moments'
qtypes = ['max_static']
# qtypes = ['l2_norm', 'aciq_laplace', 'l3_norm', 'max_static']

for mset in models_set:
    for qt in qtypes:
        for bits in [16, 8, 6, 5, 4]:
            run(["python", "quantization/qat/cnn_classifier_train.py"] + ['-a', mset['model']] + ['-b', str(mset['bs'])]
                + ['--dataset', 'imagenet'] + ['--gpu_ids'] + " ".join(map(str, mset['dev'])).split(" ")
                + "--custom_resnet".split(" ") + ['-exp', exp_name]
                + ['-bw', str(bits)] + ['--qtype', qt] + "-q -e -ls".split(" ") + ['--resume', '/data/home/cvds_lab/mxt-sim/ckpt/res18_kurtosis/epoch_90_checkpoint.pth.tar']
                )

            # run(["python", "quantization/qat/cnn_classifier_train.py"] + ['-a', mset['model']] + ['-b', str(mset['bs'])]
            #     + ['--dataset', 'imagenet'] + ['--gpu_ids'] + " ".join(map(str, mset['dev'])).split(" ")
            #     + "--pretrained --custom_resnet".split(" ") + ['-exp', exp_name]
            #     + ['-bw', str(bits)] + ['--qtype', qt] + "-q -e -ls".split(" ")
            #     )
