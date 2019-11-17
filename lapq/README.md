# Loss Aware Post-training Quantization

## Dependencies
- python3.x
- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization
- [scikit-learn](https://scikit-learn.org) for kmeans clustering
- [mlflow](https://mlflow.org/) for logging
- [tqdm](https://tqdm.github.io/) for progress
- [scipy](https://scipy.org/) for Powell and Brent


## HW requirements
NVIDIA GPU / cuda support

## Data
- To run this code you need validation set from ILSVRC2012 data
- Configure your dataset path by providing --data "PATH_TO_ILSVRC" or copy ILSVRC dir to ~/datasets/ILSVRC2012.
- To get the ILSVRC2012 data, you should register on their site for access: <http://www.image-net.org/>

## Prepare environment
- Clone source code
```
git clone https://github.com/ynahshan/nn-quantization-pytorch.git
cd cnn-quantization
```
- Create virtual environment for python3 and activate:
```
virtualenv --system-site-packages -p python3 venv3
. ./venv3/bin/activate
```
- Install dependencies
```
pip install torch torchvision bokeh pandas sklearn mlflow tqdm scipy
```

### Run experiments
- To reproduce resnet18 experiment run:
```
cd nn-quantization-pytorch
python lapq/layer_scale_optimization_opt.py -a resnet18 --dataset imagenet -b 256 --pretrained --custom_resnet -ba 2 --min_method Powell -maxi 2 -exp temp -cs 512
```

- To reproduce resnet50 experiment run:
```
cd nn-quantization-pytorch
python lapq/layer_scale_optimization_opt.py -a resnet50 --dataset imagenet -b 128 --pretrained --custom_resnet -ba 2 --min_method Powell -maxi 1 -exp temp -cs 512
```

To reproduce results for other models change model name after "-a". All other arguments are same as resnet50.
