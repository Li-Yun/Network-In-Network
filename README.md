# Network-In-Network

This repo is a simple implementation in Pytorch for the Network-In-Network network architecture. The testing benchmark is CIFAR-10, and the top-1 classification accuracy achieves between 87% and 89% by only leveraging data argumentation and input normalization to -1 and 1.

# Usage
The scripts have been tested on anaconda3.

For training a model:
```
python main.py --max-ep=1000 --lr=0.1
```

For evaluation:
```
python main.py --training_testing=evaluation
```
