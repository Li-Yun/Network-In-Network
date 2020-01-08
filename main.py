import os
import sys
import shutil
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from train_save_model import ModelTrainingSaving
from testing import ModelEvaluation

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', dest = 'dataset_directory', default = 'data')
parser.add_argument('--model_dir', dest = 'model_directory', default = 'nin_model')
parser.add_argument('--training_testing', dest = 'model_training_or_testing', default = 'training')
parser.add_argument('--bs', dest = 'batch_size', default = 128)
parser.add_argument('--max_ep', dest = 'maximum_epoch', default = 120000, type = int)
parser.add_argument('--lr', dest = 'learning_rate', default = 0.1, type = float)
parser.add_argument('--mt', dest = 'momentum', default = 0.9, type = float)
parser.add_argument('--wd', dest = 'weight_decay', default = 0.0001, type = float)
args = parser.parse_args()

full_dataset_dir_path = os.path.abspath(args.dataset_directory)
full_model_dir_path = os.path.abspath(args.model_directory)
BATCH_SIZE = args.batch_size

try:
    shutil.rmtree(full_dataset_dir_path)
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))

if not os.path.isdir(full_model_dir_path):
    os.mkdir(full_model_dir_path)

def downloading_dataset():
    # transformation: data agumentation (RandomHorizontalFlip and )
    # normalize the range of inputs to -1 and 1
    normalization = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
            ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalization
            ])

    print ('Download the training set...')
    trainset = torchvision.datasets.CIFAR10(root = full_dataset_dir_path, train = True,
                                            download = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE,
                                              shuffle = True, num_workers = 2)
    print ('Download the testing set...')
    testset = torchvision.datasets.CIFAR10(root = full_dataset_dir_path, train = False,
                                       download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1,
                                         shuffle = False, num_workers = 2)
    return trainloader, testloader

def main():
    # download cifar10 dataset
    trainloader, testloader = downloading_dataset()

    # network training and saving
    if args.model_training_or_testing == 'training':
        trainer_saver = ModelTrainingSaving(args.maximum_epoch, args.learning_rate, args.momentum, args.weight_decay)
        trainer_saver.fit(trainloader)
        trainer_saver.save(full_model_dir_path)
    elif args.model_training_or_testing == 'evaluation':
        # evaluate the model
        ModelEvaluation(full_model_dir_path).predict(testloader)

if __name__ == "__main__":
    main()
