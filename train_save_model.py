import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

from network_in_network import NetworkInNetwork
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelTrainingSaving():
    def __init__(self, max_epoch, lr, mt, wd):
        # use GPU or CPU
        self.nin = NetworkInNetwork()
        self.nin.to(device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.nin.parameters(), lr = lr, momentum = mt, weight_decay = wd)
        self.max_epoch = max_epoch

    def save(self, model_dir):
        # save the model
        entire_path = os.path.join(model_dir, 'nin_model.pt')
        torch.save(self.nin.state_dict(), entire_path)

    def adjust_lr(self, epoch):
        if (epoch + 1) % 30 == 0:
            for param in self.optimizer.param_groups:
                param['lr'] /= 10

    def fit(self, trainloader):
        self.nin.train()
        # network training
        for epoch in range(self.max_epoch):
            # adjust learning rate
            self.adjust_lr(epoch)

            # mini-batch
            for idx, data in enumerate(trainloader):
                # get examples and labels
                input_images, labels = Variable(data[0].to(device)), Variable(data[1].to(device))

                # zero out the parameter gradients
                self.optimizer.zero_grad()

                # foreard, backward, and optimization
                outputs = self.nin(input_images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if idx % 50 == 0:
                    print ('[ epoch %d, batch %5d/%5d ] loss: %.8f' % 
                            (epoch + 1, (idx + 1) * len(data[0]), 
                             len(trainloader.dataset), loss.item()))

        print ('Training has been done !!')

