import os
import glob
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

from network_in_network import NetworkInNetwork
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelEvaluation():
    def __init__(self, model_dir_path):
        self.model_path = glob.glob(os.path.join(model_dir_path, "*.pt"))[0]
        self.nin_model = NetworkInNetwork()
        self.nin_model.to(device)

    def load_model(self):
        self.nin_model.load_state_dict(torch.load(self.model_path))
        self.nin_model.eval()

    def predict(self, testloader):
        # load a pre-trained model
        self.load_model()

        # evaluate new images
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                input_images, labels = Variable(data[0].to(device)), Variable(data[1].to(device))
                outputs = self.nin_model(input_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if predicted.item() == labels.item():
                    correct += 1
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

