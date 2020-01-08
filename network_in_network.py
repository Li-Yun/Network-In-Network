import torch.nn as nn

class NetworkInNetwork(nn.Module):
    # create all layers in Network-in-Network
    def __init__(self):
        super(NetworkInNetwork, self).__init__()
        # first block layer
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size = 5, padding = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 160, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(160, 96, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
            nn.Dropout(0.5)
        )
        # second block layer
        self.block_2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size = 5, padding = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(3, stride = 2, ceil_mode = True),
            nn.Dropout(0.5)
        )
        # thrid block layer
        self.block_3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 10, 1),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(8, stride = 1)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        #reshape the output tensor to (x.size(0), 10)
        x = x.view(x.size(0), 10)
        return x
