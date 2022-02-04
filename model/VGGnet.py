# import package : model
import torch
from torch import nn
from torch.nn import functional as F

# VGG type dict
# int : output channels after conv layer with kernel (3,3)
# _int : output channels after conv layer with kernel (1,1)
# 'M' : max pooling layer
VGG_types = {
    'A' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M',512, 512, 'M'],
    'A-LRN' : [64, 'LRN', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C' : [64, 64, 'M', 128, 128, 'M', 256, 256, '_256', 'M', 512, 512, '_512', 'M', 512, 512, '_512', 'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

# define VGGnet class
class VGGnet(nn.Module):
    def __init__(self, model, in_channels=3, num_classes=10, init_weights=True):
        super(VGGnet,self).__init__()
        self.in_channels = in_channels

        # create conv_layers corresponding to VGG type
        self.conv_layers = self.create_conv_laters(VGG_types[model])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.fcs(x)
        return x
    
    # define a function to create conv layer taken the key of VGG_type dict 
    def create_conv_laters(self, architecture):
        layers = []
        in_channels = self.in_channels # 3

        for x in architecture:
            if type(x) == int: # int means conv layer
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == '_256':
                layers += [nn.Conv2d(in_channels=256, out_channels=256,
                                     kernel_size=(1,1), stride=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
            elif x == '_512':
                layers += [nn.Conv2d(in_channels=512, out_channels=512,
                                     kernel_size=(1,1), stride=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            elif x == 'LRN':
                # parameters of LRN layer are those of (Krizhevsky et al., 2012)
                layers += [nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)]
        
        return nn.Sequential(*layers)