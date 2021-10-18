import torch
import torch.nn as nn

class EmotionNet(nn.Module):
    def __init__(self, num_classes = 7, init_weights = True):
        super(EmotionNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 1,
                                            out_channels = 64,
                                            kernel_size = (3, 1),
                                            padding = 'same',
                                            stride = 1),
                                    nn.Conv2d(in_channels = 64,
                                            out_channels = 64,
                                            kernel_size = (1, 3),
                                            padding = 'same',
                                            stride = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2),
                                    nn.Dropout(0.25))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 64,
                                              out_channels = 128,
                                              kernel_size = (3, 1),
                                              padding = 'same',
                                              stride = 1),
                                    nn.Conv2d(in_channels = 128,
                                              out_channels = 128,
                                              kernel_size = (1, 3),
                                              padding = 'same',
                                              stride = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2),
                                    nn.Dropout(0.25))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 128,
                                              out_channels = 256,
                                              kernel_size = (3, 1),
                                              padding = 'same',
                                              stride = 1),
                                    nn.Conv2d(in_channels = 256,
                                              out_channels = 256,
                                              kernel_size = (1, 3),
                                              padding = 'same',
                                              stride = 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2),
                                    nn.Dropout(0.25))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 256,
                                              out_channels = 512,
                                              kernel_size = (3, 1),
                                              padding = 'same',
                                              stride = 1),
                                    nn.Conv2d(in_channels = 512,
                                              out_channels = 512,
                                              kernel_size = (1, 3),
                                              padding = 'same',
                                              stride = 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2),
                                    nn.Dropout(0.25))

        self.linear = nn.Sequential(nn.Linear(in_features = 3*3*512, out_features = 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Dropout(0.25),
                                    nn.Linear(in_features = 512, out_features = 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Dropout(0.25),
                                    nn.Linear(in_features = 256, out_features = num_classes))

        if init_weights:
            self._initailize_weight()

    def _initailize_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = 'fan_out', nonlinearity = 'relu')
                if module.bias is None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = torch.flatten(output, 1)
        output = self.linear(output)
        return output