# EmotionNet
## Emotional detection with face landmark detection
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge  
This is current work on final project in Application Programming.
To verify emotion network.
- emotion.py : file with train mode, service mode(real time), image_test mode
- train.py : define training on Network
- data.py : dataset and dataloader used in Network
---
  
1. This is the model concatenate Dlib model with emotion detection model
2. Training and Validation data used FER-2013
3. ADAM Optimizer

---
## Follow Instructions to run program

1. If you have virtual env like conda, you can use it as virtual environment
2. Required package, module
    - pytorch == 1.9.1
    - numpy == 1.21.2
    - opencv-python
    - dlib(Probably you need to install cmake first)
    - torchsummary
    - imutils
    - ...

___
## clone github repository with your own device.
```
    git clone https://github.com/junia3/EmotionNet.git
```

___
## Download dataset from drive
https://drive.google.com/file/d/1UB6LmoIsE-V0FnFICI5WDEj5ZJBkogem/view?usp=sharing

Extract this file into repository, It looks like this.

> emotion_net
>    > data
>    >    - train
>    >    - test
>    >    - data.py

___
## Download required packages
1. pip install pytorch
    - Or you can use conda install pytorch instead
2. pip install cmake
3. pip install opencv-python
4. pip install torchvision
5. pip install imutils
6. pip install torchsummary
7. pip install dlib
8. ... 이후에도 설치할 게 좀 있음

If there is any error message with running code, you can install it in your virtual env with "pip + install + packagename"

___
## Run training mode
```
    python emotion.py --mode train
```

___
## Logging example code

```python
def log_progress(epoch, num_epoch, iteration, num_data, batch_size, loss, acc):
    progress = int(iteration / (num_data // batch_size) * 100 // 4)
    print("Epoch : %d/%d >>>> train : %d/%d(%.2f%%) ( " % (epoch, num_epoch, iteration, num_data // batch_size, iteration / (num_data // batch_size) * 100)
          + '=' * progress + '>' + ' ' * (25 - progress) + " ) loss : %.6f, accuracy : %.2f%%" % (loss, acc * 100), end='\r')
```

___
## Simpler model you can use

Model structure is imple, but validation/test accuracy not good.

```python
import torch
import torch.nn as nn

class EmotionNet(nn.Module):
    def __init__(self, num_classes = 7, init_weights = True):
        super(EmotionNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 1,
                                            out_channels = 32,
                                            kernel_size = 3,
                                            stride = 1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 32,
                                            out_channels = 64,
                                            kernel_size = 3,
                                            stride = 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2),
                                    nn.Dropout(0.25))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 64,
                                              out_channels = 128,
                                              kernel_size = 3,
                                              stride = 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride = 2),
                                    nn.Conv2d(in_channels = 128,
                                              out_channels = 128,
                                              kernel_size = 3,
                                              stride = 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2),
                                    nn.Dropout(0.25))

        self.linear = nn.Sequential(nn.Linear(in_features = 4*4*128, out_features = 1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(in_features = 1024, out_features = num_classes))

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
        output = torch.flatten(output, 1)
        output = self.linear(output)
        return output
```

- **Use 64 batches per each training epoch**
- **Use 50 epochs to train**
___
### Result of Simpler model
![result](./result_cnn.png)
