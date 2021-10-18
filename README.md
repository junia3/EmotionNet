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
