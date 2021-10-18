import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class EmotionDataset(Dataset):
    def __init__(self, mode = 'train', resize = True, rescale = True):

        if mode not in ['train', 'test']:
            raise ValueError("mode should be one of train or test")

        self.dir = 'data/' + mode + '/'
        self.rescale = rescale
        self.resize = resize
        self.emotion_dict = {"angry" : 0,"disgusted" : 1,"fearful" : 2,"happy" : 3,"neutral" : 4,"sad" : 5,"surprised" : 6}
        self.file_paths = []
        for emotion in list(self.emotion_dict.keys()):
            file_path = os.path.join(os.getcwd(), self.dir + emotion)
            for root, dirs, files in os.walk(file_path):
                for filename in files:
                    filename = os.path.join(root, filename)
                    self.file_paths.append((filename, self.emotion_dict[emotion]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path, data_label = self.file_paths[idx]

        data = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)

        if self.resize:
            data = cv2.resize(data, (48, 48))

        if self.rescale:
            data = data/255.0

        return data, data_label


class ResNetDataset(Dataset):
    def __init__(self, mode = 'train', resize = True, rescale = True):

        if mode not in ['train', 'test']:
            raise ValueError("mode should be one of train or test")

        self.dir = 'data/' + mode + '/'
        self.rescale = rescale
        self.resize = resize
        self.emotion_dict = {"angry" : 0,"disgusted" : 1,"fearful" : 2,"happy" : 3,"neutral" : 4,"sad" : 5,"surprised" : 6}
        self.file_paths = []
        for emotion in list(self.emotion_dict.keys()):
            file_path = os.path.join(os.getcwd(), self.dir + emotion)
            for root, dirs, files in os.walk(file_path):
                for filename in files:
                    filename = os.path.join(root, filename)
                    self.file_paths.append((filename, self.emotion_dict[emotion]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path, data_label = self.file_paths[idx]

        data = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)

        if self.resize:
            data = cv2.resize(data, (48, 48))

        if self.rescale:
            data = data/255.0

        return data, data_label