import numpy as np
import os
import argparse
import cv2
from model.model import EmotionNet
from data.data import EmotionDataset
import dlib
from imutils import face_utils
import torch
from torchsummary import summary as summary_
from torch.utils.data import DataLoader
from tool.train import *

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Your current device is", device)
model = EmotionNet().to(device)


if mode == 'image_test':
    state_dict = torch.load(os.path.join(os.getcwd(), 'output/final_state.pth'), map_location=device)
    model.load_state_dict(state_dict)
    emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}
    image = cv2.imread('emotion.jpg', cv2.IMREAD_GRAYSCALE)

    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    faces = detector(image)
    for face in faces:
        shape = predictor(image, face)
        shape = face_utils.shape_to_np(shape)
        x1 = face.left()
        y1 = face.bottom()
        x2 = face.right()
        y2 = face.top()

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
        try:
            check_area = torch.Tensor(cv2.resize(image[x1:x2, y2:y1], (48, 48))).unsqueeze(0).unsqueeze(0)
            _, emotion = model(check_area).max(1)
            cv2.putText(image, emotion_dict[int(emotion)], (x1, y2 - 20), fontFace=4, fontScale=1, color=(26, 183, 125))
        except:
            pass

        for (x, y) in shape:
            cv2.line(image, (x, y), (x, y), (0, 0, 255), 4)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == 'service':
    state_dict = torch.load(os.path.join(os.getcwd(), 'output/final_state.pth'), map_location=device)
    model.load_state_dict(state_dict)
    emotion_dict = {0 : "angry", 1 : "disgusted", 2 : "fearful", 3 : "happy", 4 : "neutral", 5 : "sad", 6 : "surprised"}
    cap = cv2.VideoCapture(0)
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    emotion, update, update_period = 0, 0, 25
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            x1 = face.left()
            y1 = face.bottom()
            x2 = face.right()
            y2 = face.top()

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)

            if update == update_period:
                try:
                    check_area = torch.Tensor(cv2.resize(gray[x1:x2, y2:y1], (48, 48))).unsqueeze(0).unsqueeze(0)
                    _, emotion = model(check_area).max(1)
                except:
                    pass
                update = 0

            else:
                update += 1

            cv2.putText(frame, emotion_dict[int(emotion)], (x1, y2 - 20), fontFace=4, fontScale= 1, color = (26, 183, 125))
            for (x, y) in shape:
                cv2.line(frame, (x, y), (x, y), (0, 0, 255), 4)
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break


    cv2.destroyAllWindows()
    cap.release()

elif mode == 'train':
    batch_size = 64
    num_epoch = 30

    train_dataset = EmotionDataset('train', True, True)
    val_dataset = EmotionDataset('test', True, True)

    args = {'num_train': len(train_dataset), 'num_val' : len(val_dataset), 'batch_size' : batch_size, 'num_epoch' : num_epoch}

    print("Train dataset ready : %d"%len(train_dataset))
    print("Validation dataset ready : %d"%len(val_dataset))

    summary_(model, (1, 48, 48), batch_size = batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    train(num_epoch, train_dataloader, val_dataloader, model, args, device)