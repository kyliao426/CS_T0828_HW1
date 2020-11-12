# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:31:00 2020

@author: 冠宇
"""
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms


class HW1TestDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.transforms = transforms
        self.root_dir = Path(root_dir)
        self.input = []
        self.filename = []
        for i, file_path in enumerate(self.root_dir.glob('*')):
            self.input.append(file_path)
            self.filename.append(file_path.name)

    def __getitem__(self, index):
        image = Image.open(self.input[index]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)

        return image, self.filename[index]

    def __len__(self):
        return len(self.input)


class HW1TrainDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.input = file
        self.filename = number.tolist()

    def __getitem__(self, index):
        image = Image.open(self.input[index]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
            target2 = self.filename[index]
        return image, target2

    def __len__(self):
        return len(self.input)


def train(train_dataset, load):
    train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=10,
                shuffle=True,
                )

    if load:
        model = torch.load('HW1_ResNext101')
    else:
        model = models.resnext101_32x8d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 196)

    model = model.cuda()
    model.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.001,
                                momentum=0.9)

    for epoch in range(2):
        print('Epoch: {:d}'.format(epoch+1))
        print('-' * len('Epoch: {:d}'.format(epoch+1)))
        train_loss = 0.0
        train_corrects = 0

        for step, (image, label) in enumerate(train_loader):
            image, label = image.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(image)
            loss = loss_func(output, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * image.size(0)
            _, preds = torch.max(output.data, 1)
            train_corrects += torch.sum(preds == label.data)

        train_loss = train_loss / len(train_dataset)
        train_acc = train_corrects.double() / len(train_dataset)
        print('Training loss: {:.4f} accuracy: {:.4f}   '
              .format(train_loss, train_acc))
        if epoch == 0:
            best_loss = train_loss
            torch.save(model, 'HW1_ResNext101')
        elif train_loss < best_loss:
            best_loss = train_loss
            torch.save(model, 'HW1_ResNext101')


def test(test_path, classes):
    model = torch.load('HW1_ResNext101')
    model = model.cuda()
    model.eval()

    test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                )

    table = [['id', 'label']]
    with torch.no_grad():
        for image, label in test_loader:
            image = image.cuda()
            output = model(image)
            _, prediction = torch.max(output.data, 1)
            table.append([label[0][:-4], classes[prediction.item()]])

    with open('HW1_test_result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)


if __name__ == '__main__':
    csv_path = 'training_labels.csv'
    test_path = 'dataset/testing_data'

    train_label = pd.read_csv(csv_path)

    car = train_label['label']
    car = car.to_numpy()
    car_list = list(set(car))
    car_dic = {}
    for i in range(len(car_list)):
        car_dic[car_list[i]] = i

    file = train_label['id']
    file = [str('%06d' % int(i)) + ".jpg" for i in file]
    file = ['./dataset/training_data/' + str(i) for i in file]

    brand = train_label['label'].to_numpy()
    number = []
    for i in range(len(brand)):
        number.append(car_dic[brand[i]])
    number = np.array(number)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    load = False

    train_dataset = HW1TrainDataset(data_transforms['train'])
    train(train_dataset, load)
    test_dataset = HW1TestDataset(test_path, data_transforms['test'])
    test(test_path, car_list)

    torch.cuda.empty_cache()
