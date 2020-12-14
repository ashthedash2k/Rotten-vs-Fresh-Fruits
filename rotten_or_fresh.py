import torch
from torchvision import datasets, models, transforms, utils
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


fruit_train = '/Users/ashley/Deeplearning/fresh_and_rotton/dataset/train'
fruit_test = '/Users/ashley/Deeplearning/fresh_and_rotton/dataset/test'
data_dir = "/Users/ashley/Deeplearning/fresh_and_rotton/dataset"


data_transform = {'train':transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),

                  'test':transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ]) }



image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['train', 'test']}


data_loader = {x:torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=124, num_workers=0) for x in ['train', 'test']}


class_names = image_datasets['train'].classes


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


inputs, classes = next(iter(data_loader['train']))


out = utils.make_grid(inputs)


imshow(out, title=[class_names[x] for x in classes])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8*56*56, 56) #256
        self.fc2 = nn.Linear(56, 6)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = F.max_pool2d(self.relu(self.conv1(x)), 2)
        x = F.max_pool2d(self.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x)) #softmax
        x = self.fc2(x)
        return x

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.0001)
cross_el = nn.CrossEntropyLoss()

EPOCHS = 4


for epoch in range(EPOCHS):
    net.train()
    for data in data_loader['train']:
        x, y = data
        net.zero_grad()
        output = net(x)
        loss = cross_el(output, y)
        loss.backward()
        optimizer.step()


correct = 0
total = 0


with torch.no_grad():
    for data in data_loader['test']:
        x, y = data
        output = net(x)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct +=1
            total +=1
print(f'accuracy: {round(correct/total, 3)}')
