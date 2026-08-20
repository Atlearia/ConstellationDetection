import torch
import torch.nn as nn
from torchtyping import TensorType
from torchvision.io import read_image
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import os
from PIL import Image
import torch.optim

all_images = []

dataset = MNIST(root="images", train=True, download=True)

print(dataset.data.shape, dataset.targets.shape) #torch.Size([60000, 28, 28]) torch.Size([60000])

x = dataset.data[:50].float() / 255
y = dataset.targets[:50]
x = x.flatten(start_dim=1)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 512 )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.lin2 = nn.Linear(512, 10)
        self.sig = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        x = self.lin(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.sig(x)
        return x

    def trainingg(self, x):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        for _ in range(200):
            output = self(x)
            lossfunc = nn.CrossEntropyLoss()
            loss = lossfunc(output, y)
            self.zero_grad()
            loss.backward()
            optimizer.step()

    def judge(self):
        model.eval()

        with torch.no_grad():
            output = model(x[:50])
            predictions = output.argmax(dim=1)

        print("Predicted:", predictions)
        print("Actual:   ", y[:50])



model = Model()
model.trainingg(x)
model.judge()
