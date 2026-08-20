import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision.transforms import transforms


from PIL import Image


#labels --> id(there are 16 classes) box_x box_y box_width box_height

#images --> 640x640
import os



images = os.listdir(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\images")
labels = os.listdir(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\labels")
print(images)
print(labels)
to_tensor = transforms.ToTensor()

class Stargazer(nn.Module):


    def __init__(self):
        super().__init__()




        self.layer1 = nn.Conv2d(3, 32, 3)
        self.actfunc1 = nn.ReLU()
        self.drop = nn.Dropout()
        self.output = nn.Linear(32*638*638, 9*21)
        self.sigmoid = nn.Sigmoid()
        self.optim = optim.Adam(self.parameters(), lr=1e-3)



    def forward(self, x):
        x = self.layer1(x)
        x = self.actfunc1(x)
        x = self.drop(x)
        x = x.flatten(start_dim=1)
        x = self.output(x)
        x = x.reshape(9, 21)
        x = self.sigmoid(x)
        return x

# 0 is objectness
# 1-4 is box_x box_y box_width box_height
# 5-21 is ID


    def train_stargazer(self, epoques: int=5):
        self.model.train()
        index = 0
        for _ in range(epoques):
            for image in images:
                image_path = os.path.join(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\images", images[index])
                labels_path = os.path.join(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\labels", labels[index])
                index = index + 1
                opened_image = Image.open(image_path).convert("RGB")
                label = []
                with open(labels_path, "r") as file:
                    for line in file:
                        label.append(torch.tensor(float(x) for x in line.split()))
                label = to_tensor(label)


                x = to_tensor(opened_image)
                object = torch.zeros((3,3))
                for l in label:
                    x_box = l[1]
                    y_box = l[2]
                    label[int(x_box*3)][int(y_box*3)] = 1

                    prediction = self.forward(x) #(9, 21)
                    self.optim.zero_grad()
                    lossObjectness, lossBox, lossID = 0
                    index2 = 0
                    for _ in range(9):
                        lossObjectness = lossObjectness + nn.BCEWithLogitsLoss()(prediction[index2][0], ) # 0 is non object, 1 is object
                        lossBox = lossBox + nn.MSELoss()(prediction[index2][1:4], l[1:4])
                        lossID = lossID + nn.CrossEntropyLoss(prediction[index2][5:21], l[0])

                    loss = lossObjectness + lossBox + lossID
                    loss.backward()


    def validate_stargazer(self):
        pass
    def test_stargazer(self):
        pass

