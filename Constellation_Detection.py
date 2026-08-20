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

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),  # 320
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 160
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 80
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 40
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # 20
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),  # 10
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),  # 5
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(),  # 3
        )
        self.drop = nn.Dropout()
        self.output = nn.Conv2d(256, 21, kernel_size=1)
        self.optim = optim.Adam(self.parameters(), lr=1e-3)



    def forward(self, x):
        x = self.layer1(x)
        x = self.drop(x)
        x = self.output(x)
        x = x.permute(1, 2, 0)  # (3, 3, 21) -> H, W, C
        x = x.reshape(9, 21)  # row-major: index = row*3 + col, matches object.reshape(9)
        return x

# 0 is objectness
# 1-4 is box_x box_y box_width box_height
# 5-21 is ID


    def train_stargazer(self, epochs: int=5):
        self.train()

        for _ in range(epochs):
            index = 0

            for image in images:
                image_path = os.path.join(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\images", images[index])
                labels_path = os.path.join(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\labels", labels[index])
                index = index + 1
                opened_image = Image.open(image_path).convert("RGB")
                label = []
                with open(labels_path, "r") as file:
                    for line in file:
                        label.append(torch.tensor([float(x) for x in line.split()]))
                label = torch.stack(label)


                x = to_tensor(opened_image)
                object = torch.zeros((3,3))
                for l in label:
                    x_box = l[1]
                    y_box = l[2]
                    object[int(y_box*3)][int(x_box*3)] = 1

                object = object.reshape(9)
                lossObjectness, lossBox, lossID = 0, 0, 0

                index2 = 0
                # objectness
                for _ in range(9):
                    lossObjectness = lossObjectness + nn.BCEWithLogitsLoss()(prediction[index2][0], object[
                        index2])  # 0 is non object, 1 is object
                    index2 = index2 + 1

                index2 = 0
                for l in label:
                    prediction = self.forward(x) #(9, 21)
                    self.optim.zero_grad()






                    x_box = l[1]
                    y_box = l[2]

                    row = int(y_box * 3)
                    col = int(x_box * 3)

                    index2 = row * 3 + col

                    lossBox = lossBox + nn.MSELoss()(
                        torch.sigmoid(prediction[index2][1:5]),
                        l[1:5]
                    )

                    lossID = lossID + nn.CrossEntropyLoss()(
                        prediction[index2][5:21],
                        l[0].long()
                    )



                loss = lossObjectness + lossBox + lossID
                loss.backward()
                self.optim.step()
            index = 0

    def validate_stargazer(self):
        pass

    def test_stargazer(self):
        self.eval()

        count = 0

        with torch.no_grad():
            for index in range(len(images)):

                image_path = os.path.join(
                    r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\images",
                    images[index]
                )

                labels_path = os.path.join(
                    r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\labels",
                    labels[index]
                )

                opened_image = Image.open(image_path).convert("RGB")
                x = to_tensor(opened_image)

                prediction = self.forward(x)  # (9, 21)

                with open(labels_path, "r") as file:
                    for line in file:

                        actual = torch.tensor(
                            [float(v) for v in line.split()]
                        )

                        # actual:
                        # [class, x, y, width, height]

                        x_box = actual[1]
                        y_box = actual[2]

                        row = int(y_box * 3)
                        col = int(x_box * 3)

                        cell = row * 3 + col

                        # prediction from the correct grid cell
                        pred_objectness = torch.sigmoid(
                            prediction[cell][0]
                        ).item()

                        pred_box = torch.sigmoid(
                            prediction[cell][1:5]
                        )

                        pred_class = torch.argmax(
                            prediction[cell][5:21]
                        ).item()

                        print(f"\nExample {count + 1}")
                        print("PREDICTED:")
                        print("  class:", pred_class)
                        print("  box:", pred_box.tolist())
                        print("  objectness:", pred_objectness)

                        print("ACTUAL:")
                        print("  class:", int(actual[0].item()))
                        print("  box:", actual[1:5].tolist())
                        print("  objectness: 1")

                        count += 1

                        if count == 10:
                            return

model = Stargazer()

model.train_stargazer(epochs=1)
model.test_stargazer()