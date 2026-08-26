import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision.transforms import transforms


from PIL import Image

import random

#labels --> id(there are 16 classes) box_x box_y box_width box_height

#images --> 640x640
import os

from ultralytics import YOLO

CLASS_NAMES = {
    0: "Aquila", 1: "Bootes", 2: "Canis Major", 3: "Canis Minor",
    4: "Cassiopeia", 5: "Cygnus", 6: "Gemini", 7: "Leo",
    8: "Lyra", 9: "Moon", 10: "Orion", 11: "Pleiades",
    12: "Sagittarius", 13: "Scorpius", 14: "Taurus", 15: "Ursa Major",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

images = os.listdir(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\images")
labels = os.listdir(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\labels")

to_tensor = transforms.ToTensor()

random.seed(67)
indices = list(range(len(images)))
random.shuffle(indices)
split_point = int(len(indices) * 0.7)
split_point2 = int(len(indices) * 0.9)
train_indices = indices[:split_point]
val_indices = indices[split_point:split_point2]
test_indices = indices[split_point2:]

print(f"train: {len(train_indices)} and validate: {len(val_indices)}")



class Stargazer(nn.Module):


    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),  # (16, 320, 320)
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # (32, 160, 160)
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 80
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 40
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # 20
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),  # 10
            nn.Conv2d(128, 256, 3, stride=2, padding=2), nn.ReLU(),  # CHANGED: 10 -> 6
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(),  # CHANGED: 6 -> 6
        )
        self.drop = nn.Dropout()
        self.output = nn.Conv2d(256, 21, kernel_size=1)
        self.optim = optim.Adam(self.parameters(), lr=1e-3)



    def forward(self, x):
        x = self.layer1(x)
        x = self.drop(x)
        x = self.output(x)
        x = x.permute(1, 2, 0)  # (3, 3, 21) -> H, W, C
        x = x.reshape(36, 21)  # row-major: index = row*3 + col, matches object.reshape(9)
        return x

# 0 is objectness
# 1-4 is box_x box_y box_width box_height
# 5-21 is ID


    def train_stargazer(self, epochs: int=10):
        self.train()
        best_val_loss = 99999

        for epoch in range(epochs):
            total_train_loss = 0

            for index in train_indices:
                image_path = os.path.join(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\images", images[index])
                labels_path = os.path.join(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\labels", labels[index])
                opened_image = Image.open(image_path).convert("RGB")
                label = []
                with open(labels_path, "r") as file:
                    for line in file:
                        label.append(torch.tensor([float(x) for x in line.split()]))
                label = torch.stack(label).to(device)


                x = to_tensor(opened_image).to(device)
                object = torch.zeros((6,6), device=device)
                for l in label:
                    x_box = l[1]
                    y_box = l[2]
                    object[int(y_box*6)][int(x_box*6)] = 1

                object = object.reshape(36)
                lossObjectness, lossBox, lossID = 0, 0, 0
                self.optim.zero_grad()
                prediction = self.forward(x)  # (9, 21)
                index2 = 0

                # objectness
                for _ in range(36):
                    cell_loss = nn.BCEWithLogitsLoss()(prediction[index2][0], object[index2])  # 0 is non object, 1 is object

                    if object[index2] == 0:
                        cell_loss = cell_loss * 0.5  # empty cells less weight
                    lossObjectness = lossObjectness + cell_loss
                    index2 = index2 + 1
                index2 = 0

                for l in label:

                    x_box = l[1]
                    y_box = l[2]

                    row = int(y_box * 6)
                    col = int(x_box * 6)

                    index2 = row * 6 + col

                    lossBox = lossBox + nn.MSELoss()(
                        torch.sigmoid(prediction[index2][1:5]),
                        l[1:5]
                    )

                    lossID = lossID + nn.CrossEntropyLoss()(
                        prediction[index2][5:21],
                        l[0].long()
                    )



                loss = lossObjectness/36 + lossBox/len(label) + lossID/len(label)
                loss.backward()
                self.optim.step()

                total_train_loss = total_train_loss + loss.item()

            avg_train_loss = total_train_loss/len(train_indices)
            avg_val_loss = self.validate_stargazer()
            print(f"epoch {epoch + 1}/{epochs}, train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model.pth")

    # AI slop
    def validate_stargazer(self):
        self.eval()
        total_val_loss = 0

        with torch.no_grad():
            for index in val_indices:
                image_path = os.path.join(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\images",
                                          images[index])
                labels_path = os.path.join(r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\labels",
                                           labels[index])
                opened_image = Image.open(image_path).convert("RGB")
                label = []
                with open(labels_path, "r") as file:
                    for line in file:
                        label.append(torch.tensor([float(x) for x in line.split()]))
                label = torch.stack(label).to(device)

                x = to_tensor(opened_image).to(device)
                object = torch.zeros((6, 6), device=device)
                for l in label:
                    x_box = l[1]
                    y_box = l[2]
                    object[int(y_box * 6)][int(x_box * 6)] = 1

                object = object.reshape(36)
                lossObjectness, lossBox, lossID = 0, 0, 0
                prediction = self.forward(x)
                index2 = 0

                for _ in range(36):
                    cell_loss = nn.BCEWithLogitsLoss()(prediction[index2][0], object[index2])

                    if object[index2] == 0:
                        cell_loss = cell_loss * 0.5
                    lossObjectness = lossObjectness + cell_loss
                    index2 = index2 + 1
                index2 = 0

                for l in label:
                    x_box = l[1]
                    y_box = l[2]

                    row = int(y_box * 6)
                    col = int(x_box * 6)

                    index2 = row * 6 + col

                    lossBox = lossBox + nn.MSELoss()(
                        torch.sigmoid(prediction[index2][1:5]),
                        l[1:5]
                    )

                    lossID = lossID + nn.CrossEntropyLoss()(
                        prediction[index2][5:21],
                        l[0].long()
                    )

                loss = lossObjectness / 36 + lossBox / len(label) + lossID / len(label)
                total_val_loss = total_val_loss + loss.item()
                print(
                    "obj:", (lossObjectness / 36).item(),
                    "box:", (lossBox / len(label)).item(),
                    "class:", (lossID / len(label)).item()
                )
        self.train()
        return total_val_loss / len(val_indices)

    def test_stargazer(self, objectness_threshold: float = 0.45):
        import matplotlib.pyplot as plt
        from PIL import ImageDraw
        self.load_state_dict(torch.load("best_model.pth", map_location=device))
        self.eval()

        current_index = 0

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        def show_image(test_position):

            axes[0].clear()
            axes[1].clear()

            # get actual dataset index from test split
            index = test_indices[test_position]

            image_path = os.path.join(
                r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\images",
                images[index]
            )

            labels_path = os.path.join(
                r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\train\labels",
                labels[index]
            )

            opened_image = Image.open(image_path).convert("RGB")

            actual_image = opened_image.copy()
            predicted_image = opened_image.copy()

            actual_draw = ImageDraw.Draw(actual_image)
            predicted_draw = ImageDraw.Draw(predicted_image)

            width, height = opened_image.size

            x = to_tensor(opened_image).to(device)

            with torch.no_grad():
                prediction = self.forward(x)  # (36, 21)

            # -------------------------
            # ACTUAL BOXES
            # -------------------------
            with open(labels_path, "r") as file:
                for line in file:
                    actual = torch.tensor(
                        [float(v) for v in line.split()]
                    )

                    x_box = actual[1]
                    y_box = actual[2]
                    box_width = actual[3]
                    box_height = actual[4]

                    x1 = (x_box - box_width / 2) * width
                    y1 = (y_box - box_height / 2) * height
                    x2 = (x_box + box_width / 2) * width
                    y2 = (y_box + box_height / 2) * height

                    actual_draw.rectangle(
                        [x1, y1, x2, y2],
                        outline="red",
                        width=3
                    )

                    actual_draw.text(
                        (x1, y1),
                        f"class {int(actual[0])}",
                        fill="red"
                    )

            # -------------------------
            # PREDICTED BOXES
            # -------------------------
            for cell in range(36):

                pred_objectness = torch.sigmoid(
                    prediction[cell][0]
                ).item()

                if pred_objectness < objectness_threshold:
                    continue

                pred_box = torch.sigmoid(
                    prediction[cell][1:5]
                )

                pred_class = torch.argmax(
                    prediction[cell][5:21]
                ).item()

                print(
                    "image:", images[index],
                    "cell:", cell,
                    "box:", pred_box.tolist(),
                    "objectness:", pred_objectness,
                    "class:", pred_class
                )

                x_box = pred_box[0].item()
                y_box = pred_box[1].item()
                box_width = pred_box[2].item()
                box_height = pred_box[3].item()

                x1 = (x_box - box_width / 2) * width
                y1 = (y_box - box_height / 2) * height
                x2 = (x_box + box_width / 2) * width
                y2 = (y_box + box_height / 2) * height

                predicted_draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="blue",
                    width=3
                )

                predicted_draw.text(
                    (x1, y1),
                    f"class {pred_class} ({pred_objectness:.2f})",
                    fill="blue"
                )

            axes[0].imshow(actual_image)
            axes[0].set_title("ACTUAL")
            axes[0].axis("off")

            axes[1].imshow(predicted_image)
            axes[1].set_title("PREDICTED")
            axes[1].axis("off")

            fig.suptitle(
                f"TEST {test_position + 1}/{len(test_indices)} - {images[index]}"
            )

            fig.canvas.draw_idle()

        def key_pressed(event):
            nonlocal current_index

            if event.key == "right":
                current_index += 1

                if current_index >= len(test_indices):
                    current_index = 0

                show_image(current_index)

            elif event.key == "left":
                current_index -= 1

                if current_index < 0:
                    current_index = len(test_indices) - 1

                show_image(current_index)

        fig.canvas.mpl_connect(
            "key_press_event",
            key_pressed
        )

        show_image(current_index)

        plt.show()

    def test_ultralytics_folder(self, folder_path, weights_path, confidence_threshold: float = 0.01):
        from ultralytics import YOLO
        import matplotlib.pyplot as plt
        from PIL import ImageDraw, ImageFont

        yolo_model = YOLO(weights_path)

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        folder_images = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_ext)
        )

        if not folder_images:
            print(f"No images found in {folder_path}")
            return

        current_index = 0

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        def show_image(position):
            axes[0].clear()
            axes[1].clear()

            image_name = folder_images[position]
            image_path = os.path.join(folder_path, image_name)

            opened_image = Image.open(image_path).convert("RGB")
            original_image = opened_image.copy()
            predicted_image = opened_image.copy()
            predicted_draw = ImageDraw.Draw(predicted_image)

            font = ImageFont.truetype("arial.ttf", 32)
            results = yolo_model.predict(source=opened_image, conf=confidence_threshold, verbose=False)
            result = results[0]

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                pred_confidence = box.conf[0].item()
                pred_class = int(box.cls[0].item())

                print("image:", image_name, "box:", [x1, y1, x2, y2], "confidence:", pred_confidence, "class:",
                      pred_class)

                predicted_draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
                predicted_draw.text(
                    (x1, y1),
                    f"{CLASS_NAMES.get(pred_class, pred_class)} ({pred_confidence:.2f})",
                    fill="red",
                    font=font
                )

            axes[0].imshow(original_image)
            axes[0].set_title("ORIGINAL")
            axes[0].axis("off")

            axes[1].imshow(predicted_image)
            axes[1].set_title("ULTRALYTICS YOLO")
            axes[1].axis("off")

            fig.suptitle(f"{position + 1}/{len(folder_images)} - {image_name}")
            fig.canvas.draw_idle()

        def key_pressed(event):
            nonlocal current_index
            if event.key == "right":
                current_index = (current_index + 1) % len(folder_images)
                show_image(current_index)
            elif event.key == "left":
                current_index = (current_index - 1) % len(folder_images)
                show_image(current_index)

        fig.canvas.mpl_connect("key_press_event", key_pressed)
        show_image(current_index)
        plt.show()


def train_ultralytics():
    model = YOLO("yolo26n.pt")

    model.train(
        data=r"C:\Users\ronan\Desktop\AI\TestingConst.v1i.yolo26\data.yaml",
        epochs=20,
        imgsz=640,
        device=0,
        amp = False,
        workers = 0
    )
    return model.trainer.best



# model.train_stargazer(epochs=20)
if __name__ == "__main__":
    model = Stargazer().to(device)
    weights_path = r"C:\Users\ronan\Desktop\AI\runs\detect\train-13\weights\best.pt"
    #model.test_ultralytics(weights_path=weights_path)
    model.test_ultralytics_folder(
        folder_path=r"C:\Users\ronan\Desktop\AI\lol",
        weights_path=weights_path
    )
