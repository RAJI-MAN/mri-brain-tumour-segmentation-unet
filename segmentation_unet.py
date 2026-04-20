import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------------
# LOAD DATA (FROM YES FOLDER)
# ----------------------------
data_path = "data/yes"

images = []

for file in os.listdir(data_path)[:100]:   # limit for speed
    img = cv2.imread(os.path.join(data_path, file), 0)

    if img is None:
        continue

    img = cv2.resize(img, (128,128))
    img = img / 255.0
    images.append(img)

images = np.array(images)

# ----------------------------
# CREATE SIMPLE MASKS
# ----------------------------
# This is NOT real segmentation — just learning step
masks = (images > 0.5).astype(np.float32)

# Convert to tensors
X = torch.tensor(images).unsqueeze(1).float()
y = torch.tensor(masks).unsqueeze(1).float()

# ----------------------------
# SIMPLE U-NET STYLE MODEL
# ----------------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = UNet()

# ----------------------------
# DICE LOSS
# ----------------------------
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-5

    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ----------------------------
# TRAIN MODEL
# ----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    pred = model(X)
    loss = dice_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ----------------------------
# VISUALISE RESULTS
# ----------------------------
pred = torch.sigmoid(model(X)).detach().numpy()

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(images[0], cmap='gray')
plt.title("MRI")

plt.subplot(1,3,2)
plt.imshow(masks[0], cmap='gray')
plt.title("Mask")

plt.subplot(1,3,3)
plt.imshow(pred[0][0], cmap='gray')
plt.title("Prediction")

plt.show()