import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------------
# LOAD MRI + MASK
# ----------------------------
base_path = "data/brats/patient1"

mri_path = os.path.join(base_path, "BraTS20_Training_001_flair.nii")
mask_path = os.path.join(base_path, "BraTS20_Training_001_seg.nii")

mri = nib.load(mri_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

# ----------------------------
# PREPARE SLICES
# ----------------------------
images = []
masks = []

for i in range(60, 100):
    img_slice = mri[:, :, i]
    mask_slice = mask[:, :, i]

    img_slice = img_slice / np.max(img_slice)
    mask_slice = (mask_slice > 0).astype(np.float32)

    images.append(img_slice)
    masks.append(mask_slice)

images = np.array(images)
masks = np.array(masks)

X = torch.tensor(images).unsqueeze(1).float()
y = torch.tensor(masks).unsqueeze(1).float()

# ----------------------------
# REAL U-NET (WITH SKIP CONNECTIONS)
# ----------------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),  # skip connection
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)

        # Decoder
        x4 = self.up(x3)

        # 🔥 Skip connection
        x5 = torch.cat([x4, x1], dim=1)

        x6 = self.dec1(x5)

        return self.final(x6)

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
# DICE SCORE
# ----------------------------
def dice_score(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-5)

# ----------------------------
# TRAINING
# ----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):   # increased epochs
    pred = model(X)
    loss = dice_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ----------------------------
# EVALUATION
# ----------------------------
with torch.no_grad():
    pred = model(X)
    score = dice_score(pred, y)

print("\nDice Score:", score.item())

# ----------------------------
# VISUALISATION
# ----------------------------
pred = torch.sigmoid(model(X)).detach().numpy()

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(images[0], cmap='gray')
plt.title("MRI")

plt.subplot(1,3,2)
plt.imshow(masks[0], cmap='gray')
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(pred[0][0], cmap='gray')
plt.title("Prediction")

plt.show()