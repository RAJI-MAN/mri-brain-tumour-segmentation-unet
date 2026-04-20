# MRI Brain Tumour Segmentation using U-Net

This project implements a deep learning-based MRI segmentation pipeline using the BraTS dataset for brain tumour detection and localisation.

---

## 🚀 Key Features
- U-Net architecture with skip connections
- Multi-patient MRI dataset training
- Dice loss optimisation
- Quantitative evaluation using Dice score

---

## 📊 Results

Dice Score achieved: **0.73**

### Segmentation Output
![Segmentation](results/brats_segmentation.png)

---

## 🧠 Methodology
- Processed 3D MRI volumes into 2D slices
- Converted tumour labels into binary segmentation masks
- Trained convolutional neural network (U-Net)
- Evaluated using Dice similarity coefficient

---

## 🛠 Technologies
- Python
- PyTorch
- NumPy
- Nibabel
- Matplotlib

---

## 📌 Dataset
BraTS (Brain Tumor Segmentation Challenge)

---

## 👨‍💻 Author
Rajeevan
