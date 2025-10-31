# Brain Tumor Necrosis Segmentation (Whole-Slide Images)

This project focuses on **automatic segmentation of tumor necrosis** regions from **gigapixel Whole-Slide Images (WSI)** in brain histopathology.  
The model is based on **EfficientNet-B4 + UNet++**, optimized for high-resolution patch-based inference.

<p align="center">
  <img src="assets/cover_wsi_example.png" width="700"/>
</p>

---

## 🔬 Background

**Necrosis** in brain tumors is a key indicator of:
- Tumor grade
- Aggressiveness
- Treatment prognosis

However, **WSI files are extremely high resolution (up to 80,000 × 80,000 px)**, requiring a specialized pipeline for:
- Gigapixel **downscaling**
- **Patch extraction**
- Segmentation and reconstruction back to WSI space

---

## 🧠 Project Pipeline

| Step | Description |
|------|-------------|
| 1️⃣ Data Preparation | Whole-Slide Images & pathology annotations were collected |
| 2️⃣ Downscaling | WSI resolution reduced for patch grid extraction |
| 3️⃣ Patch Extraction | Extracted image & mask patches from annotated regions |
| 4️⃣ Model | EfficientNet-B4 UNet++ trained on extracted patches |
| 5️⃣ Inference | Model predicts necrosis per patch |
| 6️⃣ Reconstruction | Patch predictions merged back into WSI mask |

---

## 🗂 Example Data

**Original WSI & Mask Annotation**

<p align="center">
  <img src="assets/wsi_original.png" width="420"/>
  <img src="assets/wsi_mask.png" width="420"/>
</p>

---

### 🎨 Patch Extraction

<p align="center">
  <img src="assets/patch_original.png" width="420"/>
  <img src="assets/patch_mask.png" width="420"/>
</p>

---

## 🧩 Model Architecture
