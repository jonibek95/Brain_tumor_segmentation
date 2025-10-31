# Brain Tumor Segmentation

## Overview
This repository contains experimentation code for brain tumor segmentation on histopathology whole-slide images (WSI) and their derived patches. The project leverages PyTorch Lightning, `segmentation_models_pytorch`, and a collection of WSI utilities to extract, preprocess, augment, train, and evaluate segmentation models on datasets such as CODIPAI and Camelyon16.

## Key Features
- **Lightning Segmentation Model** – `CODIPAIModel.py` defines a configurable U-Net++ model with EfficientNet encoders and Dice loss.
- **Flexible Datasets** – `CODIPAIDataSet.py` supports both `.png` and `.npy` patch formats and can be extended to new layouts.
- **WSI Tooling** – Utilities under `modules/` and `wsi_image_utils.py` handle OpenSlide integration, patch extraction, and heatmap rendering.
- **Rich Augmentation Pipeline** – `dataloader_utils.py` adds augmentation strategies such as rotations, elastic/grid distortions, histogram/color transfer, and inpainting.
- **Camelyon16 Integration** – `cam_dataloader.py` provides ready-to-use dataloaders and S3 helpers for this benchmark dataset.

## Repository Structure
```
├── CODIPAIDataSet.py           # Patch-level dataset for PNG/NPY samples
├── CODIPAIModel.py             # PyTorch Lightning segmentation module
├── cam_dataloader.py           # Camelyon16 dataset loader and helpers
├── dataloader_utils.py         # Augmentation pipeline and helpers
├── modules/
│   ├── dataset/                # Annotation/label utilities and patch metadata
│   ├── models/                 # ResUNet, DSMIL, and model utilities
│   └── patch/, preprocess/, ...# WSI patch extraction and processing helpers
├── TEST/                       # End-to-end WSI inference example
├── TRAIN.ipynb                 # Notebook for PNG-based training
├── Train_npy.ipynb             # Notebook for NPY-based training
├── extract_patches_from_*.ipynb# WSI patch extraction workflows
├── codipai.tsv                 # Dataset path configuration
└── codipai_anno.tsv            # Annotation path configuration
```

## Environment Setup
The code targets Python 3.9+. GPU acceleration is recommended but not required. Install system dependencies (e.g., `openslide-tools`) before installing Python packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchmetrics pytorch-lightning
pip install segmentation-models-pytorch albumentations timm
pip install opencv-python pillow scikit-image pandas openslide-python boto3 fiona rich
```

> Note: `segmentation_models_pytorch` requires `timm` for EfficientNet encoders. On Linux, install OpenSlide libraries via `sudo apt install openslide-tools`.

## Data Preparation

1. **Configure Paths**  
   Update `codipai.tsv` and `codipai_anno.tsv` to reflect local WSI, annotation, and output directories.

2. **Extract Patches**  
   Use the notebooks `extract_patches_from_jpg_png.ipynb` or `extract_patches_from_WSI.ipynb` to generate patches and masks. Adjust notebook parameters to your storage layout.

3. **Expected Directory Layout**  
   When training with PNG patches, the dataloaders expect a structure similar to:
   ```
   data/
   └── patches/
       ├── 01-original/
       │   ├── train/
       │   └── val/
       └── 02-mask/
           ├── train/
           └── val/
   ```
   For `.npy` datasets, uncomment the relevant sections in `CODIPAIDataSet.py`.

## Quick Start: Training
```python
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from CODIPAIDataSet import CODIPAIDataSet
from CODIPAIModel import CODIPAIModel

train_ds = CODIPAIDataSet(
    "./data/patches/01-original/train",
    "./data/patches/02-mask/train",
)
val_ds = CODIPAIDataSet(
    "./data/patches/01-original/val",
    "./data/patches/02-mask/val",
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

model = CODIPAIModel(
    arch="unetplusplus",
    encoder_name="efficientnet-b3",
    in_channels=3,
    out_classes=1,
)

trainer = Trainer(
    accelerator="auto",
    devices="auto",
    max_epochs=100,
    precision="16-mixed",
    log_every_n_steps=10,
)

trainer.fit(model, train_loader, val_loader)
```

Training logs are saved to `lightning_logs/`. To monitor with TensorBoard, run `pip install tensorboard` followed by `tensorboard --logdir lightning_logs`.

## Evaluation
- **Patch-level metrics**: `CODIPAIModel.shared_epoch_end` logs IoU, accuracy, precision, recall, and F1 scores for training/validation/test splits.
- **Lightning testing**: invoke `trainer.test()` with appropriate dataloaders to reuse the same metrics on held-out sets.

## Whole-Slide Inference
The `TEST/TEST_Codipai.ipynb` notebook demonstrates a full WSI segmentation workflow:
1. Load the WSI via `openslide` using helpers in `wsi_image_utils.py`.
2. Extract tiles, run the Lightning model, and aggregate predictions.
3. Save heatmaps and segmentation overlays under `./results/wsi/<slide-id>/`.

Adjust tile size, stride, and batching to match hardware constraints.

## Additional Utilities
- `modules/models/res_U_net.py`: Residual U-Net variants for both classification and segmentation tasks.
- `modules/dataset/_annotation.py`: Patch dataset with random flips/rotations applied to annotations.
- `dataloader_utils.py`: Data augmentation routines, histogram matching, and visualization helpers.
- `cam_dataloader.py`: Camelyon16-specific dataloaders and XML/GeoJSON parsing for annotations; includes AWS S3 download support.

## Troubleshooting
- **OpenSlide errors**: Ensure system-level OpenSlide libraries are installed and accessible.
- **Missing encoders**: Install the required `timm` version when using EfficientNet or other pretrained backbones.
- **Mask scaling**: `CODIPAIDataSet` normalizes masks by dividing by 255. Adjust or disable if masks are already in `[0, 1]`.
- **Performance tuning**: Reduce image size, batch size, or `num_workers` when running on CPU-only environments.

## Contributing
Contributions are welcome. To propose changes:
1. Open an issue describing the bug or enhancement.
2. Provide reproducible steps or minimal examples.
3. Submit a pull request with concise commits and adhere to existing coding conventions.

## License
No explicit license file is present. Please verify usage rights with the repository owners before distributing or commercializing the code.
