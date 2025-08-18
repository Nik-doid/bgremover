# Background Remover API

A Python FastAPI application that removes image backgrounds using **U²-Net**.  
The API returns images with a **white background** (JPEG format).

---

## Features

- Upload an image via web form or API.
- Automatically removes background.
- Outputs JPEG with white background.
- Handles images entirely in memory; no temporary files are saved.
- Uses U²-Net deep learning model for accurate human/object segmentation.
- Optionally uses `rembg` for faster pre-processing if available.

---

## Citation

```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```
