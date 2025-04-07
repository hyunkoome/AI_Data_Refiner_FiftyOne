# AI_Data_Refiner_FiftyOne
An AI-powered data refinement tool built on top of FiftyOne, combining automated curation and generative sample synthesis.

# AI_Data_Refiner_FiftyOne

An automated data refinement tool built with [FiftyOne](https://voxel51.com/tools/fiftyone/), [Diffusers](https://huggingface.co/docs/diffusers/index), and Generative AI.  
This tool detects low-confidence predictions from a dataset and generates synthetic images to enrich and balance the dataset.

---

## 🚀 Project Purpose

> "Data-centric AI made simple with FiftyOne + Generative AI."

This project focuses on:
- Automatically identifying low-confidence samples in object detection datasets
- Generating new, diverse samples using Stable Diffusion
- Creating a refined dataset to support retraining or analysis

---

## 🛠️ Features

- 🧠 **Prediction filtering**: Finds predictions with confidence below a threshold
- 🎨 **Image generation**: Uses `stabilityai/sd-turbo` to synthesize new data
- 🗂️ **Dataset tracking**: Organized with FiftyOne datasets
- 📊 **Sample preview**: Quickly explore labels, classes, and confidence stats
- ♻️ **Dummy prediction fallback**: Works even without a model (for testing)

---

## 📁 Project Structure

```
AI_Data_Refiner_FiftyOne/
├── config.py                 # Global config: thresholds, paths, model name
├── fiftyone_utils.py        # All FiftyOne dataset-related utility functions
├── setup_dataset.py         # First-time dataset download & registration
├── main.py                  # Main pipeline: filter → prompt → generate → save
├── requirements.txt         # install packages
├── assets/                  # Folder where generated images are saved
└── README.md                # ← This file
```

---

## ⚙️ Setup & Requirements


Create Python Env
```shell
# Recommended: Python 3.10, CUDA 12.6

conda create -n datarefiner python==3.10
activate datarefiner

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

---

## 🔧 Step-by-Step Usage

### 1. Download and register BDD100K

```bash
python setup_dataset.py
```

Make sure to manually download BDD100K data into this folder structure:

```
BDD100K/
├── images/
│   └── 100k/
│       ├── train/
│       └── val/
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

### 2. Run the main pipeline

```bash
python main.py
```

- Filters low-confidence samples
- Generates new images using prompts
- Saves outputs to `assets/` folder and registers a new FiftyOne dataset

---

## 📸 Sample Output

Each generated sample includes:
- `filepath`: path to the generated image
- `prompt`: used to guide image generation
- `tags`: e.g., `["generated"]`

You can visually inspect the dataset using FiftyOne GUI.

---

## 🔮 Future Plans

- Replace dummy predictions with Hugging Face detection models (e.g. `facebook/detr-resnet-50`)
- Add CLIP-based prompt generation
- Export generated data in COCO or YOLO format
- Integrate with active learning frameworks

---

## 🤝 Acknowledgements

- [FiftyOne](https://voxel51.com/tools/fiftyone/) by Voxel51
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face

---

## 📬 Contact

Feel free to reach out if you're interested in collaborations or have feedback!