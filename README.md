# Fashion Intelligence through Multitask Learning

A unified deep learning system for **fashion image classification**, **attribute prediction**, and **image-based retrieval**, designed using **multitask learning** principles and trained on the DeepFashion dataset.

This project is part of the BA/BSc thesis requirements at Central European University, but the codebase is developed as a **generalizable software product** that can be used off-the-shelf by researchers and developers working on fashion vision applications.

---

## Overview

This repository implements a multitask model that simultaneously:

- Classifies clothing images into fine-grained **category labels** (e.g. blouse, tee, pants)
- Predicts multiple **visual attributes** per image (e.g. floral, long-sleeve, striped)
- Retrieves **visually similar items** from a gallery using **triplet-loss embeddings**

These tasks are handled within a shared ResNet-50 backbone architecture, trained using staged multitask learning for improved performance and stability.

---

## Key Use Cases (Beyond the Thesis)

Although the thesis uses the DeepFashion dataset, this system is adaptable to **any structured fashion image dataset** and can be applied in real-world settings like:

-  **Secondhand marketplaces** – Automatically tag user-uploaded images and retrieve visually similar resale items
-  **Visual search engines** – Build fashion recommendation or discovery tools based on image similarity
-  **Retail catalog optimization** – Clean and enrich large fashion inventories using automated categorization and tagging
-  **Academic research** – Use the modular training framework to benchmark or extend fashion-related vision tasks

---

##  Repository Structure

fashion-multitask-model/

├── classification/      →  Single-task models (category & attribute)

├── retrieval/           →  Triplet-loss image retrieval model

├── multitask/           →  Unified multitask models (staged & joint training)

├── tests/               →  Basic unit tests for model functionality

├── requirements.txt     →  Python dependencies

└── README.md            →  Project overview and instructions


---

##  Installation

### Clone the repository
bash:
git clone https://github.com/EleneZuroshvili/fashion-multitask-model.git
cd fashion-multitask-model

### Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

### Install dependencies
pip install -r requirements.txt

This project is compatible with **Python ≥ 3.8**. GPU is recommended for training.

---

## Dataset Setup

This project uses the **DeepFashion In-Shop Clothes Retrieval and Attribute Prediction Benchmarks**.

* Download the datasets: [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
* Unzip the contents and place them in a `data/` folder at the root of this repo:

  E.g: fashion-multitask-model/data/

Note: You may adapt this repo to other datasets with compatible label formats.

---

## How to Run

All training pipelines are runnable via Python scripts or Jupyter notebooks.

### 1. Train Single-Task Classifier

bash:
python training/train_category.py
python training/train_attribute.py

### 2. Train Retrieval Model

bash:
python training/train_retrieval.py

### 3. Train Multitask Model 1 (Classification)

bash
python training/train_multi_classification.py

### 3. Train Multitask Model Final (Staged Training)

bash
python training/train_multi_final.py

---

## Inputs and Outputs

### Inputs

* Image folders (structured or referenced via text files)
* Category labels (single-class per image)
* Attribute labels (multi-label per image)
* Product IDs (for retrieval triplets)

### Outputs

* Trained models saved as `.pth` files
* Evaluation metrics (e.g., accuracy, F1, Recall\@K)
* Retrieval embeddings (for use in similarity search)

---

## Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.10
* torchvision
* scikit-learn
* matplotlib
* pandas
* tqdm
* albumentations
* pytorch-metric-learning

To install everything:

bash
pip install -e 

- If you want a more comprehensive requirements list, run this command:

bash
pip install -r 

---

## License & Attribution

This project is licensed under the MIT License. You are free to use, modify, and distribute this code, including for commercial purposes, provided that the original copyright notice is retained.

Please cite this project as:

> Zuroshvili, E. (2025). *Fashion Intelligence Through Multitask Learning: A Unified Model for Product Categorization and Retrieval*. BA thesis, Central European University, Vienna.

**Credits**

* Author: Elene Zuroshvili
* University: Central European University, Vienna

---

## Contact

Feel free to open an [Issue](https://github.com/EleneZuroshvili/fashion-multitask-model/issues) or reach out via [LinkedIn](https://www.linkedin.com/in/elene-zuroshvili).

---
