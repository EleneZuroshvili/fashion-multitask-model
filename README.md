# Fashion Intelligence through Multitask Learning

A unified deep learning system for **fashion image classification**, **attribute prediction**, and **image-based retrieval**, designed using **multitask learning** principles and trained on the DeepFashion dataset.

This project is part of the BA/BSc thesis 'Fashion Intelligence through Multitask Learning: A Unified Model for Product Classification and Retrieval' by Elene Zuroshvili at Central European University, but the codebase is developed as a **generalizable software product** that can be used off-the-shelf by researchers and developers working on fashion vision applications.

---

## Overview

This repository implements a multitask model that simultaneously:

- Classifies clothing images into fine-grained **category labels** (e.g. blouse, tee, pants)
- Predicts multiple **visual attributes** per image (e.g. floral, long-sleeve, striped)
- Retrieves **visually similar items** from a gallery using **triplet-loss embeddings**

These tasks are handled within a shared ResNet-50 backbone architecture, trained using staged multitask learning for improved performance and stability. 

The repository also features models for each of these three tasks separately that can be used without the multitask architecture. In summary, one could implement five different models: single category classification, single attribute prediction, single image retrieval, multitask classification for categories and attributes, and a multitask architecture for category classification, attribute prediction, and image retrieval.

---

## Key Use Cases (Beyond the Thesis)

Although the repository uses the DeepFashion dataset, this system is adaptable to **any structured fashion or garment image dataset** and can be applied in real-world settings like:

-  **Secondhand marketplaces** – Automatically tag user-uploaded images and retrieve visually similar resale items
-  **Visual search engines** – Build fashion recommendation or discovery tools based on image similarity
-  **Retail catalog optimization** – Clean and enrich large fashion inventories using automated categorization and tagging
-  **Academic research** – Use the modular training framework to benchmark or extend fashion-related vision tasks

---

##  Repository Structure

the repository is structured in the following parts:

### 1. src/data - This file contains PyTorch `Dataset` classes that serve as custom data loaders for various supervised tasks in the DeepFashion dataset. Each class handles a specific task — category classification, attribute prediction, retrieval, or multitask learning — and provides images and labels in a format ready for model training and evaluation.

#### Defined Dataset Classes

- **`DeepFashionCategories`**  
  Loads image–category pairs for **multi-class category classification**.  
  - Reads image paths from `<split>.txt` and labels from `<split>_cate.txt`
  - Labels are zero-indexed category IDs
  - **Returns:** `(image, category_label)`

- **`DeepFashionAttributes`**  
  Loads image–attribute pairs for **multi-label attribute classification**.  
  - Reads image paths from `<split>.txt` and binary attribute vectors from `<split>_attr.txt`
  - **Returns:** `(image, attribute_tensor)` (multi-hot encoded)

- **`DeepFashionRetrieval`**  
  Loads image–item ID pairs for **image retrieval tasks using triplet loss**.  
  - Parses a centralized annotation file (e.g., `list_eval_partition.txt`)  
  - Keeps only samples from the selected `split` (e.g., `train`, `query`, `gallery`)
  - **Returns:** `(image, item_id)`

- **`DeepFashionMulti`**  
  Combined loader for **multitask learning**, returning both category and attribute labels.  
  - Reads from `<split>.txt`, `<split>_cate.txt`, and `<split>_attr.txt`
  - **Returns:** `(image, category_label, attribute_tensor)`

### 2. src/models - This folder contains task-specific PyTorch model architectures built on a shared **ResNet-50 backbone**, adapted for classification, attribute prediction, image retrieval, and multitask learning. Each model includes custom heads with dropout, batch normalization, and non-linear layers tailored to the output type.

#### Model Files

- **`attribute_prediction.py`**  
  A model for **multi-label attribute prediction**.  
  - Backbone: ResNet-50 (pretrained or not)  
  - Head: Fully connected layers with ReLU, batch normalization, dropout  
  - **Output:** A vector of length `num_attrs` (binary relevance for each attribute)

- **`category_classification.py`**  
  A model for **multi-class category classification**.  
  - Identical ResNet-50 backbone  
  - Final layer: Softmax-style head producing logits over `num_categories`

- **`image_retrieval.py`**  
  A model for **deep metric learning with triplet loss**.  
  - Replaces classification head with a **projection head** that outputs embeddings  
  - Used for computing similarity between items in embedding space

- **`multitask_cclassification.py`**  
  A multitask model that jointly predicts both **categories** and **attributes**.  
  - Shared ResNet-50 feature extractor  
  - Two separate heads: one for classification, one for attribute prediction  
  - **Output:** `(category_logits, attribute_vector)`

- **`multitask_final.py`**  
  A multitask model that combines all three tasks: **category classification, attribute prediction, and image retrieval**.  
  - Shared ResNet-50 backbone  
  - Three heads: classification, attributes, and embedding projection  
  - **Output:** `(category_logits, attribute_vector, embedding_vector)`
 
### 3. src/training - This folder contains training scripts for each individual task and multitask setup. Each script follows a consistent pipeline using PyTorch and includes argument parsing, dataset loading, data augmentation, model initialization, training loop, validation evaluation, and early stopping.

#### Training Script Structure

All scripts follow a similar layout:
- Argument parser for flexible CLI training (`--img-dir`, `--anno-dir`, `--output-dir`, etc.)
- Reproducibility setup (manual seeds for `random`, `numpy`, and `torch`)
- Data augmentation using `torchvision.transforms` (e.g., random flips, color jitter, erasing)
- Loss weighting based on attribute imbalance (for attribute tasks)
- Learning rate scheduling (`CosineAnnealingLR`)
- Evaluation metrics (e.g., **Hamming loss**, **Micro F1**, **accuracy**, **Recall@K**)
- **Early stopping** based on validation performance
- Best model checkpoint saved to disk

#### Example Files

- **`train_attribute.py`**  
  Trains the `AttributeModel` using **multi-label BCEWithLogitsLoss** with positive class weighting.  
  - Validation includes Hamming loss and Micro-F1
  - Saves best model to: `best_attribute_model.pth`

- **`train_category.py`**  
  Trains the `CategoryModel` using **cross-entropy loss** for multi-class classification.  
  - Validation uses top-1 accuracy, and Macrp-F1

- **`train_retrieval.py`**  
  Trains the `RetrievalModel` using **triplet loss** (e.g., with batch-hard mining).  
  - Evaluation via Recall@K over query/gallery sets

- **`train_multi_classification.py`**  
  Trains a multitask model with shared backbone and two heads: category + attributes.  
  - Uses a combined loss: `CE + BCEWithLogits`

- **`train_multitask_final.py`**  
  Trains the full multitask model with **three outputs**: category, attributes, and embeddings.  
  - Total loss is a weighted sum: `CE + BCE + TripletLoss`
  - Evaluation includes classification accuracy, attribute metrics, and retrieval recall

#### Output

Each script saves the best model (by validation metric) to the directory specified via `--output-dir`. Checkpoints are stored as `.pth` files and can be reloaded for fine-tuning or inference.

### 4. src/evaluation - This folder contains a single script, `eval.py`, which provides a **unified interface for evaluating all models** trained on the DeepFashion dataset. It supports classification, attribute prediction, image retrieval, multitask learning, and full multimodal evaluation.

A modular evaluation script that loads a trained model checkpoint, prepares the appropriate dataset split, and computes relevant evaluation metrics. Run it with the `--task` argument to evaluate a specific model type.

##### Supported Tasks & Metrics

- **`classification`**  
  Evaluates category classification model.  
  - **Metrics:** Accuracy, Macro-F1, `classification_report`  
  - **Dataset:** `DeepFashionCategories`

- **`attribute`**  
  Evaluates multi-label attribute prediction.  
  - **Metrics:** Hamming loss, Micro-F1  
  - **Dataset:** `DeepFashionAttributes`

- **`retrieval`**  
  Evaluates retrieval model using Recall@1 over query–gallery split.  
  - **Metrics:** Recall@1  
  - **Dataset:** `DeepFashionRetrieval`

- **`multitask`**  
  Evaluates multitask model with both classification and attributes.  
  - **Metrics:** Macro-F1 for categories, Micro-F1 for attributes  
  - **Dataset:** `DeepFashionMulti`

- **`multimodal`**  
  Evaluates the full multitask + retrieval model.  
  - **Metrics:** Category F1 score and Recall@1  
  - **Datasets:**  
    - Attributes: `DeepFashionAttributes`  
    - Retrieval: `DeepFashionRetrieval` (query/gallery)

##### Example Usage

bash: 
python evaluation/eval.py \
  --task classification \
  --img-dir path/to/images \
  --anno-dir path/to/annotations \
  --checkpoint path/to/best_model.pth

### 5. setup.py: 
Defines the project as a pip-installable package for easier integration and CLI access. You can install the package locally with:

bash: pip install -e .

Console Commands:

After installation, the following CLI commands become available:

| Command              | Description                                      |
|----------------------|--------------------------------------------------|
| `train-classification` | Run category classification training            |
| `train-attribute`      | Run attribute prediction training               |
| `train-retrieval`      | Run image retrieval training                    |
| `train-multitask`      | Train category + attribute multitask model      |
| `train-multimodal`     | Train full multitask + retrieval model          |
| `eval-fashion`         | Run evaluation on any trained model             |


These entry points map directly to the respective training and evaluation scripts in the training/ and evaluation/ folders.

### 6.requirements.txt:
A full list of pinned dependencies for exact reproducibility. Install using:
bash
pip install -r requirements.txt

---


## Getting Started: Step-by-Step Instructions

---

### 1. Install Dependencies

Clone the repo and install all required packages:

```bash
git clone https://github.com/yourusername/fashion-multitask-model.git
cd fashion-multitask-model

# Install exact versions for reproducibility
pip install -r requirements.txt

# (Optional) Install as a local package with CLI commands
pip install -e .
```

---

### 2. Prepare the Dataset
This project uses the **DeepFashion In-Shop Clothes Retrieval and Attribute Prediction Benchmarks**.

* Download the datasets: [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
* Unzip the contents and place them in a `data/` folder at the root of this repo:

  E.g: fashion-multitask-model/data/

Note: You may adapt this repo to other datasets with compatible label formats.

If you decide to use the the DeepFashion dataset, it needs to be structured like this:

```
deepfashion/
├── img/
│   └── <image files>
└── annotations/
    ├── train/
    │   ├── train.txt
    │   ├── train_cate.txt
    │   └── train_attr.txt
    ├── val/
    ├── test/
    └── list_eval_partition.txt  # E.g. For retrieval
```

Tip: Place the dataset wherever you like, then pass the path using `--img-dir` and `--anno-dir` in the commands below.

---

### 3. Train a Model

Choose a task and run the corresponding training script:

| Task                     | Command Example                                      |
|--------------------------|------------------------------------------------------|
| Category Classification  | `train-classification --img-dir ... --anno-dir ...` |
| Attribute Prediction     | `train-attribute --img-dir ... --anno-dir ...`       |
| Image Retrieval          | `train-retrieval --img-dir ... --anno-path ...`      |
| Multitask (Cate + Attr)  | `train-multitask --img-dir ... --anno-dir ...`       |
| Full Multimodal (All 3)  | `train-multimodal --attr-img-dir ... --retr-img-dir ...` |

Example:

```bash
train-attribute \
  --img-dir ./deepfashion/img \
  --anno-dir ./deepfashion/annotations \
  --output-dir ./checkpoints/attribute
```

Each script supports common flags like:

- `--batch-size`
- `--lr`
- `--epochs`
- `--dropout`
- `--pretrained`

---

### 4. Evaluate a Trained Model

Use `eval-fashion` and specify the task:

```bash
eval-fashion \
  --task attribute \
  --img-dir ./deepfashion/img \
  --anno-dir ./deepfashion/annotations \
  --checkpoint ./checkpoints/attribute/best_attribute_model.pth
```

| Task          | Flag: `--task`     |
|---------------|--------------------|
| Category      | `classification`   |
| Attributes    | `attribute`        |
| Retrieval     | `retrieval`        |
| Multitask     | `multitask`        |
| Multimodal    | `multimodal`       |

---

### 5. Output Files

- Trained model checkpoints are saved in `--output-dir` as `.pth` files.
- Evaluation prints key metrics like Accuracy, F1 score, Recall@1, Hamming loss, etc.

---

### 6. Project Structure

You can structure your project like this for convenience:

```
fashion-multitask-model/
├── data/               # PyTorch dataset classes
├── models/             # Model architectures
├── training/           # Training scripts
├── evaluation/         # Evaluation logic
├── requirements.txt
├── setup.py
└── README.md
```

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
## Acknowledgment and Citation:

Dataset: [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

@inproceedings{liuLQWTcvpr16DeepFashion, author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou}, title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations}, booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, month = {June}, year = {2016} }

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
