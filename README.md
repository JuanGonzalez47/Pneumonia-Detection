# Pneumonia Detection

This project provides a complete pipeline for detecting pneumonia from chest X-ray images using classical machine learning and deep learning techniques. The workflow includes image preprocessing, feature extraction (HOG, LBP), feature selection, dimensionality reduction, model training, evaluation, and reporting. The project is implemented in Python and includes Jupyter notebooks for reproducibility.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)

## Project Overview
The goal of this project is to classify chest X-ray images as either normal or showing signs of pneumonia. The pipeline covers:
- Image preprocessing (CLAHE, resizing)
- Feature extraction (Histogram of Oriented Gradients, Local Binary Patterns)
- Feature selection and statistical analysis
- Dimensionality reduction (PCA, LDA, t-SNE)
- Model training and evaluation (Logistic Regression, QDA, GaussianNB, CNN)
- Visualization and reporting

## Dataset
The dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) available on Kaggle. It contains chest X-ray images categorized as 'Pneumonia' and 'Normal'.

**How to obtain the dataset:**
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. Download the dataset (you need a Kaggle account).
3. Extract the contents into the `data/raw/` directory of this repository.

## Installation
1. Clone this repository:
	```bash
	git clone https://github.com/yourusername/Pneumonia-Detection.git
	cd Pneumonia-Detection
	```
2. (Recommended) Create a virtual environment:
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	```
3. Install the required packages:
	```bash
	pip install -r requirements.txt
	```

## Usage
The main workflow is organized in Jupyter notebooks located in the `notebooks/` directory:

- `preproc_pneumonia.ipynb`: Image preprocessing
- `feature_engineering.ipynb` and `feature_engineering_02.ipynb`: Feature extraction and selection
- `modeling_evaluation.ipynb`: Model training and evaluation

To run a notebook:
```bash
jupyter notebook
```
Then open the desired notebook and run the cells sequentially.

## Project Structure
```
Pneumonia-Detection/
├── data/
│   ├── raw/                # Raw images (downloaded from Kaggle)
│   ├── processed/          # Processed data and extracted features
│   └── samples/            # Sample images for quick tests
├── notebooks/              # Jupyter notebooks for each pipeline stage
├── src/                    # Source code (modularized functions and scripts)
├── experiments/            # Experiment logs and runs
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Results
The project demonstrates that Local Binary Patterns (LBP) features provide strong discriminative power for pneumonia detection. Classical models such as Logistic Regression, QDA, and GaussianNB were evaluated, with Logistic Regression showing the best specificity for medical applications. All results, visualizations, and exported features are available in the `data/processed/` and `experiments/` folders.

## Future Work
- Increase dataset size and diversity
- Apply data augmentation techniques
- Test more robust models (e.g., ensemble methods, advanced CNNs)
- Integrate clinical validation and real-world testing

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
