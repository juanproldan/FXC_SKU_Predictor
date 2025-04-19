# SKU Prediction System for Car Parts Bids

This project aims to automate the prediction of Stock Keeping Units (SKUs) for car parts based on bid descriptions using machine learning.

## Project Goal

To build a scalable and maintainable batch processing system that improves the efficiency and accuracy of SKU identification from potentially inconsistent bid descriptions.

## Features

- Load historical bid data from a configurable JSON file path.
- Preprocess text descriptions (spelling correction, standardization, handling Spanish language).
- Utilize an SKU equivalency table.
- Engineer features using techniques like TF-IDF.
- Train a classification model (e.g., Logistic Regression, Random Forest).
- Predict SKUs or equivalency groups for new bids.
- Output predictions in JSON format.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure data path:**
    - Update `config/config.yaml` to point to your data file.

## Usage

```bash
python src/main.py
```

## Project Structure

```
. 
├── .gitignore 
├── Fuente_Json_Consolidado/Consolidado.json        # Provided large data file (potentially outside repo, now in subdirectory)
├── PRD_SKU_desde_descripcion.md 
├── README.md
├── TODO_SKU_desde_descripcion.md
├── config
│   └── config.yaml       # Configuration file (e.g., data paths)
├── data                    # Sample data or processed data (if needed)
├── notebooks               # Jupyter notebooks for exploration
├── requirements.txt        # Project dependencies
├── src                     # Source code
│   ├── __init__.py
│   ├── config_loader.py  # Loads configuration
│   ├── data_loader.py    # Handles data loading
│   ├── preprocessing.py  # Text preprocessing functions
│   ├── feature_engineering.py # Feature generation
│   ├── model_training.py # Model training pipeline
│   ├── prediction.py     # Prediction logic
│   └── main.py           # Main script execution point
├── tests                   # Unit and integration tests
│   ├── __init__.py
│   └── test_*.py
└── venv/                   # Virtual environment (ignored by git)
```
