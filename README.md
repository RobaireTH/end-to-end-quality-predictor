# ZenML Wine Quality Prediction Pipeline

This project demonstrates a complete, end-to-end machine learning pipeline built with [ZenML](https://zenml.io/). The pipeline ingests wine quality data, preprocesses it, trains a model (supporting Random Forest, SVC, or Linear Regression), and evaluates its performance.

## Features

*   **Modular Pipeline**: Built with ZenML for reproducible and maintainable ML workflows.
*   **Data Ingestion**: Loads data directly from a compressed `.zip` file.
*   **Data Preprocessing**: Includes steps for data cleaning, imputation, and splitting.
*   **Flexible Model Training**: Easily switch between `RandomForestClassifier`, `SVC`, and `LinearRegression` models.
*   **Model Evaluation**: Generates a performance report on the test set.

! [Pipeline Diagram](Screenshot from 2025-08-01 11-49-36.png)
## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    It is recommended to create a `requirements.txt` file with the following content:
    ```
    zenml
    scikit-learn
    pandas
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the machine learning pipeline, execute the main pipeline script:

```bash
python pipeline/run_pipeline.py
```

### Selecting a Model

You can change the model being trained by modifying the `model_type` argument in the `model_builder_step` call within `pipeline/run_pipeline.py`.

Supported values for `model_type` are:
*   `"random_forest"`
*   `"svc"`
*   `"linear_regression"`

**Example:** To use the Random Forest model, ensure the line looks like this:

```python
# In pipeline/run_pipeline.py
trained_model = model_builder_step(train_df=train_df, target="Class", model_type="random_forest")
```

## Pipeline Steps

The pipeline consists of the following ZenML steps:

1.  **`data_ingestion_step`**: Reads the dataset from `../data/wine.zip` into a pandas DataFrame.
2.  **`data_imputation_step`**: Cleans the data and handles any missing values.
3.  **`data_splitter_step`**: Splits the cleaned DataFrame into training and testing sets.
4.  **`model_builder_step`**: Selects a model builder based on the `model_type` parameter, trains the model on the training data, and returns the trained model artifact.
5.  **`model_evaluator_step`**: Evaluates the trained model on the test set and returns a classification/evaluation report.
