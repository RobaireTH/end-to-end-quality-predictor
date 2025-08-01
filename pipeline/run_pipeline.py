from steps.data_injector_step import data_imputation_step
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.model_builder_step import model_builder_step
from steps.model_evaluator_step import model_evaluator
from sklearn.linear_model import LinearRegression

from zenml import Model, pipeline
from zenml import logger

logger.get_logger(__name__)

"""
The Feature Engineering Step Was Not Implemented In This Project
"""

# Define the model configuration
model_config = Model(
    name="data_pipeline_model",
    license="Apache-2.0",
    description="Model for handling data ingestion, imputation, and splitting.",
    version = "v0.1",
    limitations = None
)

@pipeline(model=model_config, name="data_pipeline")
def ml_pipeline():
    """
    ZenML pipeline for data ingestion, imputation, and splitting.
    """
    # Step 1: Ingest data from a zip file
    df = data_ingestion_step(file_path="../data/wine.zip")

    # Step 2: Impute and clean data
    df_cleaned = data_imputation_step(df=df)

    # Step 3: Split data into training and testing sets
    train_df, test_df = data_splitter_step(df=df_cleaned)
    
    # Step 4: Building and returning the Classifier model
    trained_model = model_builder_step(train_df=train_df, target="Class", model_type="linear_regression")
    
    # Step 5: Evaluating model performance
    report =  model_evaluator(trained_model, test_df, target= "Class")
    return report
    
if __name__ == "__main__":
    ml_pipeline()