import pandas as pd
from zenml import step
from src.model_building import (
    LinearRegressionBuilder,
    RandomForestBuilder,
    SVCBuilder,
)
from sklearn.base import BaseEstimator
from typing_extensions import Annotated


@step
def model_builder_step(
    train_df: pd.DataFrame,
    target: str,
    model_type: str = "random_forest",
) -> Annotated[BaseEstimator, "trained_model"]:
    """
    ZenML step for building a model.

    Args:
        train_df: The training DataFrame.
        target: The name of the target column.
        model_type: The type of model to build ('random_forest', 'svc', 'linear_regression').

    Returns:
        The trained model.
    """
    if model_type == "random_forest":
        builder = RandomForestBuilder(train_df=train_df, target=target)
    elif model_type == "svc":
        builder = SVCBuilder(train_df=train_df, target=target)
    elif model_type == "linear_regression":
        builder = LinearRegressionBuilder(train_df=train_df, target=target)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    trained_model = builder.build()
    return trained_model