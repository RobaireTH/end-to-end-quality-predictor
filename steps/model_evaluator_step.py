from zenml import step
from src.model_evaluator import ModelEvaluator
from sklearn.base import BaseEstimator
import pandas as pd

@step
def model_evaluator(model, test_df: pd.DataFrame, target: str):
    """ZenML step to evaluate performance."""
    evaluator = ModelEvaluator()
    accuracy = evaluator.evaluate_model(model, test_df, target)
    return accuracy