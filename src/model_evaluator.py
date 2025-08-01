from abc import ABC, abstractmethod

from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.base import BaseEstimator
import pandas as pd
from typing_extensions import Annotated
from sklearn.metrics import accuracy_score

class ModelEvaluatorTemplate(ABC):
    @abstractmethod
    def evaluate_model(self, model: BaseEstimator, test_df: pd.DataFrame, target: str) -> Annotated[str, int, float]:
        """This outputs the evaluation metrics of the model"""
        pass
    
class ModelEvaluator(ModelEvaluatorTemplate):
    def evaluate_model(self, model: BaseEstimator, test_df:pd.DataFrame, target: str):
        """ This outputs the evaluation metrics of the model
        Args:
            model: The trained model
            test_df: The test data
            target: The target column
        Returns:
            The evaluation metrics of the model."""
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]
        
        y_pred = model.predict(X_test)
        
        acc  = model.score(X_test, y_test)

        return acc