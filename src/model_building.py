# src/model_building.py
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


class ModelBuilder(ABC):
    """Abstract base class for model builders."""
    def __init__(self, train_df: pd.DataFrame, target: str):
        """
        Initializes the ModelBuilder.
        Args:
            train_df: The training DataFrame.
            target: The name of the target column.
        """
        self.train_df = train_df
        self.target = target
        self.model: BaseEstimator

    @abstractmethod
    def build(self) -> BaseEstimator:
        """
        Builds and trains the model, then returns it.
        This method must be implemented by subclasses.
        """
        pass


class RandomForestBuilder(ModelBuilder):
    """Builds a RandomForestClassifier."""

    def build(self) -> BaseEstimator:
        self.model = RandomForestClassifier(n_jobs=-1, random_state=42)
        X_train = self.train_df.drop(self.target, axis=1)
        y_train = self.train_df[self.target]
        self.model.fit(X_train, y_train)
        return self.model


class SVCBuilder(ModelBuilder):
    """Builds an SVC model."""
    def build(self) -> BaseEstimator:
        self.model = SVC(random_state=42)
        X_train = self.train_df.drop(self.target, axis=1)
        y_train = self.train_df[self.target]
        self.model.fit(X_train, y_train)
        return self.model


class LinearRegressionBuilder(ModelBuilder):
    """Builds a LinearRegression model."""
    def build(self) -> BaseEstimator:
        self.model = LinearRegression()
        X_train = self.train_df.drop(self.target, axis=1)
        y_train = self.train_df[self.target]
        self.model.fit(X_train, y_train)
        return self.model