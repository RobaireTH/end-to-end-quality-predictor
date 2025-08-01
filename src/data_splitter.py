from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataSplitter(ABC):
    @abstractmethod
    def split_data(self, df, target: str) -> None:
        """This method splits the data into train and test sets, and must be implemented by the subclasses"""
        pass
    
class TrainTestSplitter(DataSplitter):
    def split_data(self, df, target: str) -> Tuple[pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series]:
        """
        Splits the data into train and test sets.
        :param df:  pandas.DataFrame to be split
        :param target: target column
        :return: X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target])
        y = df[target]
        # Splitting into training and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test