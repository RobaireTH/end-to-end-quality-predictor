import pandas as pd
from abc import ABC

class DataInsight(ABC):
    def __init__(self):
        self.df = None

class DataTypeChecker(DataInsight):
    @staticmethod
    def check_dtype(df: pd.DataFrame) -> object:
        """ Prints the datatype of the dataframe.
         :param df: pandas.DataFrame
         :returns: Object of the datatype of the dataframe"""
        print(f"Data Types: \n{df.dtypes}")


class DataShapeChecker(DataInsight):
    @staticmethod
    def check_shape(df: pd.DataFrame) -> tuple:
        """ Prints the shape of the dataframe.
         :param df: pandas.DataFrame
         :returns:a tuple containing the shape of the dataframe"""
        return df.shape


class StatisticsSummary(DataInsight):
    @staticmethod
    def stats(df: pd.DataFrame):
        """Prints the statistics of the dataframe. Describing both Numerical and Categorical statistics.
        :param df: pandas.DataFrame
        :returns: Both numerical and ~categorical statistics~"""
        print(f"Numerical Statistics: \n{df.describe()}")
        # No categorical feature
       # print(f"Categorical Statistics: \n{df.describe(include=["O"])}")
        
class Inspection(DataInsight):
    @staticmethod
    def inspect(df: pd.DataFrame) -> str:
        """Shows Data Info
        :param df: pandas.DataFrame
        :returns: pandas.DataFrame.info() """
        return f"Data Information: \n{df.info()}"