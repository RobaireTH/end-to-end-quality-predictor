from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysisTemplate(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    @abstractmethod
    def plot_distribution(self, column: str) -> None:
        """ This method plots the distribution of a column in the dataframe
         :param column: name of the feature/column to plot
         :return: None"""
        pass
    
class NumericalUnivariateAnalyzer(UnivariateAnalysisTemplate):
    """Visualize the distribution of numerical variables in an univariate analysis template"""
    def plot_distribution(self, column: str) -> None:
        """ This method plots the distribution of a column in the dataframe
        :param column: name of the feature/column to plot
        :return: a visual plot of the distribution"""
        plt.figure(figsize=(12,8))
        sns.histplot(self.df[column], kde=True)
        plt.xticks(rotation=90)
        plt.xlabel(column.capitalize())
        plt.title("Distribution of numerical variables in {}".format(column.capitalize()))
        plt.show()
        
class CategoricalUnivariateAnalyzer(UnivariateAnalysisTemplate):
    """Visualize the distribution of categorical variables in an univariate analysis template"""
    def plot_distribution(self, column: str) -> None:
        """ This method plots the distribution of a column in the dataframe
        :param column: name of the feature/column to plot
        :return: a visual plot of the distribution"""
        plt.figure(figsize=(12,8))
        sns.countplot(x=self.df[column], hue=self.df[column], palette="Set1")
        plt.xticks(rotation=90)
        plt.title("Distribution of categorical variables in {}".format(column.capitalize()))
        plt.show()