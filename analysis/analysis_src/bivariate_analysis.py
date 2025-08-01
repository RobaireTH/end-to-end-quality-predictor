from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional

class BivariateAnalysisTemplate(ABC):
    @abstractmethod
    def plot(self, feature1: str, feature2: str, target: Optional[str]) -> None:
        """ This produces a plot showing the relationship between the two features/columns
        :param feature1:
        :param feature2:
        :param target: None
        :returns: None (displays plot)"""
        pass

    
class NumericalVSNumericalAnalysis(BivariateAnalysisTemplate):
    """" Performs Bivariate Analysis on Numerical Data"""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def plot(self, feature1: str, feature2: str, target: str) -> None:
        """ This produces a scatter plot and a KDE plot showing the relationship between the two features/columns
        :param feature1:
        :param feature2:
        :param target:
        :returns:  displays scatter plot"""
        plt.figure(figsize=(12,8))
        sns.scatterplot(x=self.df[feature1], y=self.df[feature2], hue=self.df[target], palette="Set2", alpha= 0.3)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"Scatter Plot \n{feature1.capitalize()} vs {feature2.capitalize()}")
        plt.show()
        
        plt.figure(figsize=(12,8))
        sns.kdeplot(x=self.df[feature1],y=self.df[feature2], fill=True, cmap="Blues")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"KDE Plot \n{feature1.capitalize()} vs {feature2.capitalize()}")
        plt.show()
        return None
        
class CategoricalVSNumericalAnalysis(BivariateAnalysisTemplate):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def plot(self, feature1: str, feature2: str, target: str) -> None:
        """ This produces a bar plot and a violin  plot showing the relationship between the two features/columns
        :param feature1:The Categorical feature
        :param feature2:The Numerical feature
        :param target:
        :returns:  displays a bar plot and a violin  plot"""
        plt.figure(figsize=(12,8))
        sns.barplot(self.df, x=feature1, y=feature2, hue=target, palette="muted")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"Bar Plot \n{feature1.capitalize()} vs {feature2.capitalize()}")
        plt.show()
        
        plt.figure(figsize=(12,8))
        sns.violinplot(self.df, x=feature1, y=feature2, hue=self.df[target], palette="Set2")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"Violin Plot \n{feature1.capitalize()} vs {feature2.capitalize()}")
        plt.show()
        return None