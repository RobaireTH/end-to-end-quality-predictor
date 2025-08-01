from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import andrews_curves, parallel_coordinates, radviz
from typing import Optional


class MultivariateAnalysisTemplate(ABC):
    @abstractmethod
    def plot(self,df: pd.DataFrame, target: Optional[str]):
        """ This method is used to plot the relationship between multivariate features."""
        pass

class HeatMap(MultivariateAnalysisTemplate):
    def plot(self,df: pd.DataFrame, target: Optional[str]):
        """ This method is used to plot the relationship between multivariate features.
        :param df:
        :param target:
        :return: visual showing the relationship between multivariate features (Heatmap)."""
        plt.figure(figsize=(12,8))
        sns.heatmap(df.corr(), cmap="YlGnBu")
        plt.xticks(rotation=90)
        plt.title("HeatMap")
        plt.show()
        
class PairPlot(MultivariateAnalysisTemplate):
    def plot(self,df: pd.DataFrame, target: Optional[str]):
        """ This method is used to plot the relationship between multivariate features.
        :param df:
        :param target:
        :return: visual showing the relationship between multivariate features (PairPlot)."""
        plt.figure(figsize=(12,8))
        sns.pairplot(df, hue=target, palette="Set2")
        plt.title("Pair Plot")
        plt.show()
        
class AndrewsCurve(MultivariateAnalysisTemplate):
    def plot(self,df: pd.DataFrame, target: Optional[str]):
        """ This method is used to plot the relationship between multivariate features.
        :param df:
        :param target:
        :return: visual showing the relationship between multivariate features (AndrewsCurve)."""
        plt.figure(figsize=(12,8))
        andrews_curves(df, target, colormap=plt.get_cmap("jet"))
        plt.title("Andrews Curve")
        plt.show()
        
class ParallelPlot(MultivariateAnalysisTemplate):
    def plot(self,df: pd.DataFrame, target: Optional[str]):
        """ This method is used to plot the relationship between multivariate features.
        :param df:
        :param target:
        :return: visual showing the relationship between multivariate features (ParallelPlot)."""
        plt.figure(figsize=(12,8))
        parallel_coordinates(df, target, colormap=plt.get_cmap("jet"))
        plt.xticks(rotation=90)
        plt.title("Parallel Plot")
        plt.show()

class RadViz(MultivariateAnalysisTemplate):
    def plot(self,df: pd.DataFrame, target: Optional[str]):
        """ This method is used to plot the relationship between multivariate features.
        :param df:
        :param target:
        :return: visual showing the relationship between multivariate features (Radviz)."""
        plt.figure(figsize=(12,8))
        radviz(df, target, colormap=plt.get_cmap("jet"))
        plt.title("Radviz")
        plt.show()
    
