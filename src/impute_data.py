import pandas as pd
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore")

class Injector(ABC):
    @abstractmethod
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Receives a dataframe and returns a dataframe with missing values imputed using median or mode imputation.
        Mean imputation is susceptible to outliers. """
        pass
    
    @abstractmethod
    def drop_duplicated(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Drops duplicated rows. """
        pass

class DataInjector(Injector):
    def handle_missing_values(self,df: pd.DataFrame) -> pd.DataFrame:
        """ Performs missing value imputation on dataframe.
        :param df: pandas.DataFrame to be imputed.
       :returns: a dataframe with missing values imputed using median or mode imputation.
        """
        for column in df.columns:
            null_sum = df[column].isnull().sum()
            null_percent = null_sum / len(df)
            # Drop columns with 75% missing values
            if null_percent > 0.75:
                df.drop([column], axis=1, inplace=True)
            else:
                # Impute NaN values with either mode or mean values.
                """ Using mode fill for categorical columns is to handle multi-class columns """
                if df[column].dtype.name  == "object" or df[column].dtype.name == "category":
                    df[column].fillna(df[column].mode()[0], inplace = True)
                elif df[column].dtype.name  == "float64" or df[column].dtype.name == "int64":
                    df[column].fillna(df[column].median(), inplace = True)
        return df
            
    def drop_duplicated(self, df: pd.DataFrame) -> pd.DataFrame:
        """
         :param df:pandas.DataFrame to be imputed.
         :returns: a dataframe with duplicated data points dropped.
         """
        df.drop_duplicates(inplace=True)
        return df