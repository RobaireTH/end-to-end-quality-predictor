import pandas as pd
from zenml import step
from src.impute_data import DataInjector # Make sure the path is correct

@step
def data_imputation_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML step for cleaning and imputing data using the DataInjector.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A cleaned and imputed pandas DataFrame.
    """
    data_injector = DataInjector()
    df_imputed = data_injector.handle_missing_values(df)
    df_cleaned = data_injector.drop_duplicated(df_imputed)
    return df_cleaned

