import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

@step
def data_splitter_step(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "train_set"],
    Annotated[pd.DataFrame, "test_set"]
]:
    """
    Splits the data into training and testing sets.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A tuple containing the training and testing DataFrames.
    """
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    return train_df, test_df