from src.ingest_data import DataIngestor
from zenml import step
import pandas as pd

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    ZenML step for ingesting data from a file using the DataIngestor.
     Args:
        file_path: The path to the file to be ingested.
    Returns:
        A pandas DataFrame containing the ingested data.
    """
    # Get the appropriate DataIngestion object
    ingestor = DataIngestor.get_data_ingestion(file_path)

    # Ingest the data
    df = ingestor.ingest_data(file_path)
    return df