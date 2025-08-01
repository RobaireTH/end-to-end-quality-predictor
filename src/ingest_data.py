from abc import ABC, abstractmethod
import pandas as pd
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

class DataIngestion(ABC):
    @abstractmethod
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        pass

class ZipDataIngestion(DataIngestion):
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a zip file.
         :param file_path: path to the zip file
         :type file_path: str
         :return: a dataframe with the data loaded from the CSV file inside the zip.
        """
        if not file_path.endswith(".zip"):
            raise ValueError("This ingestor only supports .zip files.")

        # Create a directory for extracted files relative to this script
        script_dir = os.path.dirname(__file__)
        extract_dir = os.path.join(script_dir, "extracted_data")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        extracted_files = os.listdir(extract_dir)
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if not csv_files:
             raise FileNotFoundError("No CSV file found in the zip archive.")
        if len(csv_files) > 1:
             raise ValueError("Multiple CSV files found; please specify which one to use.")

        csv_file_path = os.path.join(extract_dir, csv_files[0])
        df = pd.read_csv(csv_file_path)
        return df

class DataIngestor:
     @staticmethod
     def get_data_ingestion(file_path: str) -> DataIngestion:
         """:return: The appropriate DataIngestion object for the given file path.
         :param file_path: path to the file
         :type file_path: str
         """
         if file_path.endswith(".zip"):
             return ZipDataIngestion()
         # You could add more handlers here, e.g., for .csv
         # elif file_path.endswith(".csv"):
         #     return CsvDataIngestion()
         else:
             raise ValueError(f"No Ingestor available for the file type: {file_path}")

# Use Case
if __name__ == '__main__':
    file_path = "../data/wine.zip"  # Assuming data folder is sibling to src
    try:
        ingestor = DataIngestor.get_data_ingestion(file_path)
        df1 = ingestor.ingest_data(file_path)
        print(df1.head())
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")