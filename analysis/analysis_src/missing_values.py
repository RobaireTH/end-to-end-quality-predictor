from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MissingValuesTemplate(ABC):
    @abstractmethod
    def check_missing_values(self) -> pd.DataFrame:
        """ This method checks that the missing values are present in the dataframe.
        :return: A pandas DataFrame with the count of missing values per column.
                 Returns an empty DataFrame if no missing values are found.
        """
        pass
    
    @abstractmethod
    def visualize_missing_values(self, missing_df: pd.DataFrame) -> None:
        """ This method visualizes the missing values in the dataframe.
        :param missing_df: A DataFrame containing the count of missing values per column.
                           This DataFrame is the output of check_missing_values.
        :returns: None (displays plots)
        """
        pass
    
    @abstractmethod
    def summary(self) -> None:
        """ Prints the summary of the missing data statistics."""
        pass


class MissingDataVisualizer(MissingValuesTemplate):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def check_missing_values(self) -> pd.DataFrame:
        """ This method checks the missing values that are present in a dataframe.
        :returns: a pandas DataFrame with the sum of missing values for columns that have them.
                  If no missing values are found, an empty DataFrame is returned.
        """
        # Calculate the sum of nulls for each column
        missing_counts = self.df.isnull().sum()
        
        # Filter to only include columns with missing values (count > 0)
        df_missing = pd.DataFrame(missing_counts[missing_counts > 0])
        
        if df_missing.empty:
            print("No missing values present in the dataframe.")
            # Return an empty DataFrame, which is clear for subsequent checks
            return pd.DataFrame(columns=['Missing Count'])  # Explicitly create with a column name
        else:
            df_missing.columns = ['Missing Count']  # Rename the column
            return df_missing.sort_values(by='Missing Count', ascending=False)
    
    def visualize_missing_values(self, missing_df: pd.DataFrame) -> None:
        """ This method visualizes the missing values in the dataframe.
        :param missing_df: A DataFrame containing the count of missing values per column.
                           This DataFrame is the output of check_missing_values.
        :returns: None (displays visuals of the missing values using both bar chart and heatmap)
        """
        if missing_df.empty:
            print("No missing values to visualize.")
            return
        
        # Bar Plot: Shows count of missing values per column
        plt.figure(figsize=(12, 6))
        missing_df.plot(kind='bar', ax=plt.gca(), legend=False, color='skyblue')
        plt.title('Missing Values Per Column', fontsize=16)
        plt.ylabel('Number of Missing Values', fontsize=12)
        plt.xlabel('Features', fontsize=12)
        plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Heatmap: Shows the pattern of missing values across the original DataFrame
        plt.figure(figsize=(12, 8))
        # Use isnull() on the original DataFrame to show where missing values occur
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title("Missing Data Pattern Heatmap", fontsize=16)
        plt.xlabel("Columns", fontsize=12)
        plt.ylabel("Rows (Presence of Missing Data)", fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def summary(self) -> None:
        """ Prints the summary of the missing data statistics."""
        total_missing = self.df.isnull().sum()
        percentage_missing = (total_missing / len(self.df)) * 100
        
        missing_info = pd.DataFrame({"Total Missing": total_missing, "Percentage": percentage_missing})
        
        # Filter for columns that actually have missing values
        missing_info = missing_info[missing_info['Total Missing'] > 0].sort_values(by='Percentage', ascending=False)
        
        if missing_info.empty:
            print("No missing values found in the dataframe.")
        else:
            print("\n--- Summary of Missing Data ---")
            print(missing_info.to_string()  ) # Use to_string() for full display
            print("-------------------------------\n")


# Use Case
if __name__ == "__main__":
    # Sample DataFrame with various missing data scenarios
    data = {
        'Feature_A': [1, 2, None, 4, 5, None, 7, 8, 9, 10],
        'Feature_B': [None, 7, 8, None, 10, 11, 12, None, 14, 15],
        'Feature_C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        ], # No missing values
        'Feature_D': [None, None, None, None, None, None, None, None, None, None], # All missing
        'Feature_E': [21, 22, 23, None, 25, 26, 27, 28, 29, 30]
    }
    df_with_missing = pd.DataFrame(data)
    
    print("--- Original DataFrame with Missing Values ---")
    print(df_with_missing)
    print("\n" + "="*50 + "\n")
    
    # Initialize the visualizer
    visualizer_missing = MissingDataVisualizer(df_with_missing)
    
    # 1. Check missing values
    missing_counts_df = visualizer_missing.check_missing_values()
    print("\n--- Result of check_missing_values() ---")
    print(missing_counts_df)
    print("\n" + "= "*50 + "\n")
    
    # 2. Visualize missing values
    print("--- Visualizing Missing Values ---")
    visualizer_missing.visualize_missing_values(missing_counts_df)
    print("\n" + " = "*50 + "\n")
    
    # 3. Print summary
    visualizer_missing.summary()
    print("\n" + " = "*50 + "\n")
    
    # --- Example with NO Missing Values ---
    df_no_missing = pd.DataFrame({
        'Col_X': [10, 20, 30],
        'Col_Y': [40, 50, 60]
    })
    print("--- Original DataFrame with NO Missing Values ---")
    print(df_no_missing)
    print("\n" + " = "*50 + "\n")
    
    no_missing_visualizer = MissingDataVisualizer(df_no_missing)
    
    # Check
    no_missing_counts = no_missing_visualizer.check_missing_values()
    print("\n--- Result of check_missing_values() (No Missing) ---")
    print(no_missing_counts  ) # This will be an empty DataFrame
    print("\n" + "= "*50 + "\n")
    
    # Visualize
    print("--- Visualizing Missing Values (No Missing) ---")
    no_missing_visualizer.visualize_missing_values(no_missing_counts)
    print("\n" + " = "*50 + "\n")
    
    # Summary
    no_missing_visualizer.summary()
    print("\n" + " = "*50 + "\n")
    
    # --- Example with ALL Missing Values ---
    df_all_missing = pd.DataFrame({
        'A': [None, None],
        'B': [None, None]
    })
    print("--- Original DataFrame with ALL Missing Values ---")
    print(df_all_missing)
    print("\n" + " = "*50 + "\n")
    
    all_missing_visualizer = MissingDataVisualizer(df_all_missing)
    all_missing_counts = all_missing_visualizer.check_missing_values()
    print("\n--- Result of check_missing_values() (All Missing) ---")
    print(all_missing_counts)
    print("\n" + " = "*50 + "\n")
    
    all_missing_visualizer.visualize_missing_values(all_missing_counts)
    all_missing_visualizer.summary()
    print("\n" + " = "*50 + "\n")