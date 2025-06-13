import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceDataLoader:
    """
    A class to handle loading and preprocessing of insurance data with DVC support.
    """
    
    def __init__(self, raw_data_path: str, processed_data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            raw_data_path (str): Path to the raw data file
            processed_data_path (str, optional): Path where processed data will be saved
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path) if processed_data_path else None
        self.raw_data = None
        self.processed_data = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw pipe-delimited data and convert to DataFrame.
        
        Returns:
            pd.DataFrame: Raw data as DataFrame
        """
        try:
            logger.info(f"Loading data from {self.raw_data_path}")
            
            # Check file extension to determine delimiter
            if self.raw_data_path.suffix.lower() == '.csv':
                # For CSV files, try comma first, then pipe
                try:
                    self.raw_data = pd.read_csv(
                        self.raw_data_path,
                        encoding='utf-8',
                        low_memory=False
                    )
                except:
                    # Fallback to pipe delimiter
                    self.raw_data = pd.read_csv(
                        self.raw_data_path,
                        delimiter='|',
                        encoding='utf-8',
                        low_memory=False
                    )
            else:
                # For other file types, use pipe delimiter
                self.raw_data = pd.read_csv(
                    self.raw_data_path, 
                    delimiter='|',
                    encoding='utf-8',
                    low_memory=False
                )
            
            logger.info(f"Successfully loaded {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning including duplicate removal and missing value handling.
        
        Args:
            df (pd.DataFrame): Input dataframe to clean
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Log missing values summary
        missing_summary = df_cleaned.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        if len(missing_cols) > 0:
            logger.info(f"Missing values summary:\n{missing_cols}")
        else:
            logger.info("No missing values found")
        
        logger.info("Data cleaning completed")
        return df_cleaned
    
    def basic_preprocessing(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Perform basic preprocessing on the data.
        
        Args:
            df (pd.DataFrame, optional): Input dataframe. If None, uses self.raw_data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data provided and no raw data loaded. Please call load_raw_data() first or provide a dataframe.")
            df = self.raw_data
        
        logger.info("Starting basic preprocessing...")
        
        # First clean the data
        self.processed_data = self.clean_data(df)
        
        # Convert TransactionMonth to datetime
        if 'TransactionMonth' in self.processed_data.columns:
            self.processed_data['TransactionMonth'] = pd.to_datetime(
                self.processed_data['TransactionMonth'], 
                errors='coerce'
            )
            logger.info("Converted TransactionMonth to datetime")
        
        # Convert numeric columns
        numeric_columns = [
            'TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm',
            'CustomValueEstimate', 'CapitalOutstanding', 'RegistrationYear',
            'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors',
            'NumberOfVehiclesInFleet'
        ]
        
        converted_cols = []
        for col in numeric_columns:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_numeric(
                    self.processed_data[col], 
                    errors='coerce'
                )
                converted_cols.append(col)
        
        if converted_cols:
            logger.info(f"Converted to numeric: {converted_cols}")
        
        logger.info("Basic preprocessing completed")
        return self.processed_data
    
    def save_processed_data(self, df: Optional[pd.DataFrame] = None, output_path: Optional[str] = None) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            df (pd.DataFrame, optional): Dataframe to save. If None, uses self.processed_data
            output_path (str, optional): Path where file will be saved. If None, uses self.processed_data_path
        """
        # Determine which dataframe to save
        if df is None:
            if self.processed_data is None:
                raise ValueError("No processed data available. Please run basic_preprocessing() first or provide a dataframe.")
            df_to_save = self.processed_data
        else:
            df_to_save = df
        
        # Determine output path
        if output_path is not None:
            save_path = Path(output_path)
        elif self.processed_data_path is not None:
            save_path = self.processed_data_path
        else:
            raise ValueError("No output path specified. Provide output_path parameter or set processed_data_path in constructor.")
        
        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            df_to_save.to_csv(save_path, index=False)
            logger.info(f"Processed data saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def save_as_csv(self, output_path: str) -> None:
        """
        Save the raw loaded data as CSV file (legacy method for backward compatibility).
        
        Args:
            output_path (str): Path where CSV file will be saved
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Please call load_raw_data() first.")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.raw_data.to_csv(output_path, index=False)
            logger.info(f"Raw data saved as CSV to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            raise
    
    def process_data(self) -> pd.DataFrame:
        """
        Main data processing pipeline - loads, processes, and saves data.
        
        Returns:
            pd.DataFrame: Processed data
        """
        logger.info("Starting data processing pipeline...")
        
        # Load raw data
        self.load_raw_data()
        
        # Process data
        processed_df = self.basic_preprocessing()
        
        # Save processed data if path is specified
        if self.processed_data_path is not None:
            self.save_processed_data()
        
        logger.info("Data processing pipeline completed")
        return processed_df
    
    def get_data_info(self, data_type: str = 'raw') -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Args:
            data_type (str): Type of data to analyze ('raw' or 'processed')
            
        Returns:
            Dict: Dictionary containing data information
        """
        if data_type == 'raw':
            if self.raw_data is None:
                raise ValueError("No raw data loaded. Please call load_raw_data() first.")
            df = self.raw_data
        elif data_type == 'processed':
            if self.processed_data is None:
                raise ValueError("No processed data available. Please call basic_preprocessing() first.")
            df = self.processed_data
        else:
            raise ValueError("data_type must be 'raw' or 'processed'")
        
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicates': df.duplicated().sum()
        }

# Example usage for DVC pipeline
if __name__ == "__main__":
    loader = InsuranceDataLoader(
        raw_data_path="data/raw/MachineLearningRating_v3.txt",  # or .csv
        processed_data_path="data/processed/cleaned_insurance_data.csv"
    )
    
    # Run the complete pipeline
    processed_data = loader.process_data()
    
    # Or run steps individually
    # loader.load_raw_data()
    # loader.basic_preprocessing()
    # loader.save_processed_data()