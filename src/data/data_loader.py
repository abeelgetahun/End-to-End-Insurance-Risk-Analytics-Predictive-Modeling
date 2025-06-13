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
    A class to handle loading and preprocessing of insurance data.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the raw data file
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw pipe-delimited data and convert to DataFrame.
        
        Returns:
            pd.DataFrame: Raw data as DataFrame
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            # Read pipe-delimited file
            self.raw_data = pd.read_csv(
                self.data_path, 
                delimiter='|',
                encoding='utf-8',
                low_memory=False
            )
            
            logger.info(f"Successfully loaded {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def save_as_csv(self, output_path: str) -> None:
        """
        Save the loaded data as CSV file.
        
        Args:
            output_path (str): Path where CSV file will be saved
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Please call load_raw_data() first.")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.raw_data.to_csv(output_path, index=False)
            logger.info(f"Data saved as CSV to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dict: Dictionary containing data information
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Please call load_raw_data() first.")
        
        return {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'dtypes': self.raw_data.dtypes.to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'memory_usage': self.raw_data.memory_usage(deep=True).sum()
        }
    
    def basic_preprocessing(self) -> pd.DataFrame:
        """
        Perform basic preprocessing on the data.
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Please call load_raw_data() first.")
        
        logger.info("Starting basic preprocessing...")
        self.processed_data = self.raw_data.copy()
        
        # Convert TransactionMonth to datetime
        if 'TransactionMonth' in self.processed_data.columns:
            self.processed_data['TransactionMonth'] = pd.to_datetime(
                self.processed_data['TransactionMonth'], 
                errors='coerce'
            )
        
        # Convert numeric columns
        numeric_columns = [
            'TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm',
            'CustomValueEstimate', 'CapitalOutstanding', 'RegistrationYear',
            'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors',
            'NumberOfVehiclesInFleet'
        ]
        
        for col in numeric_columns:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_numeric(
                    self.processed_data[col], 
                    errors='coerce'
                )
        
        logger.info("Basic preprocessing completed")
        return self.processed_data