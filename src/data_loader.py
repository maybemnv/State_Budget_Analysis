import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import io
import streamlit as st

from .config import DATA_DIR, APP_CONFIG
from .utils.date_utils import parse_financial_year, detect_date_format

class DataLoader:
    """Class to handle data loading and preparation."""
    
    def __init__(self):
        self.data = None
        self.metadata = {
            'source': None,
            'columns': [],
            'date_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'shape': (0, 0),
            'missing_values': 0,
            'duplicates': 0
        }
    
    def load_data(self, source: Union[str, Path, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            source: Path to file or pandas DataFrame
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        if isinstance(source, (str, Path)):
            source = str(source)
            if source.endswith('.csv'):
                self.data = pd.read_csv(source, **kwargs)
            elif source.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(source, **kwargs)
            elif source.endswith('.parquet'):
                self.data = pd.read_parquet(source, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {source}")
            
            self.metadata['source'] = source
        elif isinstance(source, pd.DataFrame):
            self.data = source.copy()
            self.metadata['source'] = 'dataframe_input'
        else:
            raise ValueError("Source must be a file path or pandas DataFrame")
        
        self._update_metadata()
        return self.data
    
    def load_from_streamlit_uploader(self, uploaded_file, **kwargs) -> pd.DataFrame:
        """
        Load data from a Streamlit file uploader.
        
        Args:
            uploaded_file: File object from st.file_uploader()
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        if uploaded_file is None:
            return None
            
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        try:
            if file_ext == '.csv':
                self.data = pd.read_csv(uploaded_file, **kwargs)
            elif file_ext in ['.xls', '.xlsx']:
                self.data = pd.read_excel(uploaded_file, **kwargs)
            elif file_ext == '.parquet':
                self.data = pd.read_parquet(uploaded_file, **kwargs)
            else:
                st.error(f"Unsupported file format: {file_ext}")
                return None
                
            self.metadata['source'] = uploaded_file.name
            self._update_metadata()
            return self.data
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def _update_metadata(self) -> None:
        """Update metadata after loading data."""
        if self.data is None:
            return
            
        self.metadata.update({
            'columns': self.data.columns.tolist(),
            'shape': self.data.shape,
            'missing_values': self.data.isnull().sum().sum(),
            'duplicates': self.data.duplicated().sum()
        })
        
        # Detect date columns
        self.metadata['date_columns'] = [
            col for col in self.data.columns 
            if self._is_date_column(self.data[col])
        ]
        
        # Detect numeric columns
        self.metadata['numeric_columns'] = self.data.select_dtypes(
            include=['number']
        ).columns.tolist()
        
        # Categorical columns (non-numeric, non-date)
        self.metadata['categorical_columns'] = [
            col for col in self.data.columns 
            if col not in self.metadata['date_columns'] 
            and col not in self.metadata['numeric_columns']
        ]
    
    @staticmethod
    def _is_date_column(series: pd.Series, sample_size: int = 20) -> bool:
        """Check if a column contains date-like values."""
        # Skip if all values are null
        if series.isna().all():
            return False
            
        # Sample the data for efficiency
        sample = series.dropna().head(sample_size)
        if len(sample) == 0:
            return False
            
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
            
        # Try to convert to datetime
        try:
            # Check a sample for date-like strings
            date_like = pd.to_datetime(sample, errors='coerce')
            if date_like.notna().sum() / len(sample) > 0.8:  # 80% match
                return True
                
            # Check for financial year patterns
            fy_count = sample.astype(str).str.contains(
                r'\d{4}[-/]\d{2}|FY\d{2}|FY\d{4}', 
                regex=True
            ).sum()
            
            if fy_count / len(sample) > 0.8:  # 80% match
                return True
                
        except:
            pass
            
        return False
    
    def get_date_columns(self) -> List[str]:
        """Get list of date columns in the data."""
        return self.metadata.get('date_columns', [])
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns in the data."""
        return self.metadata.get('numeric_columns', [])
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns in the data."""
        return self.metadata.get('categorical_columns', [])
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the loaded data."""
        return self.metadata.copy()
    
    def get_data_summary(self) -> pd.DataFrame:
        """Generate a summary of the loaded data."""
        if self.data is None:
            return pd.DataFrame()
            
        summary = pd.DataFrame({
            'Column': self.data.columns,
            'Type': self.data.dtypes.astype(str),
            'Missing Values': self.data.isnull().sum().values,
            'Unique Values': self.data.nunique().values,
            'Sample Values': self.data.iloc[0].values
        })
        
        return summary
