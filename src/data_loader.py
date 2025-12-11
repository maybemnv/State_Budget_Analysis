import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import io
import base64
import streamlit as st

from .config import DATA_DIR, APP_CONFIG
from .utils.date_utils import parse_financial_year, detect_date_format

class DataLoader:
    """Class to handle data loading and preparation."""
    
    def clean_data(self):
        """
        Perform basic data cleaning operations on the loaded data.
        - Trims whitespace from string columns
        - Drops fully empty rows and columns
        - Resets index
        """
        if self.data is not None:
            # 1. Trim whitespace from string columns
            df_obj = self.data.select_dtypes(['object'])
            self.data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip().replace(r'^\s*$', np.nan, regex=True))
            
            # 2. Drop rows and columns that are completely empty
            self.data.dropna(how='all', axis=0, inplace=True)
            self.data.dropna(how='all', axis=1, inplace=True)
            
            # 3. Reset index
            self.data.reset_index(drop=True, inplace=True)
            
    def __init__(self):
        """Initialize the DataLoader class."""
        self._init_session_state()
        
    def _init_session_state(self):
        """Initialize session state variables if they don't exist."""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'filename' not in st.session_state:
            st.session_state.filename = None
        if 'metadata' not in st.session_state:
            st.session_state.metadata = {
                'source': None,
                'columns': [],
                'date_columns': [],
                'numeric_columns': [],
                'categorical_columns': [],
                'shape': (0, 0),
                'missing_values': 0,
                'duplicates': 0
            }
            
    @property
    def data(self):
        return st.session_state.data
        
    @data.setter
    def data(self, value):
        st.session_state.data = value
        
    @property
    def metadata(self):
        return st.session_state.metadata

    def create_upload_widget(self):
        """
        Create a file uploader widget for CSV and Excel files.
        
        Returns:
            uploaded_file: The uploaded file object or None
        """
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
        return uploaded_file
    
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
                st.session_state.filename = Path(source).name
            elif source.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(source, **kwargs)
                st.session_state.filename = Path(source).name
            elif source.endswith('.parquet'):
                self.data = pd.read_parquet(source, **kwargs)
                st.session_state.filename = Path(source).name
            else:
                raise ValueError(f"Unsupported file format: {source}")
            
            self.metadata['source'] = source
        elif hasattr(source, 'read') and hasattr(source, 'name'):
             # Assume it is a Streamlit UploadedFile or file-like object
             return self.load_from_streamlit_uploader(source, **kwargs) is not None
        else:
            raise ValueError("Source must be a file path, pandas DataFrame, or file-like object")
        
        self.clean_data()
        self._update_metadata()
        return True # Return success boolean for compatibility
    
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
            
            self.clean_data()
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
    
    def get_data(self):
        """
        Get the currently loaded data.
        
        Returns:
            pandas.DataFrame: The loaded data or None if no data is loaded
        """
        return self.data
    
    def get_filename(self):
        """
        Get the filename of the currently loaded data.
        
        Returns:
            str: The filename or None if no file is loaded
        """
        return st.session_state.filename
        
    def get_data_info(self):
        """
        Get basic information about the loaded data (API compatibility wrapper).
        """
        if self.data is not None:
            df = self.data
            
            # Calculate memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            if memory_usage < 1024:
                memory_str = f"{memory_usage} bytes"
            elif memory_usage < 1024**2:
                memory_str = f"{memory_usage/1024:.2f} KB"
            else:
                memory_str = f"{memory_usage/(1024**2):.2f} MB"
            
            return {
                "filename": st.session_state.filename,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage": memory_str,
                "missing_values": df.isnull().sum().sum()
            }
        return None

    def get_column_info(self):
        """
        Get detailed information about each column (API compatibility wrapper).
        """
        if self.data is not None:
            df = self.data
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique = df[col].nunique()
                missing = df[col].isnull().sum()
                missing_pct = (missing / len(df)) * 100
                
                col_info.append({
                    "Column": col,
                    "Data Type": dtype,
                    "Unique Values": unique,
                    "Missing Values": missing,
                    "Missing (%)": f"{missing_pct:.2f}%"
                })
            return col_info
        return None

    def create_download_link(self, processed_df=None, filename=None):
        """
        Create a download link for the data.
        
        Args:
            processed_df (pandas.DataFrame, optional): A processed dataframe to download.
                If None, the original data will be used.
            filename (str, optional): Filename for the download.
                
        Returns:
            str: HTML link for downloading the data
        """
        if processed_df is None:
            processed_df = self.data
        
        if processed_df is not None:
            csv = processed_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            
            if filename is None:
                orig_name = self.metadata.get('source', 'data')
                if isinstance(orig_name, str):
                    filename = Path(orig_name).stem
                else:
                    filename = "data"
            
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}_processed.csv">Download CSV File</a>'
        return None
