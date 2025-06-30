"""
File parsers for spectroscopy data import wizard.
Handles .dat (binary spectrum), .hdr (ASCII metadata), and .csv (tabular data) files.
"""

import numpy as np
import pandas as pd
import struct
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Custom exception for file parsing errors."""
    pass


class FileParser(ABC):
    """Abstract base class for file parsers."""
    
    @abstractmethod
    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse file and return structured data."""
        pass
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate parsed data structure."""
        pass


class HdrFileParser(FileParser):
    """Parser for ASCII header files with 'keyword = value' format."""
    
    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse .hdr file into metadata dictionary."""
        if not Path(filepath).exists():
            raise ParseError(f"Header file not found: {filepath}")
        
        metadata = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments (starting with '/')
                    if not line or line.startswith('/'):
                        continue
                    
                    # Handle 'continue' entries that extend previous values
                    if line.startswith('continue='):
                        key = 'continue'
                        value = line.split('=', 1)[1].strip()
                        if key in metadata:
                            metadata[key] += ' ' + value
                        else:
                            metadata[key] = value
                        continue
                    
                    # Parse keyword = value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes and comments from value
                        if '/' in value:
                            value = value.split('/')[0].strip()
                        value = value.strip('\'"')
                        
                        # Convert numeric values
                        metadata[key] = self._convert_value(value)
                    else:
                        logger.warning(f"Malformed line {line_num} in {filepath}: {line}")
        
        except Exception as e:
            raise ParseError(f"Error parsing header file {filepath}: {str(e)}")
        
        return {
            'metadata': metadata,
            'filepath': filepath,
            'file_type': 'hdr'
        }
    
    def _convert_value(self, value: str) -> Union[str, int, float]:
        """Convert string value to appropriate type."""
        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate header data structure."""
        required_fields = ['metadata', 'filepath', 'file_type']
        return all(field in data for field in required_fields)


class DatFileParser(FileParser):
    """Parser for binary spectrum data files."""
    
    def parse(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parse .dat binary file into spectrum array."""
        if not Path(filepath).exists():
            raise ParseError(f"Data file not found: {filepath}")
        
        # Default metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Determine endianness from metadata (default to little endian)
        bocode = metadata.get('bocode', 0)
        endian = '<' if bocode == 0 else '>'
        
        # Get number of points from metadata
        npo = metadata.get('npo', None)
        
        try:
            # Read binary data as 4-byte floats
            dtype = f'{endian}f4'
            spectrum = np.fromfile(filepath, dtype=dtype)
            
            # Validate array size against metadata
            if npo is not None and len(spectrum) != npo:
                logger.warning(f"Spectrum length {len(spectrum)} doesn't match metadata npo={npo}")
            
            # Generate wavenumber array if parameters available
            wavenumbers = None
            if all(key in metadata for key in ['wstart', 'delw']):
                wstart = metadata['wstart']
                delw = metadata['delw']
                wavenumbers = np.arange(len(spectrum)) * delw + wstart
            
            return {
                'spectrum': spectrum,
                'wavenumbers': wavenumbers,
                'metadata': metadata,
                'filepath': filepath,
                'file_type': 'dat',
                'npoints': len(spectrum)
            }
        
        except Exception as e:
            raise ParseError(f"Error parsing binary file {filepath}: {str(e)}")
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate spectrum data structure."""
        required_fields = ['spectrum', 'filepath', 'file_type', 'npoints']
        if not all(field in data for field in required_fields):
            return False
        
        # Validate spectrum is numpy array
        if not isinstance(data['spectrum'], np.ndarray):
            return False
        
        # Validate wavenumbers if present
        if data.get('wavenumbers') is not None:
            if not isinstance(data['wavenumbers'], np.ndarray):
                return False
            if len(data['wavenumbers']) != len(data['spectrum']):
                return False
        
        return True


class CsvFileParser(FileParser):
    """Parser for CSV and text tabular data files."""
    
    def parse(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """Parse CSV/text file with first line as column metadata."""
        if not Path(filepath).exists():
            raise ParseError(f"CSV file not found: {filepath}")
        
        try:
            # Auto-detect delimiter
            delimiter = self._detect_delimiter(filepath)
            
            # Read CSV with pandas
            df = pd.read_csv(filepath, delimiter=delimiter, **kwargs)
            
            # Extract column metadata (first row contains column info)
            columns = list(df.columns)
            
            return {
                'data': df,
                'columns': columns,
                'nrows': len(df),
                'ncols': len(columns),
                'filepath': filepath,
                'file_type': 'csv',
                'delimiter': delimiter
            }
        
        except Exception as e:
            raise ParseError(f"Error parsing CSV file {filepath}: {str(e)}")
    
    def _detect_delimiter(self, filepath: str) -> str:
        """Auto-detect CSV delimiter."""
        with open(filepath, 'r') as f:
            first_line = f.readline()
        
        # Check common delimiters
        delimiters = [',', '\t', ';', '|']
        delimiter_counts = {d: first_line.count(d) for d in delimiters}
        
        # Return delimiter with highest count
        return max(delimiter_counts, key=delimiter_counts.get)
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate CSV data structure."""
        required_fields = ['data', 'columns', 'nrows', 'ncols', 'filepath', 'file_type']
        if not all(field in data for field in required_fields):
            return False
        
        # Validate DataFrame
        if not isinstance(data['data'], pd.DataFrame):
            return False
        
        # Validate consistency
        df = data['data']
        if len(df) != data['nrows'] or len(df.columns) != data['ncols']:
            return False
        
        return True


class LinelistFileParser(FileParser):
    """Parser for ASCII linelist files (.aln, .lst, .wavcorr) containing spectral line data."""
    
    def parse(self, filepath: str) -> Dict[str, Any]:
        """
        Parse ASCII linelist file with format:
        First 4 lines: metadata
        Remaining lines: number, wavenumber, peak, width, dmp, eq.Width, itn, H, tags
        """
        if not Path(filepath).exists():
            raise ParseError(f"Linelist file not found: {filepath}")
        
        lines_data = []
        metadata = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
                # Extract first 4 lines as metadata
                for i in range(min(4, len(all_lines))):
                    metadata.append(all_lines[i].strip())
                
                # Process remaining lines as data
                for line_num, line in enumerate(all_lines[4:], 5):  # Start from line 5
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#') or line.startswith('/'):
                        continue
                    
                    parts = line.split()
                    
                    # Ensure minimum required fields
                    if len(parts) < 9:
                        logger.warning(f"Line {line_num} has insufficient fields: {line}")
                        continue
                    
                    try:
                        # Parse according to specified format
                        line_data = {
                            'number': int(parts[0]),
                            'wavenumber': float(parts[1]),
                            'peak': float(parts[2]),
                            'width': float(parts[3]),
                            'dmp': float(parts[4]),
                            'eq_width': float(parts[5]),
                            'itn': int(parts[6]),
                            'H': int(parts[7]),
                            'tags': parts[8] if len(parts) > 8 else ""
                        }
                        
                        lines_data.append(line_data)
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse line {line_num}: {line} - {str(e)}")
                        continue
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(lines_data)
            
            return {
                'data': df,
                'lines': lines_data,
                'metadata': metadata,
                'nlines': len(lines_data),
                'columns': ['number', 'wavenumber', 'peak', 'width', 'dmp', 'eq_width', 'itn', 'H', 'tags'],
                'filepath': filepath,
                'file_type': 'linelist'
            }
        
        except Exception as e:
            raise ParseError(f"Error parsing linelist file {filepath}: {str(e)}")
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate linelist data structure."""
        required_fields = ['data', 'lines', 'metadata', 'nlines', 'columns', 'filepath', 'file_type']
        if not all(field in data for field in required_fields):
            return False
        
        # Validate DataFrame
        if not isinstance(data['data'], pd.DataFrame):
            return False
        
        # Validate metadata is a list
        if not isinstance(data['metadata'], list):
            return False
        
        # Validate required columns exist
        required_cols = ['number', 'wavenumber', 'peak', 'width', 'dmp', 'eq_width', 'itn', 'H', 'tags']
        if not all(col in data['data'].columns for col in required_cols):
            return False
        
        # Validate consistency
        if len(data['lines']) != data['nlines']:
            return False
        
        return True


class FileParserFactory:
    """Factory class to create appropriate parser based on file extension."""
    
    _parsers = {
        '.hdr': HdrFileParser,
        '.dat': DatFileParser,
        '.csv': CsvFileParser,
        '.txt': CsvFileParser,
        '.aln': LinelistFileParser,
        '.lst': LinelistFileParser,
        '.wavcorr': LinelistFileParser
    }
    
    @classmethod
    def create_parser(cls, filepath: str) -> FileParser:
        """Create parser based on file extension."""
        ext = Path(filepath).suffix.lower()
        
        if ext not in cls._parsers:
            raise ParseError(f"Unsupported file type: {ext}")
        
        return cls._parsers[ext]()
    
    @classmethod
    def parse_file(cls, filepath: str, **kwargs) -> Dict[str, Any]:
        """Convenience method to parse file with appropriate parser."""
        parser = cls.create_parser(filepath)
        
        # Handle DatFileParser which needs metadata parameter
        if isinstance(parser, DatFileParser) and 'metadata' in kwargs:
            return parser.parse(filepath, kwargs['metadata'])
        else:
            return parser.parse(filepath)