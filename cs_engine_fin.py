#!/usr/bin/env python3
"""
Maintenance Report NLP Analysis

This script implements an NLP pipeline for analyzing maintenance reports using BERT and NLTK.
"""

# Standard library imports
import os
import sys
import re
import time
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import timedelta

# Third-party imports
import torch
import pandas as pd
import nltk
import pyodbc
from tqdm import tqdm
from spellchecker import SpellChecker

# Deep learning imports
import torch
import transformers
from transformers import (
    BertTokenizer, 
    BertForTokenClassification,
    AutoTokenizer,
    AutoModelForTokenClassification, 
    logging as transformers_logging
)

# NLTK specific imports
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()

# Set up device (GPU, MPS, or CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("cpu")

# Constants
CSV_FILE_PATH = 'data/ice_makers.csv'
DB_CONFIG = {
    'driver': 'ODBC Driver 17 for SQL Server',
    'server': '35.184.99.218',
    'database': 'coolsys',
    'uid': 'sqlserver',
    'pwd': 'Ybz8Vq+|>\\H/<2py'
}

SQL_QUERY = """
SELECT
    w.wrkordr_wrk_rqstd,
    w.wrkordr_wrk_prfrmd,
    w2.wrkordreqpmnt_wrk_rqstd,
    w2.wrkordreqpmnt_wrk_prfrmd,
    w3.wrkordrinvntry_dscrptn
FROM
    coolsys.dbo.wrkordr w
    INNER JOIN coolsys.dbo.wrkordrinvntry w3 ON w.wrkordr_rn = w3.wrkordr_rn
    INNER JOIN coolsys.dbo.wrkordreqpmnt w2 ON w.wrkordr_rn = w2.wrkordr_rn
WHERE
    w.wrkordr_wrk_rqstd LIKE '%ICE MACHINE%'
    OR w.wrkordr_wrk_prfrmd LIKE '%ICE MACHINE%'
    OR w2.wrkordreqpmnt_wrk_rqstd LIKE '%ICE MACHINE%'
    OR w2.wrkordreqpmnt_wrk_prfrmd LIKE '%ICE MACHINE%'
    OR w3.wrkordrinvntry_dscrptn LIKE '%ICE MACHINE%';
"""

@contextmanager
def database_connection():
    """Context manager for database connections with proper error handling."""
    conn = None
    try:
        conn_str = (
            f'DRIVER={{{DB_CONFIG["driver"]}}};'
            f'SERVER={DB_CONFIG["server"]};'
            f'DATABASE={DB_CONFIG["database"]};'
            f'UID={DB_CONFIG["uid"]};'
            f'PWD={DB_CONFIG["pwd"]}'
        )
        conn = pyodbc.connect(conn_str, timeout=30)  # Add connection timeout
        logger.info("Database connection established")
        yield conn
    except pyodbc.Error as e:
        logger.error(f"Database connection error: {str(e)}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")

def fetch_data_from_db() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch data from database with optimized performance and error handling."""
    try:
        with database_connection() as conn:
            # Configure connection for better performance
            conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
            conn.setencoding(encoding='utf-8')
            
            # Use pandas read_sql for better performance
            start_time = time.time()
            df = pd.read_sql(SQL_QUERY, conn)
            logger.info(f"Query executed in {time.time() - start_time:.2f} seconds")
            
            return df, None
    except Exception as e:
        error_msg = f"Error fetching data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

def save_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV with error handling."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save with optimized settings
        df.to_csv(filepath, index=False, compression='infer')
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}", exc_info=True)
        raise

def load_data() -> pd.DataFrame:
    """Main function to load data with caching mechanism."""
    try:
        # Try loading from cache first
        if os.path.exists(CSV_FILE_PATH):
            logger.info(f"Loading data from cache: {CSV_FILE_PATH}")
            return pd.read_csv(CSV_FILE_PATH)
        
        # Fetch from database if cache doesn't exist
        logger.info("Cache not found, fetching from database")
        df, error = fetch_data_from_db()
        
        if error:
            raise RuntimeError(error)
        
        if df is not None and not df.empty:
            # Save to cache
            save_to_csv(df, CSV_FILE_PATH)
            return df
        else:
            raise ValueError("No data retrieved from database")
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise

def check_environment() -> None:
    """Check and log the execution environment."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"Using device: {device}")

def analyze_root_cause(verb, obj, text):
    """Analyze the root cause of an issue based on the maintenance action and context."""
    # Define detailed component categories and their related terms
    components = {
        'ice_making': {
            'ice maker', 'ice machine', 'ice production', 'ice cube',
            'evaporator', 'freeze plate', 'harvest', 'bin', 'ice thickness'
        },
        'refrigeration': {
            'refrigerant', 'compressor', 'condenser', 'evaporator',
            'expansion valve', 'cooling', 'freeze', 'temperature'
        },
        'water_system': {
            'water', 'pump', 'filter', 'flow', 'pressure', 'valve',
            'inlet', 'outlet', 'drain', 'reservoir', 'distribution'
        },
        'electrical': {
            'power', 'circuit', 'voltage', 'current', 'electrical',
            'switch', 'sensor', 'control board', 'pcb', 'thermostat'
        },
        'mechanical': {
            'motor', 'fan', 'belt', 'bearing', 'gear', 'pulley',
            'shaft', 'blade', 'auger', 'agitator'
        }
    }
    
    # Define problem categories and their indicators
    problems = {
        'failure': {
            'fail', 'broken', 'not working', 'malfunction', 'error',
            'fault', 'dead', 'stopped', 'inoperative'
        },
        'noise': {
            'noise', 'loud', 'vibration', 'rattle', 'buzz', 'hum',
            'squeal', 'grinding', 'knocking'
        },
        'leakage': {
            'leak', 'drip', 'overflow', 'spill', 'seep', 'discharge',
            'escape', 'loss'
        },
        'quality': {
            'dirty', 'contaminated', 'scale', 'buildup', 'quality',
            'taste', 'odor', 'color', 'appearance'
        }
    }
    
    # Define maintenance types
    maintenance_types = {
        'preventive': {
            'inspect', 'check', 'clean', 'adjust', 'calibrate',
            'maintain', 'service', 'test'
        },
        'corrective': {
            'repair', 'replace', 'fix', 'rebuild', 'overhaul',
            'restore', 'modify'
        },
        'emergency': {
            'emergency', 'urgent', 'immediate', 'critical', 'severe',
            'dangerous', 'safety'
        }
    }
    
    def find_matches(text, categories):
        """Find matching categories based on text content."""
        text = text.lower()
        matches = []
        for category, terms in categories.items():
            if any(term in text for term in terms):
                matches.append(category)
        return matches
    
    # Combine all text for analysis
    full_text = f"{verb} {obj} {text}".lower()
    
    # Find matches in each category
    component_matches = find_matches(full_text, components)
    problem_matches = find_matches(full_text, problems)
    maint_matches = find_matches(full_text, maintenance_types)
    
    # Return analysis results
    return {
        'components': component_matches,
        'problems': problem_matches,
        'maintenance_type': maint_matches[0] if maint_matches else 'other'
    }

def main():
    """Main execution function."""
    try:
        # Check environment
        check_environment()
        
        # Load data
        df = load_data()
        logger.info(f"DataFrame shape: {df.shape}")
        
        # Create output directory
        output_dir = 'data/maintenance_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting to process maintenance records...")
        start_time = time.time()
        
        # Initialize counters and storage
        component_stats = {
            'components': Counter(),
            'problems': Counter(),
            'maintenance_types': Counter(),
            'tasks': defaultdict(int)
        }
        
        # Process each maintenance record
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
            for field in df.columns:
                text = str(row[field])
                if pd.notna(text) and text.strip():
                    analysis = analyze_root_cause('', '', text)
                    
                    # Update statistics
                    component_stats['components'].update(analysis['components'])
                    component_stats['problems'].update(analysis['problems'])
                    component_stats['maintenance_types'].update([analysis['maintenance_type']])
                    
                    # Create task description
                    task = f"{text[:100]}..." if len(text) > 100 else text
                    component_stats['tasks'][task] += 1
        
        # Create summary DataFrames
        maintenance_df = pd.DataFrame({
            'Component': list(component_stats['components'].keys()),
            'Count': list(component_stats['components'].values())
        }).sort_values('Count', ascending=False)
        
        problems_df = pd.DataFrame({
            'Problem': list(component_stats['problems'].keys()),
            'Count': list(component_stats['problems'].values())
        }).sort_values('Count', ascending=False)
        
        maintenance_types_df = pd.DataFrame({
            'Type': list(component_stats['maintenance_types'].keys()),
            'Count': list(component_stats['maintenance_types'].values())
        }).sort_values('Count', ascending=False)
        
        # Save results
        maintenance_df.to_csv(f'{output_dir}/component_analysis.csv', index=False)
        problems_df.to_csv(f'{output_dir}/problem_analysis.csv', index=False)
        maintenance_types_df.to_csv(f'{output_dir}/maintenance_types.csv', index=False)
        
        # Print summary
        print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
        print("\nComponent Analysis:")
        print(maintenance_df)
        print("\nProblem Analysis:")
        print(problems_df)
        print("\nMaintenance Types:")
        print(maintenance_types_df)
        
        print(f"\nAnalysis files saved to: {output_dir}")
        
    except Exception as e:
        logger.error("Error in main execution", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
