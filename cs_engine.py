#%% Imports and Setup
##################
# Core Imports
##################

# Standard library imports
import os
import sys
import re
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import timedelta

# Data structures and utilities
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter, defaultdict
from contextlib import contextmanager

##################
# Data Science Stack
##################

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

##################
# ML and NLP
##################

# PyTorch
import torch

# Transformers
import transformers
from transformers import (
    BertTokenizer, 
    BertForTokenClassification,
    AutoTokenizer,
    AutoModelForTokenClassification, 
    logging as transformers_logging
)
# Configure transformers logging first
transformers_logging.set_verbosity_error()

# NLTK components
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

##################
# Infrastructure
##################

# Database and progress tracking
import pyodbc
from tqdm import tqdm
from spellchecker import SpellChecker
import sqlalchemy as sa
from sqlalchemy.engine import URL

##################
# Utility Functions and Decorators
##################

def validate_dataframe(func):
    """Decorator for DataFrame validation."""
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        if df is None or df.empty:
            raise ValueError("Invalid DataFrame returned")
        return df
    return wrapper

def setup_nltk(data_path: Optional[str] = None):
    """Centralized NLTK setup function."""
    resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'{resource}')
        except LookupError:
            nltk.download(resource, quiet=True, download_dir=data_path)

def create_distribution(data: List, category_name: str) -> pd.DataFrame:
    """Generic function for creating distribution analysis."""
    counts = Counter(data)
    df = pd.DataFrame(list(counts.items()), columns=[category_name, 'Count'])
    return add_percentage(df)

##################
# Configuration
##################

# Define global font sizes for consistency across all charts
LABEL_SIZE = 12       # Font size for axis labels and tick labels
TITLE_SIZE = 14       # Title size
PERCENT_SIZE = 10     # Font size for percentage labels
PIE_FONT_SIZE = 10    # Font size for pie chart legends and labels
HEATMAP_FONT_SIZE = 10  # Font size for heatmap labels and annotations

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Device setup - ensure compatibility with coolsys_env
device = torch.device("cpu")  # Use CPU as default
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")

# Ensure NLTK data is available
setup_nltk()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%% Constants and Configuration
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

# Constants for maintenance types and problems
MAINTENANCE_TYPES = {
    'PREVENTIVE': [
        'inspect', 'check', 'clean', 'test', 'maintain',
        'adjust', 'lubricate', 'tighten', 'calibrate',
        'measure', 'scheduled', 'routine', 'preventative',
        'monitor', 'service', 'tune', 'balance'
    ],
    'CORRECTIVE': [
        'repair', 'replace', 'fix', 'adjust',
        'rebuild', 'overhaul', 'restore', 'recondition',
        'patch', 'modify', 'correct', 'resolve',
        'troubleshoot', 'service'
    ],
    'EMERGENCY': [
        'emergency', 'urgent', 'immediate', 'breakdown',
        'failure', 'critical', 'malfunction', 'outage',
        'shutdown', 'stop', 'crash', 'break'
    ],
    'UPGRADE': [
        'upgrade', 'improve', 'enhance', 'modify',
        'update', 'modernize', 'retrofit', 'revamp',
        'optimize', 'augment', 'renovate'
    ],
    'DIAGNOSTIC': [
        'diagnose', 'analyze', 'investigate', 'assess',
        'evaluate', 'inspect', 'examine', 'check',
        'test', 'monitor', 'measure', 'observe'
    ]
}

PROBLEM_TYPES = {
    'MECHANICAL': [
        'belt', 'motor', 'bearing', 'gear', 'pump', 'shaft', 'compressor',
        'pulley', 'coupling', 'alignment', 'wear', 'tension', 'loose',
        'seized', 'broken', 'misaligned', 'overheating'
    ],
    'ELECTRICAL': [
        'power', 'voltage', 'circuit', 'wire', 'electrical', 'sensor',
        'overload', 'short', 'tripped', 'fuse', 'breaker', 'connection',
        'grounding', 'capacitor', 'contactor', 'relay', 'transformer',
        'control board'
    ],
    'COOLING': [
        'refrigerant', 'temperature', 'cooling', 'freeze', 'cold', 'frost',
        'superheat', 'subcooling', 'pressure', 'charge', 'leak', 'ice buildup',
        'defrost', 'heat transfer', 'condenser', 'evaporator', 'TXV',
        'head pressure', 'suction pressure', 'thermostat'
    ],
    'WATER_SYSTEM': [
        'water', 'leak', 'flow', 'drain', 'pipe', 'valve',
        'pressure', 'blockage', 'clog', 'scale', 'mineral', 'buildup',
        'overflow', 'backup', 'pump', 'filter', 'connection', 'fitting',
        'water level', 'water quality'
    ],
    'ICE_QUALITY': [
        'ice', 'cube', 'size', 'shape', 'quality', 'production',
        'cloudy', 'clear', 'hollow', 'incomplete', 'bridging',
        'harvest', 'cycle time', 'capacity', 'rate', 'consistency',
        'melting', 'sticking', 'clumping'
    ],
    'NOISE': [
        'noise', 'vibration', 'loud', 'sound', 'rattling',
        'squealing', 'grinding', 'humming', 'buzzing', 'knocking',
        'clicking', 'banging', 'scraping', 'whistling'
    ],
    'CONTROL': [
        'control', 'setting', 'program', 'display', 'interface',
        'parameter', 'calibration', 'setpoint', 'configuration',
        'programming', 'communication', 'error code', 'alarm',
        'safety switch', 'high limit', 'low limit','display'
    ]
}

parts_consumables = {
    'PARTS': ['motor', 'compressor', 'fan', 'pump', 'valve', 'filter', 'sensor', 'board', 'switch', 'thermostat', 'belt','scroll compressor', 'reciprocating compressor', 'compressor body', 'compressor motor', 'crankshaft', 'piston', 'valve plate','condenser coil', 'condenser fan', 'condenser motor', 'heat exchanger', 'fins', 'condenser tube', 'evaporator coil', 'evaporator fan', 'evaporator motor', 'defrost heater', 'distribution tubes','expansion valve', 'TXV', 'solenoid valve', 'check valve', 'service valve', 'hot gas valve','filter drier', 'suction filter', 'strainer', 'accumulator'],
    'CONSUMABLES': ['refrigerant', 'oil', 'cleaner', 'lubricant', 'chemical','sanitizer','degreaser']
}

#%% Database Functions
@contextmanager
def database_connection():
    """Context manager for database connections with proper error handling."""
    engine = None
    try:
        connection_url = URL.create(
            "mssql+pyodbc",
            username=DB_CONFIG["uid"],
            password=DB_CONFIG["pwd"],
            host=DB_CONFIG["server"],
            database=DB_CONFIG["database"],
            query={
                "driver": DB_CONFIG["driver"],
                "TrustServerCertificate": "yes",
            },
        )
        engine = sa.create_engine(connection_url)
        logger.info("Database connection established")
        yield engine
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}", exc_info=True)
        raise
    finally:
        if engine:
            engine.dispose()
            logger.info("Database connection closed")

def fetch_data_from_db() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch data from database with optimized performance and error handling."""
    try:
        with database_connection() as engine:
            start_time = time.time()
            df = pd.read_sql(SQL_QUERY, engine)
            execution_time = time.time() - start_time
            
            logger.info(f"Data fetched successfully in {execution_time:.2f} seconds")
            return df, None
            
    except Exception as e:
        error_msg = f"Error fetching data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

@validate_dataframe
def load_data(use_cache: bool = True) -> pd.DataFrame:
    """Unified function to load data with optional caching."""
    try:
        if use_cache and os.path.exists(CSV_FILE_PATH):
            logger.info(f"Loading data from cache: {CSV_FILE_PATH}")
            return pd.read_csv(CSV_FILE_PATH)
        
        logger.info("Fetching data from database")
        df, error = fetch_data_from_db()
        
        if error:
            raise RuntimeError(f"Database fetch error: {error}")
        
        if df is not None and not df.empty:
            if use_cache:
                save_to_csv(df, CSV_FILE_PATH)
                logger.info(f"Saved {len(df)} records to cache")
            return df
        
        raise ValueError("No data retrieved from database")
        
    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}", exc_info=True)
        raise

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

def check_environment() -> None:
    """Check and log the execution environment."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")


#%%
def analyze_root_cause(verb, obj, text):
    """Analyze the root cause of an issue based on the maintenance action and context."""
    # Initialize variables
    root_cause = None
    confidence = 0.0
    
    # Define root cause patterns
    root_cause_patterns = {
        'wear_and_tear': {
            'worn', 'damaged', 'deteriorated', 'corroded',
            'rusted', 'degraded', 'aged', 'fatigued'
        },
        'improper_operation': {
            'misuse', 'overload', 'incorrect', 'improper',
            'wrong', 'error', 'mistake', 'fault'
        },
        'environmental': {
            'temperature', 'humidity', 'dust', 'dirt',
            'debris', 'contamination', 'environment'
        },
        'quality_issues': {
            'defective', 'faulty', 'poor quality', 'bad',
            'substandard', 'inferior', 'flawed'
        }
    }
    
    # Process text to find root cause
    text_lower = text.lower()
    
    # Find maintenance type
    maintenance_type = None
    max_matches = 0
    
    for mtype, keywords in MAINTENANCE_TYPES.items():
        matches = sum(1 for word in keywords if word.lower() in text_lower)
        if matches > max_matches:
            max_matches = matches
            maintenance_type = mtype
    
    # Determine maintenance type
    maintenance_type = determine_maintenance_type(text)
    
    return {
        'components': [],
        'problems': [],
        'parts': [],
        'maintenance_type': maintenance_type
    }

def determine_maintenance_type(text):
    """Determine the maintenance type based on text analysis."""
    if not text:
        return 'UNSPECIFIED'
        
    text_lower = text.lower()
    
    # Check each maintenance type
    for mtype, keywords in MAINTENANCE_TYPES.items():
        if any(keyword in text_lower for keyword in keywords):
            return mtype
    
    return 'OTHER'

def determine_problem_type(text):
    """Determine the problem type based on text analysis."""
    if not text:
        return 'UNSPECIFIED'
        
    text_lower = text.lower()
    found_problems = []
    
    for problem_type, keywords in PROBLEM_TYPES.items():
        if any(keyword in text_lower for keyword in keywords):
            found_problems.append(problem_type)
    
    if found_problems:
        return found_problems[0]  # Return the first found problem type
    return 'OTHER'

def find_parts_consumables(text):
    """Find parts and consumables mentioned in the text."""
    text_lower = text.lower()
    found_parts = []
    
    for category, subcategories in parts_consumables.items():
        for subcategory, parts in subcategories.items():
            if any(part.lower() in text_lower for part in parts):
                found_parts.append(f"{category}_{subcategory}")
    
    return found_parts

def process_parts_data(df):
    """Process parts and consumables data from maintenance records."""
    parts_consumables = defaultdict(int)
    parts = defaultdict(int)
    consumables = defaultdict(int)
    
    # Print DataFrame columns for debugging
    print("Available columns:", df.columns.tolist())
    
    # Define parts keywords - physical components that need replacement
    parts_keywords = {
        # Filters
        'filter': {'filter', 'water filter', 'air filter', 'filtration'},
        
        # Motors and fans
        'motor': {'motor', 'drive motor', 'fan motor'},
        'fan': {'fan', 'fan blade'},
        
        # Pumps and valves
        'pump': {'pump', 'water pump'},
        'valve': {'valve', 'solenoid', 'water valve'},
        
        # Control and sensing
        'board': {'board', 'control board', 'circuit board', 'pcb'},
        'sensor': {'sensor', 'thermistor', 'probe'},
        'switch': {'switch', 'power switch'},
        'thermostat': {'thermostat', 'thermal'},
        
        # Compressor and related
        'compressor': {'compressor', 'compression'},
        
        # Structural and connectors
        'hose': {'hose', 'tube', 'pipe'},
        'fitting': {'fitting', 'connector'},
        'drain': {'drain', 'drainage'}
    }

    # Define consumables keywords - items that are used up during operation/maintenance
    consumables_keywords = {
        # Chemicals for cleaning and treatment
        'chemicals': {'cleaner', 'sanitizer', 'chemical', 'acid', 'descaler',
                   'detergent', 'degreaser', 'sanitizing solution', 'treatment'},
        
        # Refrigerant and coolants
        'refrigerant': {'coolant', 'freon', 'r404a', 'r134a'},
        
        # Other fluids
        'other': {'sealant', 'adhesive', 'tape', 'test strip'},
        
    }

    # Define oil-related terms with regex patterns
    oil_terms = [
        r'(?<!c)\boil\b(?!er)',     # matches "oil" but not in "coil" or "boiler"
        r'(?<!c)\boils\b',          # matches "oils" but not "coils"
        r'\blubrican[t]?\b',        # matches "lubricant" or "lubricate"
        r'\bgrease[d]?\b',          # matches "grease" or "greased"
        r'\bcompressor oil\b',      # specific to compressor oil
        r'\bmineral oil\b',         # specific to mineral oil
        r'\boil level\b',           # specific to oil level checks
        r'\boil change\b',          # specific to oil changes
        r'\boil leak\b'             # specific to oil leaks
    ]
    
    # Refrigerant types that should be counted under the "Refrigerant" category
    refrigerant_types = {'freon', 'r404a', 'r134a'}
    
    # Process text from all relevant columns
    text_columns = ['wrkordr_wrk_rqstd', 'wrkordr_wrk_prfrmd', 'wrkordreqpmnt_wrk_rqstd', 'wrkordreqpmnt_wrk_prfrmd']
    
    for column in text_columns:
        if column in df.columns:
            print(f"Processing column: {column}")
            for text in df[column].fillna('').astype(str):
                text = text.lower()
                
                # Check for parts by category
                for category, terms in parts_keywords.items():
                    if any(term in text for term in terms):
                        category_name = category.title()  # Convert to title case
                        parts_consumables[category_name] += 1
                        parts[category_name] += 1
                
                # Check for consumables by category
                for category, terms in consumables_keywords.items():
                    if any(term in text for term in terms):
                        category_name = category.title()  # Convert to title case
                        parts_consumables[category_name] += 1
                        consumables[category_name] += 1
                
                # Check for oil-related terms
                oil_pattern = '|'.join(oil_terms)
                if re.search(oil_pattern, text, re.IGNORECASE):
                    parts_consumables["Oil"] += 1
                    consumables["Oil"] += 1
                
                # Check for refrigerant types and consolidate them
                for ref_type in refrigerant_types:
                    if ref_type in text:
                        parts_consumables["Refrigerant"] += 1
                        consumables["Refrigerant"] += 1
    
    return parts_consumables, parts, consumables

def process_maintenance_types(df):
    """Process and categorize maintenance types from the data."""
    maintenance_types_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing maintenance types"):
        text = f"{row['wrkordr_wrk_rqstd']} {row['wrkordr_wrk_prfrmd']}"
        mtype = determine_maintenance_type(text)
        maintenance_types_list.append(mtype)
    return maintenance_types_list

def process_problem_types(df):
    """Process and identify problem types from the data."""
    problem_types_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing problem types"):
        text = f"{row['wrkordr_wrk_rqstd']} {row['wrkordr_wrk_prfrmd']}"
        ptype = determine_problem_type(text)
        problem_types_list.append(ptype)
    return problem_types_list

def process_parts_consumables(df):
    """Process and identify parts and consumables from the data."""
    parts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing parts and consumables"):
        text = f"{row['wrkordr_wrk_rqstd']} {row['wrkordr_wrk_prfrmd']}"
        found_parts = find_parts_consumables(text)
        parts.extend(found_parts)
    return parts

def create_maintenance_type_distribution(types: List[str]) -> pd.DataFrame:
    """Create distribution analysis of maintenance types."""
    return create_distribution(types, 'Maintenance Type')

def create_problem_type_distribution(types: List[str]) -> pd.DataFrame:
    """Create distribution analysis of problem types."""
    return create_distribution(types, 'Problem Type')

def create_parts_consumables_distribution(parts: List[str]) -> pd.DataFrame:
    """Create distribution analysis of parts and consumables."""
    return create_distribution(parts, 'Part/Consumable')

def add_percentage(df):
    """Add percentage column to distribution DataFrame."""
    total = df['Count'].sum()
    df['Percentage'] = df['Count'] / total * 100
    return df

def calculate_maintenance_correlation(df, type1, type2):
    """Calculate correlation between maintenance types."""
    try:
        # Create binary vectors for each maintenance type
        type1_present = []
        type2_present = []
        
        # Process each row to check for presence of maintenance types
        for _, row in df.iterrows():
            text = ' '.join([
                str(row.get('wrkordr_wrk_rqstd', '')),
                str(row.get('wrkordr_wrk_prfrmd', '')),
                str(row.get('wrkordreqpmnt_wrk_rqstd', '')),
                str(row.get('wrkordreqpmnt_wrk_prfrmd', ''))
            ]).lower()
            
            # Check if maintenance types are present in the text
            type1_keywords = set(word.lower() for word in MAINTENANCE_TYPES.get(type1, []))
            type2_keywords = set(word.lower() for word in MAINTENANCE_TYPES.get(type2, []))
            
            type1_present.append(1 if any(keyword in text for keyword in type1_keywords) else 0)
            type2_present.append(1 if any(keyword in text for keyword in type2_keywords) else 0)
        
        # Convert to numpy arrays for calculations
        type1_array = np.array(type1_present, dtype=np.float64)
        type2_array = np.array(type2_present, dtype=np.float64)
        
        # Check if there's enough variance to calculate correlation
        if (type1_array == type1_array[0]).all() or (type2_array == type2_array[0]).all():
            return 0.0  # No correlation if one or both types have no variance
            
        # Calculate frequencies
        n = len(type1_array)
        if n == 0:
            return 0.0
            
        # Calculate proportions instead of raw counts to avoid overflow
        p11 = np.mean((type1_array == 1) & (type2_array == 1))
        p00 = np.mean((type1_array == 0) & (type2_array == 0))
        p10 = np.mean((type1_array == 1) & (type2_array == 0))
        p01 = np.mean((type1_array == 0) & (type2_array == 1))
        
        # Calculate marginal proportions
        p1_ = p11 + p10  # proportion of type1 = 1
        p0_ = p01 + p00  # proportion of type1 = 0
        p_1 = p11 + p01  # proportion of type2 = 1
        p_0 = p10 + p00  # proportion of type2 = 0
        
        # Check for zero marginal proportions
        if min(p1_, p0_, p_1, p_0) <= 0:
            return 0.0
            
        # Calculate phi coefficient using proportions
        try:
            correlation = (p11 * p00 - p10 * p01) / np.sqrt(p1_ * p0_ * p_1 * p_0)
            return correlation if not np.isnan(correlation) else 0.0
        except (RuntimeWarning, FloatingPointError):
            return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating correlation between {type1} and {type2}: {str(e)}")
        return 0.0

def calculate_correlations(maintenance_types_list, df):
    """Calculate correlations between maintenance types."""
    maintenance_correlation_data = []
    unique_types = list(set(maintenance_types_list))
    total_pairs = sum(1 for i in range(len(unique_types)) for j in range(i+1, len(unique_types)))
    
    with tqdm(total=total_pairs, desc="Calculating correlations") as pbar:
        for i, type1 in enumerate(unique_types):
            for type2 in unique_types[i:]:  # Include diagonal and upper triangle
                if type1 == type2:
                    correlation = 1.0
                else:
                    correlation = calculate_maintenance_correlation(df, type1, type2)
                    
                maintenance_correlation_data.append({
                    'Type1': type1,
                    'Type2': type2,
                    'Correlation': correlation
                })
                # Add symmetric pair for non-diagonal elements
                if type1 != type2:
                    maintenance_correlation_data.append({
                        'Type1': type2,
                        'Type2': type1,
                        'Correlation': correlation
                    })
                pbar.update(1)
    
    correlation_df = pd.DataFrame(maintenance_correlation_data)
    correlation_df = correlation_df.sort_values('Correlation', ascending=False)
    return correlation_df

def save_results(maintenance_type_df, problem_type_df, parts_consumables_df, parts_df, consumables_df, correlation_df):
    """Save analysis results to CSV files."""
    maintenance_type_df.to_csv('maintenance_types_analysis.csv', index=False)
    problem_type_df.to_csv('problem_types_analysis.csv', index=False)
    parts_consumables_df.to_csv('parts_consumables_analysis.csv', index=False)
    parts_df.to_csv('parts_analysis.csv', index=False)
    consumables_df.to_csv('consumables_analysis.csv', index=False)
    correlation_df.to_csv('maintenance_correlations.csv', index=False)

def create_visualizations(maintenance_type_df, problem_type_df, parts_consumables_df, parts_df, consumables_df, correlation_df):
    """Create and save all visualizations."""
    logger.info("Creating visualizations...")
    
    # Set the default figure size and DPI
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    
    print("\nDataFrames at visualization time:")
    print("\nMaintenance Types:")
    print(maintenance_type_df)
    print("\nProblem Types:")
    print(problem_type_df)
    
    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    correlation_df_filtered = correlation_df[
        ~correlation_df['Type1'].isin(['OTHER']) & 
        ~correlation_df['Type2'].isin(['OTHER'])
    ]
    pivot_corr = correlation_df_filtered.pivot(index='Type1', columns='Type2', values='Correlation')
    sns.heatmap(pivot_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
               annot_kws={'size': HEATMAP_FONT_SIZE})
    plt.title('Maintenance Type Correlations', fontsize=HEATMAP_FONT_SIZE + 4, fontweight='bold', pad=20)
    plt.xlabel('')  # Remove x-axis label
    plt.ylabel('')  # Remove y-axis label
    plt.xticks(rotation=45, ha='right', fontsize=HEATMAP_FONT_SIZE)
    plt.yticks(rotation=0, fontsize=HEATMAP_FONT_SIZE)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Maintenance Types Distribution (Pie Chart)
    plt.figure(figsize=(8, 6))
    total = maintenance_type_df['Count'].sum()
    sizes = maintenance_type_df['Count'].values
    
    # Convert type numbers to words
    type_mapping = {
        'PREVENTIVE': 'Preventive',
        'CORRECTIVE': 'Corrective',
        'EMERGENCY': 'Emergency',
        'UPGRADE': 'Upgrade',
        'DIAGNOSTIC': 'Diagnostic',
        'OTHER': 'Other',
        'UNSPECIFIED': 'Unspecified'
    }
    
    labels = [type_mapping.get(cat, cat) for cat in maintenance_type_df['Maintenance Type']]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Maintenance Types Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Problem Types Distribution (Pie Chart)
    plt.figure(figsize=(8, 6))
    total = problem_type_df['Count'].sum()
    sizes = problem_type_df['Count'].values
    
    problem_type_mapping = {
        'MECHANICAL': 'Mechanical',
        'ELECTRICAL': 'Electrical',
        'COOLING': 'Cooling',
        'WATER_SYSTEM': 'Water System',
        'ICE_QUALITY': 'Ice Quality',
        'NOISE': 'Noise',
        'CONTROL': 'Control',
        'OTHER': 'Other',
        'UNSPECIFIED': 'Unspecified'
    }
    
    labels = [problem_type_mapping.get(cat, cat) for cat in problem_type_df['Problem Type']]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Problem Types Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Parts/Consumables Distribution (Pie Chart)
    plt.figure(figsize=(8, 6))
    total = parts_consumables_df['Count'].sum()
    sizes = parts_consumables_df['Count'].values
    labels = [cat for cat in parts_consumables_df['Part/Consumable']]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Parts/Consumables Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Plot 5: Top Parts Distribution (Bar Chart)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=parts_df.head(10), x='Count', y='Part')
    plt.title('Top 10 Parts Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
    plt.ylabel('Part', fontsize=LABEL_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Plot 6: Top Consumables Distribution (Bar Chart)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=consumables_df.head(10), x='Count', y='Consumable')
    plt.title('Top 10 Consumables Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
    plt.ylabel('Consumable', fontsize=LABEL_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    logger.info("Visualizations completed successfully")

def display_visualizations(maintenance_type_df, problem_type_df, parts_consumables_df, parts_df, consumables_df, correlation_df):
    """Display all visualizations."""
    logger.info("Displaying visualizations...")
    
    try:
        # Display Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_df_filtered = correlation_df[
            ~correlation_df['Type1'].isin(['OTHER']) & 
            ~correlation_df['Type2'].isin(['OTHER'])
        ]
        pivot_corr = correlation_df_filtered.pivot(index='Type1', columns='Type2', values='Correlation')
        sns.heatmap(pivot_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title('Maintenance Type Correlations', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        plt.xlabel('')  # Remove x-axis label
        plt.ylabel('')  # Remove y-axis label
        plt.xticks(rotation=45, ha='right', fontsize=LABEL_SIZE)
        plt.yticks(rotation=0, fontsize=LABEL_SIZE)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Maintenance Types Distribution
        plt.figure(figsize=(8, 6))
        total = maintenance_type_df['Count'].sum()
        sizes = maintenance_type_df['Count'].values
        
        # Convert type numbers to words
        type_mapping = {
            'PREVENTIVE': 'Preventive',
            'CORRECTIVE': 'Corrective',
            'EMERGENCY': 'Emergency',
            'UPGRADE': 'Upgrade',
            'DIAGNOSTIC': 'Diagnostic',
            'OTHER': 'Other',
            'UNSPECIFIED': 'Unspecified'
        }
        
        # Use Maintenance Type column for labels and apply mapping
        labels = [type_mapping.get(cat, cat) for cat in maintenance_type_df['Maintenance Type']]
        
        wedges, texts, autotexts = plt.pie(sizes, 
                                         labels=None,
                                         autopct='%1.1f%%',
                                         pctdistance=0.85,
                                         textprops={'fontsize': PIE_FONT_SIZE})
        plt.title('Maintenance Types Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        
        # Add count to legend labels
        legend_labels = [f'{label} ({count:,})' for label, count in zip(labels, sizes)]
        plt.legend(wedges, legend_labels,
                  title="Types",
                  title_fontsize=PIE_FONT_SIZE,
                  fontsize=PIE_FONT_SIZE,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Problem Types Distribution
        plt.figure(figsize=(8, 6))
        total = problem_type_df['Count'].sum()
        sizes = problem_type_df['Count'].values
        
        # Convert problem type numbers to words
        problem_type_mapping = {
            'MECHANICAL': 'Mechanical',
            'ELECTRICAL': 'Electrical',
            'COOLING': 'Cooling',
            'WATER_SYSTEM': 'Water System',
            'ICE_QUALITY': 'Ice Quality',
            'NOISE': 'Noise',
            'CONTROL': 'Control',
            'OTHER': 'Other',
            'UNSPECIFIED': 'Unspecified'
        }
        
        # Use Problem Type column for labels and apply mapping
        labels = [problem_type_mapping.get(cat, cat) for cat in problem_type_df['Problem Type']]
        
        wedges, texts, autotexts = plt.pie(sizes, 
                                         labels=None,
                                         autopct='%1.1f%%',
                                         pctdistance=0.85,
                                         textprops={'fontsize': PIE_FONT_SIZE})
        plt.title('Problem Types Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        
        # Add count to legend labels
        legend_labels = [f'{label} ({count:,})' for label, count in zip(labels, sizes)]
        plt.legend(wedges, legend_labels,
                  title="Types",
                  title_fontsize=PIE_FONT_SIZE,
                  fontsize=PIE_FONT_SIZE,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Parts and Consumables Distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(data=parts_consumables_df.head(20), x='Count', y='Part/Consumable')
        plt.title('Top 20 Most Common Parts/Consumables', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
        plt.ylabel('Part/Consumable', fontsize=LABEL_SIZE, fontweight='bold')
        plt.xticks(fontsize=LABEL_SIZE)
        plt.yticks(fontsize=LABEL_SIZE)
        total = parts_consumables_df['Count'].sum()
        for i, v in enumerate(parts_consumables_df.head(20)['Count']):
            percentage = (v / total) * 100
            plt.text(v, i, f' {percentage:.1f}%', va='center', fontsize=PERCENT_SIZE)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Parts Distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(data=parts_df.head(20), x='Count', y='Part')
        plt.title('Root Causes of Failure - Parts', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
        plt.ylabel('Part', fontsize=LABEL_SIZE, fontweight='bold')
        plt.xticks(fontsize=LABEL_SIZE)
        plt.yticks(fontsize=LABEL_SIZE)
        total = parts_df['Count'].sum()
        for i, v in enumerate(parts_df.head(20)['Count']):
            percentage = (v / total) * 100
            plt.text(v, i, f' {percentage:.1f}%', va='center', fontsize=PERCENT_SIZE)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Consumables Distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(data=consumables_df.head(20), x='Count', y='Consumable')
        plt.title('Root Causes of Failure - Consumables', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
        plt.ylabel('Consumable', fontsize=LABEL_SIZE, fontweight='bold')
        plt.xticks(fontsize=LABEL_SIZE)
        plt.yticks(fontsize=LABEL_SIZE)
        total = consumables_df['Count'].sum()
        for i, v in enumerate(consumables_df.head(20)['Count']):
            percentage = (v / total) * 100
            plt.text(v, i, f' {percentage:.1f}%', va='center', fontsize=PERCENT_SIZE)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        logger.info("Visualizations displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying visualizations: {str(e)}")
        plt.close('all')
        raise

#%% Main Function and Steps
def check_and_log_environment():
    """Check and log the environment setup."""
    logger.info(f"Running on device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")

def fetch_and_validate_data():
    """Fetch data from CSV cache or database and validate it."""
    try:
        # Try loading from cache first
        if os.path.exists(CSV_FILE_PATH):
            logger.info(f"Loading data from cache: {CSV_FILE_PATH}")
            df = pd.read_csv(CSV_FILE_PATH)
            if not df.empty:
                logger.info(f"Successfully loaded {len(df)} records from cache")
                return df
            logger.warning("Cache file exists but is empty")
            
        # Fetch from database if cache doesn't exist or is empty
        logger.info("Fetching data from database")
        df, error = fetch_data_from_db()
        
        if error:
            logger.error(f"Database fetch error: {error}")
            return None
            
        if df is not None and not df.empty:
            # Save to cache
            save_to_csv(df, CSV_FILE_PATH)
            logger.info(f"Saved {len(df)} records to cache")
            return df
        else:
            logger.error("No data retrieved from database")
            return None
            
    except Exception as e:
        logger.error(f"Error in fetch_and_validate_data: {str(e)}", exc_info=True)
        return None

def process_maintenance_data(df):
    """Process maintenance types and parts data."""
    maintenance_types_list = process_maintenance_types(df)
    problem_types_list = process_problem_types(df)
    parts_data = process_parts_data(df)
    
    return maintenance_types_list, problem_types_list, parts_data

def create_analysis_dataframes(maintenance_types_list, problem_types_list, parts_data):
    """Create analysis DataFrames for maintenance types and parts."""
    # Create distribution DataFrames with consistent column names
    maintenance_type_df = create_maintenance_type_distribution(maintenance_types_list)
    problem_type_df = create_problem_type_distribution(problem_types_list)
    
    # Unpack parts data
    parts_consumables, parts, consumables = parts_data
    
    # Create DataFrames with consistent column names
    parts_consumables_df = create_parts_consumables_distribution(list(parts_consumables.keys()))
    parts_consumables_df['Count'] = list(parts_consumables.values())
    
    parts_df = pd.DataFrame({
        'Part': list(parts.keys()),
        'Count': list(parts.values())
    })
    
    consumables_df = pd.DataFrame({
        'Consumable': list(consumables.keys()),
        'Count': list(consumables.values())
    })
    
    # Add percentage to all DataFrames
    maintenance_type_df = add_percentage(maintenance_type_df)
    problem_type_df = add_percentage(problem_type_df)
    parts_consumables_df = add_percentage(parts_consumables_df)
    parts_df = add_percentage(parts_df)
    consumables_df = add_percentage(consumables_df)
    
    return maintenance_type_df, problem_type_df, parts_consumables_df, parts_df, consumables_df

def save_results(maintenance_type_df, problem_type_df, parts_consumables_df, parts_df, consumables_df, correlation_df):
    """Save analysis results to CSV files."""
    maintenance_type_df.to_csv('maintenance_types_analysis.csv', index=False)
    problem_type_df.to_csv('problem_types_analysis.csv', index=False)
    parts_consumables_df.to_csv('parts_consumables_analysis.csv', index=False)
    parts_df.to_csv('parts_analysis.csv', index=False)
    consumables_df.to_csv('consumables_analysis.csv', index=False)
    correlation_df.to_csv('maintenance_correlations.csv', index=False)

def create_visualizations(maintenance_type_df, problem_type_df, parts_consumables_df, parts_df, consumables_df, correlation_df):
    """Create and save all visualizations."""
    logger.info("Creating visualizations...")
    
    # Set the default figure size and DPI
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    
    print("\nDataFrames at visualization time:")
    print("\nMaintenance Types:")
    print(maintenance_type_df)
    print("\nProblem Types:")
    print(problem_type_df)
    
    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    correlation_df_filtered = correlation_df[
        ~correlation_df['Type1'].isin(['OTHER']) & 
        ~correlation_df['Type2'].isin(['OTHER'])
    ]
    pivot_corr = correlation_df_filtered.pivot(index='Type1', columns='Type2', values='Correlation')
    sns.heatmap(pivot_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
               annot_kws={'size': HEATMAP_FONT_SIZE})
    plt.title('Maintenance Type Correlations', fontsize=HEATMAP_FONT_SIZE + 4, fontweight='bold', pad=20)
    plt.xlabel('')  # Remove x-axis label
    plt.ylabel('')  # Remove y-axis label
    plt.xticks(rotation=45, ha='right', fontsize=HEATMAP_FONT_SIZE)
    plt.yticks(rotation=0, fontsize=HEATMAP_FONT_SIZE)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Maintenance Types Distribution (Pie Chart)
    plt.figure(figsize=(8, 6))
    total = maintenance_type_df['Count'].sum()
    sizes = maintenance_type_df['Count'].values
    
    # Convert type numbers to words
    type_mapping = {
        'PREVENTIVE': 'Preventive',
        'CORRECTIVE': 'Corrective',
        'EMERGENCY': 'Emergency',
        'UPGRADE': 'Upgrade',
        'DIAGNOSTIC': 'Diagnostic',
        'OTHER': 'Other',
        'UNSPECIFIED': 'Unspecified'
    }
    
    labels = [type_mapping.get(cat, cat) for cat in maintenance_type_df['Maintenance Type']]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Maintenance Types Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Problem Types Distribution (Pie Chart)
    plt.figure(figsize=(8, 6))
    total = problem_type_df['Count'].sum()
    sizes = problem_type_df['Count'].values
    
    problem_type_mapping = {
        'MECHANICAL': 'Mechanical',
        'ELECTRICAL': 'Electrical',
        'COOLING': 'Cooling',
        'WATER_SYSTEM': 'Water System',
        'ICE_QUALITY': 'Ice Quality',
        'NOISE': 'Noise',
        'CONTROL': 'Control',
        'OTHER': 'Other',
        'UNSPECIFIED': 'Unspecified'
    }
    
    labels = [problem_type_mapping.get(cat, cat) for cat in problem_type_df['Problem Type']]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Problem Types Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Parts/Consumables Distribution (Pie Chart)
    plt.figure(figsize=(8, 6))
    total = parts_consumables_df['Count'].sum()
    sizes = parts_consumables_df['Count'].values
    labels = [cat for cat in parts_consumables_df['Part/Consumable']]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Parts/Consumables Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Plot 5: Top Parts Distribution (Bar Chart)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=parts_df.head(10), x='Count', y='Part')
    plt.title('Top 10 Parts Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
    plt.ylabel('Part', fontsize=LABEL_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Plot 6: Top Consumables Distribution (Bar Chart)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=consumables_df.head(10), x='Count', y='Consumable')
    plt.title('Top 10 Consumables Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
    plt.ylabel('Consumable', fontsize=LABEL_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    logger.info("Visualizations completed successfully")

def display_visualizations(maintenance_type_df, problem_type_df, parts_consumables_df, parts_df, consumables_df, correlation_df):
    """Display all visualizations."""
    logger.info("Displaying visualizations...")
    
    try:
        # Display Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_df_filtered = correlation_df[
            ~correlation_df['Type1'].isin(['OTHER']) & 
            ~correlation_df['Type2'].isin(['OTHER'])
        ]
        pivot_corr = correlation_df_filtered.pivot(index='Type1', columns='Type2', values='Correlation')
        sns.heatmap(pivot_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title('Maintenance Type Correlations', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        plt.xlabel('')  # Remove x-axis label
        plt.ylabel('')  # Remove y-axis label
        plt.xticks(rotation=45, ha='right', fontsize=LABEL_SIZE)
        plt.yticks(rotation=0, fontsize=LABEL_SIZE)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Maintenance Types Distribution
        plt.figure(figsize=(8, 6))
        total = maintenance_type_df['Count'].sum()
        sizes = maintenance_type_df['Count'].values
        
        # Convert type numbers to words
        type_mapping = {
            'PREVENTIVE': 'Preventive',
            'CORRECTIVE': 'Corrective',
            'EMERGENCY': 'Emergency',
            'UPGRADE': 'Upgrade',
            'DIAGNOSTIC': 'Diagnostic',
            'OTHER': 'Other',
            'UNSPECIFIED': 'Unspecified'
        }
        
        # Use Maintenance Type column for labels and apply mapping
        labels = [type_mapping.get(cat, cat) for cat in maintenance_type_df['Maintenance Type']]
        
        wedges, texts, autotexts = plt.pie(sizes, 
                                         labels=None,
                                         autopct='%1.1f%%',
                                         pctdistance=0.85,
                                         textprops={'fontsize': PIE_FONT_SIZE})
        plt.title('Maintenance Types Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        
        # Add count to legend labels
        legend_labels = [f'{label} ({count:,})' for label, count in zip(labels, sizes)]
        plt.legend(wedges, legend_labels,
                  title="Types",
                  title_fontsize=PIE_FONT_SIZE,
                  fontsize=PIE_FONT_SIZE,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Problem Types Distribution
        plt.figure(figsize=(8, 6))
        total = problem_type_df['Count'].sum()
        sizes = problem_type_df['Count'].values
        
        # Convert problem type numbers to words
        problem_type_mapping = {
            'MECHANICAL': 'Mechanical',
            'ELECTRICAL': 'Electrical',
            'COOLING': 'Cooling',
            'WATER_SYSTEM': 'Water System',
            'ICE_QUALITY': 'Ice Quality',
            'NOISE': 'Noise',
            'CONTROL': 'Control',
            'OTHER': 'Other',
            'UNSPECIFIED': 'Unspecified'
        }
        
        # Use Problem Type column for labels and apply mapping
        labels = [problem_type_mapping.get(cat, cat) for cat in problem_type_df['Problem Type']]
        
        wedges, texts, autotexts = plt.pie(sizes, 
                                         labels=None,
                                         autopct='%1.1f%%',
                                         pctdistance=0.85,
                                         textprops={'fontsize': PIE_FONT_SIZE})
        plt.title('Problem Types Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        
        # Add count to legend labels
        legend_labels = [f'{label} ({count:,})' for label, count in zip(labels, sizes)]
        plt.legend(wedges, legend_labels,
                  title="Types",
                  title_fontsize=PIE_FONT_SIZE,
                  fontsize=PIE_FONT_SIZE,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Parts and Consumables Distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(data=parts_consumables_df.head(20), x='Count', y='Part/Consumable')
        plt.title('Top 20 Most Common Parts/Consumables', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
        plt.ylabel('Part/Consumable', fontsize=LABEL_SIZE, fontweight='bold')
        plt.xticks(fontsize=LABEL_SIZE)
        plt.yticks(fontsize=LABEL_SIZE)
        total = parts_consumables_df['Count'].sum()
        for i, v in enumerate(parts_consumables_df.head(20)['Count']):
            percentage = (v / total) * 100
            plt.text(v, i, f' {percentage:.1f}%', va='center', fontsize=PERCENT_SIZE)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Parts Distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(data=parts_df.head(20), x='Count', y='Part')
        plt.title('Root Causes of Failure - Parts', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
        plt.ylabel('Part', fontsize=LABEL_SIZE, fontweight='bold')
        plt.xticks(fontsize=LABEL_SIZE)
        plt.yticks(fontsize=LABEL_SIZE)
        total = parts_df['Count'].sum()
        for i, v in enumerate(parts_df.head(20)['Count']):
            percentage = (v / total) * 100
            plt.text(v, i, f' {percentage:.1f}%', va='center', fontsize=PERCENT_SIZE)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Display Consumables Distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(data=consumables_df.head(20), x='Count', y='Consumable')
        plt.title('Root Causes of Failure - Consumables', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        plt.xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
        plt.ylabel('Consumable', fontsize=LABEL_SIZE, fontweight='bold')
        plt.xticks(fontsize=LABEL_SIZE)
        plt.yticks(fontsize=LABEL_SIZE)
        total = consumables_df['Count'].sum()
        for i, v in enumerate(consumables_df.head(20)['Count']):
            percentage = (v / total) * 100
            plt.text(v, i, f' {percentage:.1f}%', va='center', fontsize=PERCENT_SIZE)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        logger.info("Visualizations displayed successfully")
        
    except Exception as e:
        logger.error(f"Error displaying visualizations: {str(e)}")
        plt.close('all')
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Check and log environment
        check_and_log_environment()
        
        # Step 2: Load and validate data
        df = fetch_and_validate_data()
        
        # Step 3: Process maintenance data
        maintenance_types_list, problem_types_list, parts_data = process_maintenance_data(df)
        
        # Step 4: Create analysis DataFrames
        maintenance_type_df, problem_type_df, parts_consumables_df, parts_df, consumables_df = create_analysis_dataframes(
            maintenance_types_list, problem_types_list, parts_data
        )
        
        # Step 5: Calculate correlations
        correlation_df = calculate_correlations(maintenance_types_list, df)
        
        # Step 6: Save results to CSV files
        save_results(
            maintenance_type_df, problem_type_df, parts_consumables_df, 
            parts_df, consumables_df, correlation_df
        )
        
        # Step 7: Create and save visualizations
        create_visualizations(
            maintenance_type_df, problem_type_df, parts_consumables_df,
            parts_df, consumables_df, correlation_df
        )
        
        # Step 8: Display visualizations
        display_visualizations(
            maintenance_type_df, problem_type_df, parts_consumables_df,
            parts_df, consumables_df, correlation_df
        )
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise