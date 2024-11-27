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
        # Ice Making System
        'freeze_plate_system': {
            'evaporator plate', 'freeze plate', 'cooling surface', 'grid plate', 
            'ice mold', 'cube form', 'ice formation', 'freezing surface',
            'water distribution plate', 'cell size', 'bridge thickness'
        },
        'ice_formation_control': {
            'thickness sensor', 'ice thickness', 'bridge control',
            'water level sensor', 'formation time', 'freeze cycle',
            'water curtain', 'spray time', 'freeze timer'
        },
        'harvest_system': {
            'harvest valve', 'hot gas valve', 'defrost valve', 
            'harvest timer', 'harvest assist', 'release mechanism',
            'hot gas bypass', 'harvest pressure', 'harvest solenoid',
            'harvest check valve', 'harvest complete switch'
        },
        'ice_handling': {
            'ice bin', 'storage bin', 'bin thermostat', 'bin control',
            'ice chute', 'dispenser', 'agitator', 'auger motor',
            'ice gate', 'bin door', 'bin level sensor', 'ice shield'
        },
        'water_distribution_ice': {
            'water pump', 'spray bar', 'spray nozzles', 'water curtain',
            'water trough', 'water sump', 'distribution tube',
            'water level', 'float valve', 'reservoir'
        },
        
        # Refrigeration System
        'compressor_system': {
            'compressor', 'piston', 'scroll', 'reed valve', 'valve plate',
            'discharge line', 'suction line', 'oil level', 'crankcase',
            'compressor motor', 'windings', 'terminal', 'overload protector'
        },
        'condenser_system': {
            'condenser coil', 'fan motor', 'fan blade', 'air flow',
            'head pressure', 'subcooling', 'fins', 'coil surface',
            'condenser pressure', 'air cooled', 'water cooled'
        },
        'refrigerant_circuit': {
            'refrigerant', 'freon', 'charge', 'leak', 'pressure',
            'filter drier', 'sight glass', 'accumulator', 'receiver',
            'liquid line', 'suction line', 'discharge line'
        },
        'expansion_system': {
            'TXV', 'expansion valve', 'capillary tube', 'orifice',
            'superheat', 'bulb', 'equalizer', 'distributor',
            'metering device', 'restrictor'
        },
        
        # Water System
        'water_supply': {
            'water line', 'inlet valve', 'water pressure', 'filter',
            'strainer', 'softener', 'water quality', 'supply pipe',
            'shut off valve', 'pressure regulator'
        },
        'water_pump_system': {
            'circulation pump', 'impeller', 'seal', 'pump motor',
            'pump capacity', 'pump housing', 'bearing', 'shaft',
            'pump pressure', 'pump strainer'
        },
        'water_distribution': {
            'spray nozzles', 'water curtain', 'distributor', 'tube',
            'spray pattern', 'distribution manifold', 'spray bar',
            'water flow', 'distribution uniformity'
        },
        'drain_system': {
            'drain line', 'condensate', 'drain pan', 'float switch',
            'pump out', 'drain valve', 'trap', 'vent', 'slope',
            'drain heater', 'overflow protection'
        },
        
        # Electrical System
        'control_board': {
            'PCB', 'control board', 'controller', 'motherboard',
            'processor', 'relay board', 'display board', 'interface',
            'memory', 'firmware', 'programming'
        },
        'sensors_system': {
            'thermistor', 'probe', 'sensor', 'thermostat', 'float switch',
            'bin level sensor', 'pressure sensor', 'temperature sensor',
            'water level sensor', 'ice thickness sensor'
        },
        'electrical_components': {
            'contactor', 'relay', 'capacitor', 'transformer', 'fuse',
            'overload', 'terminal block', 'wire connector', 'breaker',
            'power supply', 'voltage regulator'
        },
        'wiring_system': {
            'wire harness', 'connection', 'terminal', 'plug', 'socket',
            'ground wire', 'power cable', 'communication wire',
            'insulation', 'conduit', 'junction box'
        }
    }
    
    # Define parts and consumables categories
    parts_consumables = {
        'refrigeration_parts': {
            'compressor': {'scroll compressor', 'reciprocating compressor', 'compressor body', 'compressor motor', 'crankshaft', 'piston', 'valve plate'},
            'condenser': {'condenser coil', 'condenser fan', 'condenser motor', 'heat exchanger', 'fins', 'condenser tube'},
            'evaporator': {'evaporator coil', 'evaporator fan', 'evaporator motor', 'defrost heater', 'distribution tubes'},
            'valves': {'expansion valve', 'TXV', 'solenoid valve', 'check valve', 'service valve', 'hot gas valve'},
            'filters': {'filter drier', 'suction filter', 'strainer', 'accumulator'}
        },
        'ice_making_parts': {
            'freeze_plate': {'evaporator plate', 'grid plate', 'water distribution plate', 'spray bar', 'spray nozzles'},
            'harvesting': {'hot gas valve', 'harvest assist', 'deflector', 'ice sweep', 'harvest probe'},
            'storage': {'bin liner', 'bin door', 'gasket', 'ice gate', 'agitator', 'auger'}
        },
        'water_system_parts': {
            'pumps': {'circulation pump', 'drain pump', 'water pump', 'pump motor', 'impeller', 'shaft seal'},
            'filtration': {'water filter', 'sediment filter', 'carbon filter', 'scale filter', 'filter housing'},
            'plumbing': {'water inlet valve', 'float valve', 'drain valve', 'water line', 'fitting', 'o-ring'}
        },
        'electrical_parts': {
            'controls': {'control board', 'display board', 'sensor', 'thermostat', 'switch', 'timer'},
            'power': {'contactor', 'relay', 'transformer', 'capacitor', 'overload', 'fuse'},
            'wiring': {'wire harness', 'power cord', 'terminal', 'connector', 'ground wire'}
        },
        'consumables': {
            'refrigerant': {'R404A', 'R134a', 'R290', 'R448A', 'R449A', 'R452A'},
            'lubricants': {'compressor oil', 'mineral oil', 'POE oil', 'grease', 'lubricant'},
            'cleaning': {'sanitizer', 'descaler', 'cleaner', 'degreaser', 'scale remover'},
            'water_treatment': {'water softener', 'scale inhibitor', 'antimicrobial', 'pH balancer'}
        }
    }
    
    def find_parts_consumables(text):
        """Find parts and consumables mentioned in the text."""
        text_lower = text.lower()
        found_items = defaultdict(list)
        
        for category, subcategories in parts_consumables.items():
            for subcategory, items in subcategories.items():
                for item in items:
                    if item.lower() in text_lower:
                        found_items[category].append(f"{subcategory}:{item}")
        
        return dict(found_items)
    
    # Define maintenance types with more detailed categories
    maintenance_types = {
        'preventive': {
            'inspect', 'clean', 'adjust', 'lubricate', 'tighten',
            'calibrate', 'test', 'check', 'measure', 'scheduled',
            'routine', 'preventative', 'monitor', 'analyze', 'trend',
            'forecast', 'predict', 'assess', 'evaluate', 'diagnose',
            'investigate', 'study'
        },
        'corrective': {
            'repair', 'replace', 'fix', 'rebuild', 'overhaul',
            'restore', 'rectify', 'correct', 'resolve', 'service'
        },
        'emergency': {
            'breakdown', 'failure', 'emergency', 'urgent', 'critical',
            'immediate', 'unplanned', 'unexpected', 'sudden', 'acute'
        }
    }
    
    def determine_maintenance_type(text):
        """Determine the maintenance type based on text analysis."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # First check for emergency as it's highest priority
        if any(indicator in text_lower for indicator in maintenance_types['emergency']):
            return 'emergency'
        
        # Check for maintenance types using word combinations
        found_types = []
        for mtype, indicators in maintenance_types.items():
            # Check for exact phrases first
            if any(indicator in text_lower for indicator in indicators if ' ' in indicator):
                found_types.append(mtype)
                continue
                
            # Then check for individual words
            if any(indicator in words for indicator in indicators if ' ' not in indicator):
                found_types.append(mtype)
        
        if found_types:
            # If multiple types found, prioritize based on specificity
            priority_order = ['emergency', 'corrective', 'preventive']
            for priority_type in priority_order:
                if priority_type in found_types:
                    return priority_type
            
            return found_types[0]
        
        # Additional context-based classification
        if any(problem in text_lower for problems in problem_indicators.values() for problem in problems):
            return 'corrective'
        
        if any(word in words for word in ['schedule', 'periodic', 'routine', 'regular', 'maintenance']):
            return 'preventive'
            
        if any(word in words for word in ['check', 'inspect', 'test', 'monitor']):
            return 'preventive'
            
        if any(word in words for word in ['new', 'install', 'replace', 'upgrade']):
            return 'corrective'
            
        if any(word in words for word in ['adjust', 'modify', 'change', 'improve']):
            return 'corrective'
        
        # Check for action verbs that indicate maintenance
        action_verbs = {'clean', 'adjust', 'lubricate', 'tighten', 'calibrate', 'service'}
        if any(verb in words for verb in action_verbs):
            return 'preventive'
        
        return 'other'

    # Define specific problem types with detailed indicators
    problem_indicators = {
        'mechanical_failure': {
            'seized', 'stuck', 'broken', 'cracked', 'worn', 'misaligned',
            'loose', 'bent', 'damaged', 'jammed', 'binding', 'stripped'
        },
        'electrical_failure': {
            'short', 'open circuit', 'voltage', 'tripped', 'burnt',
            'no power', 'grounded', 'overload', 'electrical noise',
            'intermittent power', 'voltage drop'
        },
        'ice_quality_issue': {
            'cloudy ice', 'small cubes', 'incomplete cubes', 'malformed',
            'hollow cubes', 'white spots', 'bridging', 'long harvest',
            'slow production', 'uneven size'
        },
        'performance_issue': {
            'slow', 'inefficient', 'reduced', 'low production',
            'poor quality', 'inconsistent', 'erratic', 'unstable',
            'degraded', 'below spec'
        },
        'leakage_issue': {
            'leak', 'drip', 'overflow', 'flood', 'seal failure',
            'gasket leak', 'water loss', 'refrigerant leak',
            'oil leak', 'seepage'
        },
        'contamination': {
            'dirty', 'scale', 'calcium', 'mineral', 'debris', 'buildup',
            'corrosion', 'rust', 'slime', 'mold', 'algae', 'sediment'
        },
        'noise_vibration': {
            'noise', 'vibration', 'rattle', 'squeal', 'bang', 'hum',
            'grinding', 'knocking', 'whistling', 'clicking', 'rumbling'
        },
        'control_system': {
            'error code', 'not responding', 'incorrect', 'erratic',
            'intermittent', 'display error', 'sensor error', 'communication error',
            'program error', 'calibration error'
        },
        'temperature_issue': {
            'hot', 'cold', 'warm', 'freezing', 'not cooling',
            'high temp', 'low temp', 'inconsistent temp',
            'temperature swing', 'poor temperature control'
        },
        'pressure_issue': {
            'high pressure', 'low pressure', 'no pressure', 'pressure drop',
            'pressure fluctuation', 'excessive pressure', 'insufficient pressure',
            'pressure lockout', 'pressure control'
        }
    }
    
    # Define maintenance types with detailed activities
    maintenance_types = {
        'preventive': {
            'inspect', 'clean', 'adjust', 'lubricate', 'tighten',
            'calibrate', 'test', 'check', 'measure', 'scheduled',
            'routine', 'preventative', 'monitor', 'analyze', 'trend',
            'forecast', 'predict', 'assess', 'evaluate', 'diagnose',
            'investigate', 'study'
        },
        'corrective': {
            'repair', 'replace', 'fix', 'corrective', 'breakdown', 'failure', 'malfunction',
            'not working', 'broken', 'damaged', 'failed', 'fault', 'issue', 'problem',
            'restoration', 'overhaul', 'rebuild', 'inoperative', 'defective', 'error',
            'incorrect', 'improper', 'poor', 'abnormal', 'unusual', 'excessive'
        },
        'emergency': {
            'emergency', 'urgent', 'immediate', 'critical', 'breakdown', 'failure', 'asap',
            'safety', 'hazard', 'dangerous', 'severe', 'crucial', 'priority', 'serious',
            'major', 'significant', 'important', 'vital', 'essential'
        },
        'installation': {
            'install', 'installation', 'setup', 'commissioning', 'startup', 'start-up',
            'new equipment', 'replacement', 'upgrade', 'modification', 'configure',
            'configuration', 'initialize', 'deploy', 'implement'
        },
        'diagnostic': {
            'diagnostic', 'troubleshoot', 'investigate', 'inspection', 'check', 'test',
            'evaluation', 'analysis', 'assessment', 'examination', 'monitor', 'measure',
            'verify', 'validate', 'review', 'audit', 'survey'
        },
        'modification': {
            'modify', 'upgrade', 'enhance', 'improve', 'optimization', 'retrofit',
            'reconfiguration', 'adjustment', 'alteration', 'change', 'update', 'revise',
            'redesign', 'adapt', 'customize', 'tune'
        }
    }
    
    def find_matches(text, term_dict):
        """Find matches in text for terms in term_dict."""
        text_lower = text.lower()
        matches = []
        for category, terms in term_dict.items():
            if any(term.lower() in text_lower for term in terms):
                matches.append(category)
        return matches
    
    # Combine all text fields for analysis
    full_text = f"{text} {obj} {verb}".lower()
    
    # Find all matches
    found_components = find_matches(full_text, components)
    found_problems = find_matches(full_text, problem_indicators)
    found_parts = find_parts_consumables(full_text)
    
    # Determine maintenance type
    maintenance_type = determine_maintenance_type(full_text)
    
    # Determine severity based on various factors
    severity_indicators = {
        'high': {'emergency', 'critical', 'urgent', 'failure', 'breakdown', 'safety',
                'immediate', 'severe', 'major', 'serious'},
        'medium': {'degraded', 'warning', 'attention', 'minor failure', 'reduced',
                  'inconsistent', 'poor', 'issue'},
        'low': {'routine', 'normal', 'scheduled', 'preventive', 'adjust', 'check',
                'inspect', 'clean', 'monitor'}
    }
    
    # Determine severity
    severity = 'normal'
    for level, indicators in severity_indicators.items():
        if any(indicator in full_text for indicator in indicators):
            severity = level
            break
    
    return {
        'components': found_components,
        'problem_types': found_problems,
        'maintenance_type': maintenance_type,
        'severity': severity,
        'parts_consumables': found_parts,
        'timestamp': pd.Timestamp.now()  # Add timestamp for tracking
    }

def add_percentage(df, count_column='Count'):
    total = df[count_column].sum()
    df['Percentage'] = (df[count_column] / total * 100).round(2)
    return df

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
        component_stats = defaultdict(Counter)
        
        # Process each maintenance record
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
            text = row['Description']
            if pd.notna(text) and text.strip():
                analysis = analyze_root_cause('', '', text)
                
                # Update statistics
                component_stats['maintenance_types'].update([analysis['maintenance_type']])
        
        # Create DataFrames for analysis results
        maintenance_types_df = pd.DataFrame([
            {'Type': mtype, 'Count': count}
            for mtype, count in component_stats['maintenance_types'].items()
        ])
        
        # Add percentage calculations
        maintenance_types_df = add_percentage(maintenance_types_df)
        maintenance_types_df = maintenance_types_df.sort_values('Count', ascending=False)
        
        # Save results
        maintenance_types_df.to_csv(f"{output_dir}/maintenance_types.csv", index=False)
        
        # Print completion message
        print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
        print(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
