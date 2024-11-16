"""
BERT model initialization and management module.
"""

from functools import lru_cache
import logging
import sys
from typing import Tuple, Optional

import torch
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    logging as transformers_logging
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress transformer warnings
transformers_logging.set_verbosity_error()

@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Determine the best available device for computation."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model_components(model_name: str) -> Tuple[Optional[BertTokenizer], Optional[BertForTokenClassification]]:
    """Load model components with error handling."""
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForTokenClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}", exc_info=True)
        return None, None

def initialize_model() -> Tuple[Optional[BertTokenizer], Optional[BertForTokenClassification], Optional[torch.device]]:
    """Initialize model with proper error handling and device management."""
    try:
        # Log system information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers_logging.__version__}")
        
        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Model options to try
        model_options = [
            'vblagoje/bert-english-uncased-finetuned-pos',
            'bert-base-uncased'
        ]
        
        # Try loading models
        for model_name in model_options:
            logger.info(f"Attempting to load model: {model_name}")
            tokenizer, model = load_model_components(model_name)
            
            if tokenizer is not None and model is not None:
                # Move model to device
                model = model.to(device)
                logger.info(f"Successfully loaded model: {model_name}")
                return tokenizer, model, device
            
            logger.warning(f"Failed to load model: {model_name}, trying next option...")
        
        raise RuntimeError("All model loading attempts failed")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        return None, None, None
