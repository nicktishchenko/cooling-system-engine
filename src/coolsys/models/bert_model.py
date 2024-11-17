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
    AutoTokenizer,
    AutoModelForTokenClassification,
    __version__ as transformers_version,
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
    try:
        if torch.cuda.is_available():
            logger.info("CUDA device detected")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS device detected")
            return torch.device("mps")
        logger.info("Using CPU device")
        return torch.device("cpu")
    except Exception as e:
        logger.warning(f"Error detecting device, defaulting to CPU: {str(e)}")
        return torch.device("cpu")

def load_model_components(model_name: str) -> Tuple[Optional[BertTokenizer], Optional[BertForTokenClassification]]:
    """Load model components with error handling."""
    try:
        # Set up caching directory
        cache_dir = None  # Let transformers handle caching
        
        # Try loading with Auto classes first
        logger.info("Attempting to load tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=True
            )
        except Exception as e:
            logger.warning(f"AutoTokenizer failed, trying BertTokenizer: {str(e)}")
            tokenizer = BertTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=True
            )
        
        logger.info("Attempting to load model...")
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.warning(f"AutoModel failed, trying BertForTokenClassification: {str(e)}")
            model = BertForTokenClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
        
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}", exc_info=True)
        return None, None

def initialize_model() -> Tuple[Optional[BertTokenizer], Optional[BertForTokenClassification], Optional[torch.device]]:
    """Initialize model with proper error handling and optimizations."""
    try:
        # Log environment information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers_version}")
        
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
                model.eval()  # Set to evaluation mode
                torch.set_grad_enabled(False)  # Disable gradient computation
                logger.info(f"Successfully loaded model: {model_name}")
                return tokenizer, model, device
            
            logger.warning(f"Failed to load model: {model_name}, trying next option...")
        
        raise RuntimeError("All model loading attempts failed")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        return None, None, None
