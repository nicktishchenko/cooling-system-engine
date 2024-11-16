"""
Tests for the BERT model initialization and management.
"""

import pytest
import torch
from coolsys.models.bert_model import get_device, initialize_model

def test_get_device():
    """Test device selection logic."""
    device = get_device()
    assert isinstance(device, torch.device)
    # Device should be either mps, cuda, or cpu
    assert str(device) in ['mps', 'cuda', 'cpu']

def test_initialize_model():
    """Test model initialization."""
    tokenizer, model, device = initialize_model()
    
    # Check if initialization was successful
    assert tokenizer is not None, "Tokenizer initialization failed"
    assert model is not None, "Model initialization failed"
    assert device is not None, "Device initialization failed"
    
    # Check if model is on the correct device
    assert next(model.parameters()).device == device

def test_model_output():
    """Test basic model functionality."""
    tokenizer, model, device = initialize_model()
    
    # Test input
    text = "The ice machine needs maintenance."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Check output shape
    assert outputs.logits.shape[1] == len(tokenizer.tokenize(text))

if __name__ == "__main__":
    pytest.main([__file__])
