#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the Socratic LLM fine-tuning project.
"""

import os
import yaml
import torch
from pathlib import Path

def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_yaml_config(config, config_path):
    """Save configuration to YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def get_device():
    """Get the device to use for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def ensure_dir(directory):
    """Ensure that a directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_conversation(conversation_data):
    """Format conversation data for the model."""
    formatted = []
    for turn in conversation_data:
        role = turn.get("from", "")
        content = turn.get("value", "")
        
        # Map roles to the format expected by the model
        if role.lower() in ["gpt", "assistant", "bot"]:
            formatted_role = "assistant"
        else:
            formatted_role = "user"
            
        formatted.append({
            "role": formatted_role,
            "content": content
        })
    
    return formatted

def print_model_info(model):
    """Print information about the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    # Print model architecture summary
    print("\nModel Architecture:")
    for name, module in model.named_children():
        print(f"- {name}: {module.__class__.__name__}")
