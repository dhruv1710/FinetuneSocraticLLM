#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model utilities for fine-tuning Llama 3 on the SocraticChat dataset.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import setup_chat_format
from .utils import print_model_info

def get_bnb_config(config):
    """
    Get the BitsAndBytes configuration for quantization.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        BitsAndBytesConfig: Configuration for BitsAndBytes quantization.
    """
    # Set up compute dtype
    torch_dtype = torch.float16
    
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )
    
    return bnb_config

def get_lora_config(config):
    """
    Get the LoRA configuration.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        LoraConfig: Configuration for LoRA.
    """
    peft_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=config.get("lora_target_modules", 
                                ['up_proj', 'down_proj', 'gate_proj', 
                                 'k_proj', 'q_proj', 'v_proj', 'o_proj'])
    )
    
    return peft_config

def load_base_model(model_name, config):
    """
    Load the base model with quantization.
    
    Args:
        model_name (str): Name of the model to load.
        config (dict): Configuration dictionary.
        
    Returns:
        AutoModelForCausalLM: The loaded model.
    """
    # Get BitsAndBytes config
    bnb_config = get_bnb_config(config)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=config.get("attn_implementation", "flash_attention_2")
    )
    
    return model

def apply_lora(model, config):
    """
    Apply LoRA to the model.
    
    Args:
        model (AutoModelForCausalLM): The model to apply LoRA to.
        config (dict): Configuration dictionary.
        
    Returns:
        PeftModel: The model with LoRA applied.
    """
    # Get LoRA config
    peft_config = get_lora_config(config)
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    return model

def setup_model_for_training(model_name, config, verbose=True):
    """
    Set up the model and tokenizer for training.
    
    Args:
        model_name (str): Name of the model to load.
        config (dict): Configuration dictionary.
        verbose (bool): Whether to print model information.
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load base model
    model = load_base_model(model_name, config)
    
    # Load tokenizer
    tokenizer = AutoModelForCausalLM.from_pretrained(model_name).tokenizer
    
    # Set up chat format
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    # Apply LoRA
    model = apply_lora(model, config)
    
    # Print model information
    if verbose:
        print_model_info(model)
    
    return model, tokenizer

def load_finetuned_model(base_model_name, model_path):
    """
    Load a fine-tuned model.
    
    Args:
        base_model_name (str): Name of the base model.
        model_path (str): Path to the fine-tuned model.
        
    Returns:
        tuple: (model, tokenizer)
    """
    from peft import PeftModel
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoModelForCausalLM.from_pretrained(base_model_name).tokenizer
    
    # Load fine-tuned model
    model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer
