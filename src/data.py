#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing utilities for the SocraticChat dataset.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

def load_socratic_dataset(dataset_name, split="train", sample_size=None):
    """
    Load the SocraticChat dataset.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        split (str): Dataset split to load.
        sample_size (int, optional): Number of samples to load.
        
    Returns:
        Dataset: The loaded dataset.
    """
    if sample_size:
        split = f"{split}[0:{sample_size}]"
    
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def format_socratic_conversations(example, tokenizer):
    """
    Format conversations from the SocraticChat dataset for the model.
    
    Args:
        example (dict): Example from the dataset.
        tokenizer (AutoTokenizer): Tokenizer to use for formatting.
        
    Returns:
        dict: Formatted example.
    """
    formatted_conversations = []
    
    for converse in example['converstations']:
        role = converse['from']
        content = converse['value']
        
        # Map 'gpt' to 'assistant' and 'human' to 'user'
        formatted_role = 'assistant' if role == 'gpt' else 'user'
        
        formatted_conversations.append({
            'role': formatted_role, 
            'content': content
        })
    
    # Apply chat template
    example['text'] = tokenizer.apply_chat_template(formatted_conversations, tokenize=False)
    
    return example

def prepare_dataset(dataset, tokenizer, num_proc=4):
    """
    Prepare the dataset for training.
    
    Args:
        dataset (Dataset): Dataset to prepare.
        tokenizer (AutoTokenizer): Tokenizer to use for formatting.
        num_proc (int): Number of processes to use for mapping.
        
    Returns:
        Dataset: Prepared dataset.
    """
    # Format conversations
    dataset = dataset.map(
        lambda example: format_socratic_conversations(example, tokenizer),
        num_proc=num_proc
    )
    
    return dataset

def get_tokenizer(model_name):
    """
    Get the tokenizer for a model.
    
    Args:
        model_name (str): Name of the model.
        
    Returns:
        AutoTokenizer: The tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def main():
    """Test data loading and processing."""
    # Example usage
    model_name = "meta-llama/Meta-Llama-3-8B"
    dataset_name = "FreedomIntelligence/SocraticChat"
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_name)
    
    # Load dataset
    dataset = load_socratic_dataset(dataset_name, sample_size=10)
    
    # Prepare dataset
    prepared_dataset = prepare_dataset(dataset, tokenizer)
    
    # Print sample
    print(f"Dataset size: {len(prepared_dataset)}")
    print(f"Sample: {prepared_dataset[0]['text'][:500]}...")

if __name__ == "__main__":
    main()
