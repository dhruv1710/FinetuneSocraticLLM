#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for fine-tuning Llama 3 8B on the SocraticChat dataset.
"""

import os
import argparse
import yaml
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import setup_chat_format
import wandb

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 on SocraticChat dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="models/socratic-llama-3-8b",
                        help="Directory to save the model")
    parser.add_argument("--sample_size", type=int, default=500,
                        help="Number of samples to use from the dataset")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_wandb(project_name):
    """Set up Weights & Biases for experiment tracking."""
    load_dotenv()
    wandb_token = os.getenv("WANDB_TOKEN")
    if wandb_token:
        wandb.login(key=wandb_token)
        run = wandb.init(
            project=project_name,
            job_type="training",
            anonymous="allow"
        )
        return run
    else:
        print("WANDB_TOKEN not found in environment. Skipping wandb initialization.")
        return None

def setup_huggingface():
    """Set up Hugging Face authentication."""
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    else:
        print("HUGGINGFACE_TOKEN not found in environment. You may encounter issues accessing models.")

def load_and_prepare_dataset(sample_size, tokenizer):
    """Load and prepare the SocraticChat dataset."""
    # Load dataset
    dataset = load_dataset('FreedomIntelligence/SocraticChat', split=f'train[0:{sample_size}]')
    
    # Define formatting function
    def formatting_prompts_func(example):
        formatted_conversations = []
        for converse in example['converstations']:
            role = converse['from']
            # Map 'gpt' to 'assistant' and 'human' to 'user'
            content = converse['value']
            formatted_role = 'assistant' if role == 'gpt' else 'user'
            formatted_conversations.append({'role': formatted_role, 'content': content})
        
        example['text'] = tokenizer.apply_chat_template(formatted_conversations, tokenize=False)
        return example
    
    # Apply formatting
    formatted_dataset = dataset.map(formatting_prompts_func, num_proc=4)
    return formatted_dataset

def setup_model(model_name, config):
    """Set up the model and tokenizer with QLoRA configuration."""
    # Set up compute dtype
    torch_dtype = torch.float16
    
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=config.get("attn_implementation", "flash_attention_2")
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up chat format
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    # LoRA config
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
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def train(model, tokenizer, dataset, output_dir, config):
    """Train the model."""
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        max_steps=config.get("max_steps", 100),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 50),
        save_total_limit=config.get("save_total_limit", 3),
        fp16=config.get("fp16", True),
        report_to=config.get("report_to", "wandb"),
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    
    return trainer

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up Hugging Face and Weights & Biases
    setup_huggingface()
    run = setup_wandb(config.get("project_name", "Fine-tune Llama 3 8B on Socratic Chat Dataset"))
    
    # Set up model and tokenizer
    model_name = config.get("model_name", "meta-llama/Meta-Llama-3-8B")
    model, tokenizer = setup_model(model_name, config)
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(args.sample_size, tokenizer)
    
    # Train model
    trainer = train(model, tokenizer, dataset, args.output_dir, config)
    
    # Close wandb run
    if run:
        wandb.finish()
    
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
