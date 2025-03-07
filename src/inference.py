#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for using the fine-tuned Llama 3 model.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import login

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Llama 3 model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Base model name")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    return parser.parse_args()

def setup_huggingface():
    """Set up Hugging Face authentication."""
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("HUGGINGFACE_TOKEN not found in environment. You may encounter issues accessing models.")

def load_model(model_path, base_model):
    """Load the fine-tuned model."""
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Load fine-tuned model
    model = PeftModel.from_pretrained(model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate a response from the model."""
    # Format prompt as chat
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    assistant_response = response.split("assistant")[-1].strip()
    
    return assistant_response

def interactive_mode(model, tokenizer, args):
    """Run in interactive mode."""
    print("Interactive mode. Type 'exit' to quit.")
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Generate response
        response = generate_response(
            model, 
            tokenizer, 
            user_input, 
            max_length=args.max_length, 
            temperature=args.temperature, 
            top_p=args.top_p
        )
        
        # Print response
        print(f"\nAssistant: {response}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up Hugging Face
    setup_huggingface()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    if args.interactive:
        # Run in interactive mode
        interactive_mode(model, tokenizer, args)
    else:
        # Get user input
        user_input = input("Enter your prompt: ")
        
        # Generate response
        response = generate_response(
            model, 
            tokenizer, 
            user_input, 
            max_length=args.max_length, 
            temperature=args.temperature, 
            top_p=args.top_p
        )
        
        # Print response
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()
