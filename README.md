# Socratic LLM Fine-tuning

A project for fine-tuning large language models (specifically Llama 3 8B) on the SocraticChat dataset to create a model that can engage in Socratic-style conversations.

## Overview

This project provides tools and scripts for fine-tuning the Meta Llama 3 8B model using the SocraticChat dataset from FreedomIntelligence. The fine-tuning process uses QLoRA (Quantized Low-Rank Adaptation) to efficiently adapt the model while maintaining its performance.

## Features

- Data loading and preprocessing from the SocraticChat dataset
- QLoRA fine-tuning setup for Llama 3 8B
- Integration with Weights & Biases for experiment tracking
- Efficient training with parameter-efficient fine-tuning

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/socratic.git
cd socratic

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Configuration

Set up your API keys in a `.env` file:

```
HUGGINGFACE_TOKEN=your_huggingface_token
WANDB_TOKEN=your_wandb_token
```

### Training

To start the fine-tuning process:

```bash
python src/train.py --config configs/default.yaml
```

### Inference

After training, you can use the fine-tuned model for inference:

```bash
python src/inference.py --model_path path/to/saved/model
```
## Web Interface

### Running the Chat UI

1. Install Streamlit:
   ```bash
   pip install streamlit
2. Run app
   ```bash
   streamlit run src/app.py
   
## Project Structure

```
socratic/
├── configs/              # Configuration files
├── data/                 # Data processing scripts
├── models/               # Model definitions
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── train.py          # Training script
│   ├── inference.py      # Inference script
│   └── utils.py          # Utility functions
├── tests/                # Test cases
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── LICENSE               # License file
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FreedomIntelligence](https://github.com/FreedomIntelligence) for the SocraticChat dataset
- [Meta AI](https://ai.meta.com/) for the Llama 3 model
