# Small LLM: Train a Tiny Transformer from Scratch

This project demonstrates how to train a small language model (LLM) using a transformer architecture, inspired by GPT-2, on the TinyStories dataset. It is designed for educational purposes and hands-on experimentation with modern NLP techniques.

## Features
- Loads and tokenizes the TinyStories dataset using HuggingFace Datasets and tiktoken
- Implements a transformer-based model from scratch in PyTorch
- Efficient data handling with memory-mapped files for large datasets
- Training loop with mixed precision, gradient accumulation, and learning rate scheduling
- Model checkpointing and inference for text generation

## Project Structure
```
small_llm/
├── main.py                # Main script (if any)
├── pyproject.toml         # Python project metadata
├── README.md              # Project documentation
├── samll_llm_from_scratch.ipynb  # Jupyter notebook with full code and explanations
├── .gitignore             # Ignore large/model/data files
├── train.bin              # Training data (ignored by git)
├── validation.bin         # Validation data (ignored by git)
```

## Getting Started
1. **Clone the repository:**
   ```sh
   git clone https://github.com/lakshyadaulani/hands-on-llm.git
   cd hands-on-llm/small_llm
   ```
2. **Create a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   # or use pyproject.toml with poetry/pip-tools
   ```
4. **Run the notebook:**
   Open `samll_llm_from_scratch.ipynb` in Jupyter or VS Code and follow the steps.

## Usage
- The notebook walks through:
  - Loading and tokenizing TinyStories
  - Creating memory-mapped training/validation files
  - Building the transformer model
  - Training with advanced PyTorch features
  - Generating text with the trained model

## Notes
- Large files (`train.bin`, `validation.bin`) are ignored by git and should be generated locally.
- For best performance, use a machine with a CUDA-enabled GPU.
- The code is modular and easy to adapt for other datasets or model sizes.

## License
MIT License

## Acknowledgements
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [tiktoken](https://github.com/openai/tiktoken)
- [PyTorch](https://pytorch.org/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)

---
For questions or contributions, open an issue or pull request on GitHub.
