# 🎯 Gap Filling with Masked Language Modeling

An intelligent text completion tool using BERT's Masked Language Modeling (MLM) to predict missing words in sentences based on context.

## 🌟 Features

- **Context-Aware Predictions**: Uses bidirectional context to predict masked words
- **Confidence Scores**: Shows probability for each prediction
- **Multiple Modes**: Interactive examples, custom input, and batch processing
- **Easy to Use**: Simple command-line interface
- **Pre-trained Model**: Uses BERT-base-uncased (110M parameters)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/gap-filling-mlm.git
cd gap-filling-mlm
```

2. Create virtual environment (recommended):
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the basic version:
```bash
python gap_filling_basic.py
```

## 📖 How It Works

Masked Language Modeling (MLM) is a technique where:
1. A word in a sentence is replaced with `[MASK]` token
2. The model predicts the original word using surrounding context
3. Predictions are ranked by confidence score

Example:
- Input: `"The cat sat on the [MASK]."`
- Predictions: mat (85%), floor (78%), chair (72%), etc.

## 🎓 Use Cases

- **Education**: Language learning exercises
- **Content Creation**: Autocomplete suggestions
- **Data Augmentation**: Generate training data
- **Grammar Checking**: Suggest correct words
- **Text Analysis**: Understand word relationships

## 📊 Model Information

- **Model**: BERT-base-uncased
- **Parameters**: 110 million
- **Training Data**: BooksCorpus + English Wikipedia
- **Vocabulary**: 30,000 tokens

## 🛠️ Project Structure