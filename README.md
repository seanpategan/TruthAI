# TruthAI — News Article Political Bias Analyzer

<!-- Add a screenshot: save as screenshot.png in the project root -->
<!-- ![TruthAI Screenshot](screenshot.png) -->

Analyzes news articles for political bias using a fine-tuned BERT neural network. Paste text or provide a URL — the app scrapes the article and classifies it on a left-right political spectrum.

Works on **macOS**, **Windows**, and **Linux**.

## How It Works

TruthAI uses a model called **TruthNN** — a [BERT](https://arxiv.org/abs/1810.04805) transformer fine-tuned on ~8,000 politically labeled news articles. When you submit an article, the model:

1. **Tokenizes** the text into subword pieces using BERT's WordPiece vocabulary
2. **Encodes** the tokens through 12 transformer layers, building contextual representations of each word
3. **Classifies** the article using a regression head on top of BERT's `[CLS]` token embedding
4. **Estimates confidence** by running 30 forward passes with dropout enabled ([Monte Carlo Dropout](https://arxiv.org/abs/1506.02142)) and measuring prediction variance
5. **Identifies influential words** using gradient-based attribution — computing how much each token pushes the score left or right

The output is a bias score from **-10** (far-left) to **+10** (far-right), a confidence percentage, and a list of the most influential words in the article.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Training Data

Pull labeled data from Hugging Face:

```bash
# AllSides dataset (~5K articles, editor-labeled per article)
python data_preparation.py --allsides

# BABE dataset (expert-annotated sentences)
python data_preparation.py --babe

# Both at once
python data_preparation.py --all
```

Merge local CSV files into the combined dataset (automatically deduplicates):

```bash
python data_preparation.py new_articles.csv

# Deduplicate the existing combined file
python data_preparation.py
```

### 2. Train the Model

```bash
python train_bert_improved.py combined_full_articles_training.csv
```

Auto-detects the best available device:
- **NVIDIA GPU** (CUDA) — fastest
- **Apple Silicon** (MPS) — fast on M1/M2/M3/M4 Macs
- **CPU** — works everywhere, uses half your cores to keep the machine usable

Training runs for up to 20 epochs with early stopping. Saves the best model to `truthnn.pth`.

### 3. Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** — enter a URL or paste article text to analyze.

### JSON API

```bash
# Analyze by URL
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# Analyze by text
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Article text here..."}'
```

Example response:

```json
{
  "coordinates": {"x": -4.52},
  "confidence": 78.3,
  "classification": "Left",
  "model_type": "TruthNN",
  "influential_words": {
    "left": ["inequality", "reform", "workers"],
    "right": ["taxes", "enforcement"]
  }
}
```

## Model Details

### Architecture

```
Input Text
    |
BERT Tokenizer (WordPiece, max 128 tokens)
    |
BERT Base (12 layers, 768-dim hidden, 110M params)
    |
[CLS] Token Embedding (768-dim)
    |
Dropout (0.3) → Linear (768 → 256) → ReLU
    |
Dropout (0.3) → Linear (256 → 128) → ReLU
    |
Linear (128 → 1) → Tanh × 10
    |
Bias Score (-10 to +10)
```

### Training

- **Base model**: `bert-base-uncased` (Google, 110M parameters)
- **Training data**: ~8,000 articles from AllSides, BABE, and manually collected sources
- **Loss function**: Mean Squared Error (regression)
- **Optimizer**: AdamW (lr=2e-5, weight decay=0.01)
- **Balanced sampling**: Inverse-frequency weighting so underrepresented bias categories get more training time
- **Data augmentation**: Random truncation of 20% of training samples to improve robustness on partial articles
- **Early stopping**: Patience of 5 epochs on validation loss

### Bias Scale

| Score | Classification |
|-------|---------------|
| -10 to -6 | Far-Left |
| -6 to -3 | Left |
| -3 to +3 | Center |
| +3 to +6 | Right |
| +6 to +10 | Far-Right |

### Dataset Format

```csv
text,bias_x
"Article text here...",-6.0
"Another article...",4.0
```

Or provide `text` and `source` columns — known sources are auto-mapped to bias scores:

```csv
text,source
"Article text here...",cnn
"Another article...",foxnews
```

## Project Structure

```
python_nn_news_bias/
├── app.py                              # Flask web app (URL scraping + classification)
├── truthnn.py                       # TruthNN model definition + inference
├── data_preparation.py                 # Dataset loading, merging, Hugging Face import
├── train_bert_improved.py              # Training script (CUDA / MPS / CPU)
├── combined_full_articles_training.csv # Training dataset (~8K articles)
├── templates/
│   └── index.html                      # Web UI
├── requirements.txt                    # Dependencies
└── README.md
```

## Limitations

- Most training labels are source-based — every article from the same outlet gets the same score regardless of content
- Political bias is inherently subjective and multidimensional; a single left-right score is a simplification
- URL scraping may not work on paywalled or JavaScript-rendered sites
- Confidence estimates can be miscalibrated on out-of-distribution text (e.g., non-English articles, opinion columns, satire)
- Per-article human labels would significantly improve accuracy
