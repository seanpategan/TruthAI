import pandas as pd
import numpy as np
import os
from pathlib import Path



# Mapping of media sources to political bias (left/right)
# Approximate ratings based on AllSides and Media Bias/Fact Check
# x: left (-10) to right (+10)
SOURCE_BIAS_MAP = {
    # Far-left sources
    'jacobin': -9.0,
    'commondreams': -8.0,
    'democracynow': -8.0,
    'thenation': -7.0,

    # Left sources
    'msnbc': -6.0,
    'huffpost': -6.0,
    'theguardian': -5.0,
    'vox': -5.0,
    'slate': -5.0,
    'motherjones': -6.0,
    'cnn': -3.0,
    'nytimes': -3.0,
    'washingtonpost': -3.0,

    # Center sources
    'bbc': 0.0,
    'reuters': 0.0,
    'apnews': 0.0,
    'npr': -1.0,
    'usatoday': 0.0,
    'csmonitor': 0.0,

    # Right sources
    'foxnews': 6.0,
    'wallstreetjournal': 4.0,
    'nationalreview': 6.0,
    'washingtonexaminer': 5.0,
    'nypost': 5.0,
    'washingtontimes': 6.0,

    # Far-right sources
    'breitbart': 8.0,
    'dailycaller': 7.0,
    'newsmax': 8.0,
    'oann': 9.0,
    'thefederalist': 7.0,
}


def create_sample_dataset():
    """Create a sample dataset with example articles for each bias category."""
    sample_data = [
        {
            'text': 'Workers must unite against corporate greed and demand fair wages. The billionaire class continues to exploit labor while hoarding wealth. We need radical wealth redistribution and worker ownership of the means of production.',
            'source': 'jacobin',
            'bias_x': -9.0,
        },
        {
            'text': 'The climate crisis demands immediate action through a Green New Deal. Corporate polluters must be held accountable. We need to transition away from fossil fuels and capitalism is failing to address this existential threat.',
            'source': 'commondreams',
            'bias_x': -8.0,
        },
        {
            'text': 'Healthcare is a human right, not a privilege. The Affordable Care Act was a step forward, but we need to expand coverage and make prescription drugs more affordable for all Americans.',
            'source': 'msnbc',
            'bias_x': -6.0,
        },
        {
            'text': 'Income inequality continues to grow as the wealthy benefit from tax cuts while working families struggle. Progressive taxation could help fund education and infrastructure investments.',
            'source': 'vox',
            'bias_x': -5.0,
        },
        {
            'text': 'Gun violence is a public health crisis requiring comprehensive reform including universal background checks and assault weapon restrictions to protect communities.',
            'source': 'theguardian',
            'bias_x': -5.0,
        },
        {
            'text': 'The Federal Reserve announced an interest rate decision today following months of economic uncertainty. Economists are divided on the impacts for inflation and employment.',
            'source': 'reuters',
            'bias_x': 0.0,
        },
        {
            'text': 'Congress passed a bipartisan infrastructure bill with support from both parties. The legislation includes funding for roads, bridges, and broadband internet access.',
            'source': 'apnews',
            'bias_x': 0.0,
        },
        {
            'text': 'Unemployment figures released today show mixed results across different sectors. Manufacturing jobs increased while retail positions declined in the latest quarterly report.',
            'source': 'bbc',
            'bias_x': 0.0,
        },
        {
            'text': 'Lower taxes and reduced regulations are driving economic growth and job creation. Free market solutions are more effective than government intervention in addressing economic challenges.',
            'source': 'wallstreetjournal',
            'bias_x': 4.0,
        },
        {
            'text': 'Border security remains a top priority as illegal immigration continues. Strong enforcement and wall construction are necessary to protect American jobs and national security.',
            'source': 'foxnews',
            'bias_x': 6.0,
        },
        {
            'text': 'The Second Amendment protects fundamental rights. Law-abiding citizens should not face restrictions when criminals ignore gun laws. Focus should be on enforcement, not new regulations.',
            'source': 'nationalreview',
            'bias_x': 6.0,
        },
        {
            'text': 'The mainstream media is completely biased and cannot be trusted. Traditional values are under attack by radical leftists trying to destroy American culture and freedom.',
            'source': 'breitbart',
            'bias_x': 8.0,
        },
        {
            'text': 'Socialist policies will destroy the economy and American way of life. Big government overreach threatens individual liberty. We must fight back against the radical left agenda.',
            'source': 'newsmax',
            'bias_x': 8.0,
        },
    ]

    return pd.DataFrame(sample_data)


def load_dataset(file_path):
    """Load dataset from CSV or JSON file.

    Expected format:
    - CSV: columns [text, bias_x] or [text, source]
    - JSON: array of objects with {text, bias_x} or {text, source}

    Args:
        file_path: Path to dataset file

    Returns:
        DataFrame with columns [text, bias_x]
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # If source column exists but not bias_x, map them
    if 'source' in df.columns and 'bias_x' not in df.columns:
        df['bias_x'] = df['source'].map(lambda s: SOURCE_BIAS_MAP.get(s.lower(), 0.0))

    if 'text' not in df.columns or 'bias_x' not in df.columns:
        raise ValueError(f"Missing required columns. Need: text, bias_x")

    return df[['text', 'bias_x']]


def prepare_training_data(df, test_split=0.2, random_state=42):
    """Prepare training and test datasets.

    Args:
        df: DataFrame with columns [text, bias_x]
        test_split: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_split))
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, test_df


def add_custom_article(text, bias_x, output_file='custom_data.csv'):
    """Add a custom article to dataset file.

    Args:
        text: Article text
        bias_x: Left/right coordinate (-10 to 10)
        output_file: File to append to (will create if doesn't exist)
    """
    if not -10 <= bias_x <= 10:
        raise ValueError("bias_x must be between -10 and 10")

    df = pd.DataFrame([{'text': text, 'bias_x': bias_x}])

    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)

    print(f"Added article to {output_file}")


def fetch_allsides_dataset(sample_size=5000, combined_file='combined_full_articles_training.csv'):
    """Download AllSides dataset from Hugging Face and merge into the combined dataset.

    Uses upasanachatterjee/AllSides-random-split (~29K articles) with
    left/center/right labels mapped to bias_x scores.

    Args:
        sample_size: Number of articles to sample (balanced across labels)
        combined_file: Path to the combined output file
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Run: pip3 install datasets")

    ALLSIDES_LABEL_MAP = {
        0: -7.0,   # left
        1:  0.0,   # center
        2:  7.0,   # right
    }

    print("Downloading AllSides dataset from Hugging Face...")
    ds = hf_load("upasanachatterjee/AllSides-random-split", split="train")
    df = ds.to_pandas()
    print(f"Downloaded {len(df)} articles")

    # Use content_original for fuller text, fall back to content
    text_col = 'content_original' if 'content_original' in df.columns else 'content'
    df = df[[text_col, 'bias']].copy()
    df.columns = ['text', 'bias']

    # Drop empty/short articles
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 50]

    # Map labels to bias_x
    df['bias_x'] = df['bias'].map(ALLSIDES_LABEL_MAP)
    df = df.dropna(subset=['bias_x'])

    # Balanced sample across labels
    per_label = sample_size // 3
    sampled = df.groupby('bias', group_keys=False).apply(
        lambda g: g.sample(n=min(per_label, len(g)), random_state=42)
    )
    df = sampled[['text', 'bias_x']].copy()

    print(f"Sampled {len(df)} articles (balanced across left/center/right)")
    return merge_into_combined(df, combined_file=combined_file)


def fetch_babe_dataset(combined_file='combined_full_articles_training.csv'):
    """Download the BABE dataset from Hugging Face and merge into the combined dataset.

    BABE (Bias Annotations By Experts) contains ~1,700 sentences annotated
    by political scientists with left/center/right labels.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip3 install datasets")

    print("Downloading BABE dataset from Hugging Face...")
    ds = load_dataset("mediabiasgroup/BABE", split="train")
    df = ds.to_pandas()

    # BABE columns: 'text', 'outlet', 'topic', 'label', 'biased_words'
    # 'label' is 'Biased' or 'Non-biased' — we want 'outlet' for bias_x
    # Map outlet names to bias scores using SOURCE_BIAS_MAP
    print(f"Downloaded {len(df)} entries. Mapping outlet bias scores...")

    def map_outlet(outlet):
        if not isinstance(outlet, str):
            return None
        key = outlet.lower().replace(' ', '').replace('.', '').replace('-', '')
        return SOURCE_BIAS_MAP.get(key, None)

    df['bias_x'] = df['outlet'].apply(map_outlet)
    df = df.dropna(subset=['bias_x', 'text'])
    df = df[['text', 'bias_x']].copy()

    print(f"Mapped {len(df)} articles with known outlet bias scores")
    return merge_into_combined(df, combined_file=combined_file)


def merge_into_combined(input_files, combined_file='combined_full_articles_training.csv'):
    """Merge one or more dataset files (or a DataFrame) into the combined dataset.

    Appends new data to the combined file if it exists, then deduplicates
    based on article text content.

    Args:
        input_files: Single file path (str), list of file paths, or a DataFrame
        combined_file: Path to the combined output file

    Returns:
        DataFrame of the final deduplicated combined dataset
    """
    # Accept a DataFrame directly
    if isinstance(input_files, pd.DataFrame):
        input_files = [input_files]
    elif isinstance(input_files, str):
        input_files = [input_files]

    # Load existing combined file if it exists
    frames = []
    if os.path.exists(combined_file):
        existing = pd.read_csv(combined_file)
        frames.append(existing)
        print(f"Loaded {len(existing)} existing articles from {combined_file}")

    # Load each input file or DataFrame
    for f in input_files:
        try:
            if isinstance(f, pd.DataFrame):
                frames.append(f)
                print(f"Added {len(f)} articles from DataFrame")
            else:
                df = load_dataset(f)
                frames.append(df)
                print(f"Loaded {len(df)} articles from {f}")
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not frames:
        print("No data to merge.")
        return pd.DataFrame(columns=['text', 'bias_x'])

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)

    # Deduplicate on normalized text
    combined['_text_norm'] = combined['text'].str.strip().str.lower()
    combined = combined.drop_duplicates(subset='_text_norm', keep='first')
    combined = combined.drop(columns='_text_norm')
    combined = combined.reset_index(drop=True)

    removed = before - len(combined)
    print(f"\nRemoved {removed} duplicate articles")
    print(f"Final dataset: {len(combined)} articles")

    combined.to_csv(combined_file, index=False)
    print(f"Saved to {combined_file}")

    return combined


def save_sample_dataset(output_file='sample_dataset.csv'):
    """Save the sample dataset to a file."""
    df = create_sample_dataset()
    df.to_csv(output_file, index=False)
    print(f"Saved sample dataset to {output_file} ({len(df)} articles)")
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--allsides':
        print("Fetching AllSides dataset from Hugging Face...\n")
        df = fetch_allsides_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == '--babe':
        print("Fetching BABE dataset from Hugging Face...\n")
        df = fetch_babe_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == '--all':
        print("Fetching all datasets from Hugging Face...\n")
        fetch_allsides_dataset()
        df = fetch_babe_dataset()
    elif len(sys.argv) > 1:
        input_files = sys.argv[1:]
        print(f"Merging {len(input_files)} file(s) into combined dataset...\n")
        df = merge_into_combined(input_files)
    else:
        combined = 'combined_full_articles_training.csv'
        if os.path.exists(combined):
            print(f"Deduplicating {combined}...\n")
            df = merge_into_combined([], combined_file=combined)
        else:
            print("No input files provided and no combined dataset found.")
            print("Usage:")
            print("  python3 data_preparation.py --allsides       # pull AllSides dataset (5K articles)")
            print("  python3 data_preparation.py --babe           # pull BABE dataset")
            print("  python3 data_preparation.py --all            # pull both datasets")
            print("  python3 data_preparation.py <file1.csv> ...  # merge local files")
            print("  python3 data_preparation.py                  # deduplicate existing")
            sys.exit(1)

    print(f"\nDataset statistics:")
    print(f"Total articles: {len(df)}")
    print(f"Average bias_x: {df['bias_x'].mean():.2f}")
    print(f"Bias_x range: [{df['bias_x'].min():.2f}, {df['bias_x'].max():.2f}]")
