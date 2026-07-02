import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer
import numpy as np
import sys
import os
from pathlib import Path
from collections import Counter

# Use half the CPU threads so the machine stays usable
import multiprocessing
torch.set_num_threads(max(1, multiprocessing.cpu_count() // 2))

from truthnn import TruthNN
from data_preparation import load_dataset, prepare_training_data


class BertBiasDataset(Dataset):
    """PyTorch dataset for BERT-based bias analysis with augmentation."""

    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False):
        """Initialize dataset.

        Args:
            texts: List of text strings
            labels: Array of shape (n, 1) with x coordinates (left/right)
            tokenizer: BertTokenizer instance
            max_length: Maximum sequence length for BERT
            augment: Whether to apply text augmentation
        """
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

        # Ensure labels are 1D (x only)
        if len(self.labels.shape) > 1 and self.labels.shape[1] > 1:
            self.labels = self.labels[:, 0:1]
        elif len(self.labels.shape) == 1:
            self.labels = self.labels.reshape(-1, 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Simple augmentation: randomly truncate 20% of training examples
        # This helps the model learn from partial articles
        if self.augment and np.random.random() < 0.2:
            words = text.split()
            if len(words) > 100:
                # Keep 60-90% of the article
                keep_ratio = np.random.uniform(0.6, 0.9)
                keep_words = int(len(words) * keep_ratio)
                # Randomly choose start point
                start = np.random.randint(0, len(words) - keep_words + 1)
                text = ' '.join(words[start:start + keep_words])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }


def get_class_weights(labels, num_bins=7):
    """Calculate class weights for imbalanced dataset.

    Args:
        labels: Array of bias_x values
        num_bins: Number of bins to group labels into

    Returns:
        weights: Array of weights for each sample
    """
    # Bin the continuous labels
    # Bins: [-10, -6), [-6, -3), [-3, -1), [-1, 1), [1, 3), [3, 6), [6, 10]
    bins = [-10, -6, -3, -1, 1, 3, 6, 10]
    binned_labels = np.digitize(labels.flatten(), bins) - 1

    # Count samples in each bin
    bin_counts = Counter(binned_labels)

    # Calculate inverse frequency weights
    total_samples = len(labels)
    weights = np.zeros(len(labels))

    for i, bin_idx in enumerate(binned_labels):
        # Weight = total / (num_bins * count_in_bin)
        weights[i] = total_samples / (num_bins * bin_counts[bin_idx])

    return weights


def train_improved_bert(
    dataset_path,
    bert_model='bert-base-uncased',
    batch_size=8,
    epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    patience=5,
    model_save_path='truthnn.pth'
):
    """Train BERT model with improvements for better accuracy.

    Improvements:
    - BERT-large instead of BERT-base
    - Class-balanced sampling
    - Data augmentation
    - More epochs with early stopping
    - Gradient clipping
    """
    print(f"=" * 60)
    print(f"IMPROVED BERT TRAINING")
    print(f"=" * 60)
    print(f"Model: {bert_model}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"=" * 60)

    # Load and prepare data
    print(f"\nLoading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)
    train_df, test_df = prepare_training_data(df, test_split=0.2)

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Analyze data distribution
    print(f"\nData distribution:")
    print(f"Far-left (-10 to -6): {len(train_df[train_df['bias_x'] <= -6])}")
    print(f"Left (-6 to -3): {len(train_df[(train_df['bias_x'] > -6) & (train_df['bias_x'] <= -3)])}")
    print(f"Center-left (-3 to -1): {len(train_df[(train_df['bias_x'] > -3) & (train_df['bias_x'] <= -1)])}")
    print(f"Center (-1 to 1): {len(train_df[(train_df['bias_x'] > -1) & (train_df['bias_x'] < 1)])}")
    print(f"Center-right (1 to 3): {len(train_df[(train_df['bias_x'] >= 1) & (train_df['bias_x'] < 3)])}")
    print(f"Right (3 to 6): {len(train_df[(train_df['bias_x'] >= 3) & (train_df['bias_x'] < 6)])}")
    print(f"Far-right (6 to 10): {len(train_df[train_df['bias_x'] >= 6])}")

    # Initialize tokenizer
    print(f"\nLoading BERT tokenizer ({bert_model})...")
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # Create datasets WITH augmentation for training
    train_dataset = BertBiasDataset(
        train_df['text'].tolist(),
        train_df[['bias_x']].values,
        tokenizer,
        augment=True  # Augment training data
    )

    test_dataset = BertBiasDataset(
        test_df['text'].tolist(),
        test_df[['bias_x']].values,
        tokenizer,
        augment=False  # No augmentation for test data
    )

    # Calculate sample weights for balanced sampling
    print("\nCalculating class weights for balanced sampling...")
    sample_weights = get_class_weights(train_df[['bias_x']].values)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create data loaders with weighted sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    print(f"\nInitializing {bert_model} model...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = TruthNN(bert_model_name=bert_model, dropout=0.3)
    model = model.to(device)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Training loop with early stopping
    best_test_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item()
            if device.type == 'mps':
                torch.mps.synchronize()
            train_loss += batch_loss
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        test_loss = 0.0
        test_batches = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_batches += 1

                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / test_batches

        # Calculate MAE (Mean Absolute Error) for better interpretability
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_labels)))

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Test MAE: {mae:.2f} points (on [-10, 10] scale)")

        # Learning rate scheduling
        scheduler.step(avg_test_loss)

        # Early stopping check
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0

            # Save best model
            print(f"  ✓ New best model! Saving to {model_save_path}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'bert_model_name': bert_model,
                'test_loss': avg_test_loss,
                'test_mae': mae,
                'epoch': epoch + 1
            }, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after {epoch + 1} epochs")
                break

        print("-" * 60)

    print(f"\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Model saved to: {model_save_path}")
    print(f"=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_bert_improved.py <dataset.csv>")
        print("Example:")
        print("  python train_bert_improved.py combined_full_articles_training.csv")
        sys.exit(1)

    dataset_path = sys.argv[1]

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Train with improved settings
    train_improved_bert(
        dataset_path=dataset_path,
        bert_model='bert-base-uncased',
        batch_size=8,
        epochs=20,
        learning_rate=2e-5,
        patience=5,
        model_save_path='truthnn.pth'
    )


if __name__ == "__main__":
    main()
