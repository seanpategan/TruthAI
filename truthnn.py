import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np


class TruthNN(nn.Module):
    """BERT-based neural network for political bias analysis.

    Uses pre-trained BERT to understand context and semantics,
    then adds a regression head for bias prediction.

    Input: Text (raw string)
    Output: x coordinate on political spectrum (-10 to +10)
    """

    def __init__(self, bert_model_name='bert-base-uncased', dropout=0.3):
        super(TruthNN, self).__init__()

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)

        # BERT outputs 768-dimensional embeddings
        bert_hidden_size = self.bert.config.hidden_size

        # Regression head for bias prediction
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bert_hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Single output: x-axis

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        """Forward pass through BERT and regression head.

        Args:
            input_ids: Tokenized input IDs from BERT tokenizer
            attention_mask: Attention mask for padding

        Returns:
            x: Bias score in range [-10, 10]
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token embedding (first token)
        pooled_output = outputs.pooler_output  # Shape: (batch_size, 768)

        # Pass through regression head
        x = self.dropout(pooled_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x)) * 10  # Scale to [-10, 10]

        return x

    def predict(self, text, tokenizer, max_length=512):
        """Make a prediction given raw text.

        Args:
            text: Raw news article text (string)
            tokenizer: BertTokenizer instance
            max_length: Maximum sequence length

        Returns:
            x: Bias score in range [-10, 10]
        """
        # Tokenize
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Forward pass
        self.eval()
        with torch.no_grad():
            output = self.forward(
                encoding['input_ids'],
                encoding['attention_mask']
            )
            x = output[0][0].item()

        return x

    def predict_with_confidence(self, text, tokenizer, max_length=512, n_iterations=30):
        """Make a prediction with confidence estimate using Monte Carlo dropout.

        Args:
            text: Raw news article text (string)
            tokenizer: BertTokenizer instance
            max_length: Maximum sequence length
            n_iterations: Number of forward passes for confidence estimation

        Returns:
            Tuple of (x, confidence_percentage)
        """
        # Tokenize
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Enable dropout for uncertainty estimation
        self.train()

        # Run multiple forward passes
        predictions = []
        for _ in range(n_iterations):
            with torch.no_grad():
                output = self.forward(
                    encoding['input_ids'],
                    encoding['attention_mask']
                )
                predictions.append(output[0][0].item())

        predictions = np.array(predictions)

        # Calculate mean prediction
        x = predictions.mean()

        # Calculate standard deviation (uncertainty)
        std_x = predictions.std()

        # Convert uncertainty to confidence percentage
        max_expected_std = 5.0
        confidence = max(0, min(100, 100 * (1 - std_x / max_expected_std)))

        # Set model back to eval mode
        self.eval()

        return x, confidence

    def get_influential_words(self, text, tokenizer, max_length=512, top_n=10):
        """Get the most influential words/tokens for the prediction.

        Uses gradients to identify important tokens.

        Args:
            text: Raw news article text (string)
            tokenizer: BertTokenizer instance
            max_length: Maximum sequence length
            top_n: Number of top influential words to return

        Returns:
            Dictionary with 'left_words' and 'right_words'
        """
        # Tokenize
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get tokens for interpretation
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

        # Enable gradient computation
        input_ids = encoding['input_ids'].clone().detach().requires_grad_(False)
        attention_mask = encoding['attention_mask']

        # Forward pass with gradient tracking
        self.eval()
        self.zero_grad()

        # Get embeddings with gradient tracking
        embeddings = self.bert.embeddings(input_ids)
        embeddings = embeddings.clone().detach().requires_grad_(True)

        # Forward pass through BERT
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )

        # Get regression output
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        output = self.tanh(self.fc3(x)) * 10

        # Get gradients
        output[0, 0].backward()
        gradients = embeddings.grad

        if gradients is not None:
            # Calculate importance score (L2 norm of gradients)
            importance = gradients.norm(dim=2)[0].detach().numpy()

            # Filter out special tokens and padding
            valid_indices = []
            valid_tokens = []
            valid_importance = []

            for i, (token, imp, mask) in enumerate(zip(tokens, importance, attention_mask[0])):
                if mask == 1 and token not in ['[CLS]', '[SEP]', '[PAD]']:
                    valid_indices.append(i)
                    valid_tokens.append(token)
                    valid_importance.append(imp)

            # Use gradient sign to determine left/right influence
            signed_influence = gradients.sum(dim=2)[0].detach().numpy()

            if len(valid_tokens) > 0:
                left_words = []
                right_words = []

                for i, (token, imp, mask) in enumerate(zip(valid_tokens, valid_importance, [signed_influence[vi] for vi in valid_indices])):
                    clean_token = token.replace('##', '')
                    if len(clean_token) > 2:
                        if mask < 0:
                            left_words.append((clean_token, imp))
                        else:
                            right_words.append((clean_token, imp))

                # Sort each list by importance magnitude
                left_words.sort(key=lambda x: x[1], reverse=True)
                right_words.sort(key=lambda x: x[1], reverse=True)
                left_words = left_words[:top_n]
                right_words = right_words[:top_n]
            else:
                left_words = []
                right_words = []
        else:
            left_words = []
            right_words = []

        return {
            'left_words': left_words,
            'right_words': right_words
        }


def load_truthnn(model_path, tokenizer_name='bert-base-uncased'):
    """Load a trained BERT model from file.

    Args:
        model_path: Path to saved model (.pth file)
        tokenizer_name: Name of BERT tokenizer to use

    Returns:
        model: Loaded TruthNN
        tokenizer: BertTokenizer
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    # Handle both checkpoint dict and raw state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        tokenizer_name = checkpoint.get('bert_model_name', tokenizer_name)
    else:
        state_dict = checkpoint

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = TruthNN(bert_model_name=tokenizer_name)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer
