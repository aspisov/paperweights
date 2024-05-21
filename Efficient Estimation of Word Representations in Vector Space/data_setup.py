from datasets import load_dataset
from nltk import word_tokenize
from collections import Counter
import torch


# Convert tokens to ids
def encode(tokens, vocab):
    return [vocab[token] if token in vocab else vocab["<UNK>"] for token in tokens]


# Decode ids to tokens
def decode(ids, vocab):
    id_to_token = {v: k for k, v in vocab.items()}
    if ids.dim() == 0:
        return id_to_token[ids.item()]
    return [id_to_token[id.item()] for id in ids]
    


# Generate training samples with context windows
def generate_context_samples(ids, vocab, context_window_size):
    input_sequences = []
    target_tokens = []
    # pad sequences
    padded_ids = (
        [vocab["<PAD>"]] * context_window_size
        + ids
        + [vocab["<PAD>"]] * context_window_size
    )

    for i in range(context_window_size, len(padded_ids) - context_window_size):
        context = (
            padded_ids[i - context_window_size : i]
            + padded_ids[i + 1 : i + 1 + context_window_size]
        )
        input_sequences.append(context)
        target_tokens.append(padded_ids[i])

    return input_sequences, target_tokens


# Load and preprocess the dataset
def load_and_preprocess_data(context_window_size):
    # Load Wikipedia dataset
    dataset = load_dataset("wikipedia", "20220301.simple")
    text_samples = dataset["train"]["text"][:1000]

    # Tokenize the text samples
    tokenized_articles = [word_tokenize(article.lower()) for article in text_samples]

    # Count the frequency of each token
    token_counter = Counter()
    for tokens in tokenized_articles:
        token_counter.update(tokens)

    # Select the most common tokens
    most_common_tokens = token_counter.most_common(30000)

    # Create the vocabulary with the most common tokens
    vocab = {
        token: idx + 2 for idx, (token, _) in enumerate(most_common_tokens)
    }
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    input_data = []
    target_data = []

    for tokens in tokenized_articles:
        token_ids = encode(tokens, vocab)
        context_inputs, target_outputs = generate_context_samples(
            token_ids, vocab, context_window_size
        )
        input_data.extend(context_inputs)
        target_data.extend(target_outputs)

    input_data_tensor = torch.tensor(input_data, dtype=torch.long)
    target_data_tensor = torch.tensor(target_data, dtype=torch.long)
    return input_data_tensor, target_data_tensor, vocab
