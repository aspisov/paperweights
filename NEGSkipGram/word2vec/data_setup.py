"""
Contain functions to load and prepare data for word2vec
"""
# load, process and prepare data for word2vec
from datasets import load_dataset
import nltk
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class SkipGramDataset(Dataset):
    def __init__(self, tokenized_articles, vocab, context_size, word_prob):
        self.word_to_idx = vocab
        self.idx_to_word = {v: k for k, v in vocab.items()}
        self.word_prob = word_prob
        self.target = []
        self.context = []
        for tokens in tokenized_articles:
            token_ids = encode(tokens, vocab)
            self.generate_samples(token_ids, context_size)
        print(f"Generated {len(self.target)/1e6:.2f}M training samples")
            
    # Generate training samples with context windows
    def generate_samples(self, ids, context_size):
        # pad sequences
        pad_idx = self.word_to_idx["<PAD>"]
        padded_ids = ([pad_idx] * context_size + ids + [pad_idx] * context_size)

        for i in range(context_size, len(padded_ids) - context_size):
            target = padded_ids[i]
            # Apply subsampling
            if random.random() > self.word_prob.get(target, 1.0):
                continue
            context = (padded_ids[i - context_size : i] + padded_ids[i + 1 : i + 1 + context_size])
            self.target.extend([target] * len(context))
            self.context.extend(context)
    
    def __len__(self):
        return len(self.target)  
    
    def __getitem__(self, idx):
        return torch.tensor(self.target[idx], dtype=torch.long), torch.tensor(self.context[idx], dtype=torch.long)
    
# Convert tokens to ids
def encode(tokens, vocab):
    return [vocab[token] if token in vocab else vocab["<UNK>"] for token in tokens]

# Decode ids to tokens
def decode(ids, vocab):
    id_to_token = {v: k for k, v in vocab.items()}
    if ids.dim() == 0:
        return id_to_token[ids.item()]
    return [id_to_token[id.item()] for id in ids]

def generate_noise_distribution(token_counter):
    
    num_tokens = sum(count for _, count in token_counter)
    
    # here we add [0] + [0] which are the <PAD> and <UNK> tokens so that unigram_dist is vocab_size
    unigram_dist = [0] + [0] + [count / num_tokens for token, count in token_counter]
    # heuristic from the paper
    unigram_dist = np.array(unigram_dist) ** 0.75 
    Z = unigram_dist.sum() 
    # normalize 
    noise_dist = torch.from_numpy(unigram_dist / Z)
    return noise_dist

def calculate_word_probabilities(token_counter, total_tokens, threshold=1e-5):
    word_prob = {}
    for word, count in token_counter.items():
        frequency = count / total_tokens
        word_prob[word] = min((np.sqrt(frequency / threshold) + 1) * (threshold / frequency), 1.0)
    return word_prob

# Load and preprocess the dataset
def load_and_preprocess_data(vocab_size, amount_of_articles=None):
    nltk.download('punkt')
    
    # Load Wikipedia dataset
    dataset = load_dataset("wikipedia", "20220301.simple")
    
    text_samples = dataset["train"]["text"]
    if amount_of_articles is not None:
        text_samples = text_samples[:amount_of_articles]

    # Tokenize the text samples
    tokenized_articles = [nltk.word_tokenize(article.lower()) for article in text_samples]

    # Count the frequency of each token
    token_counter = Counter()
    total_tokens = 0
    for tokens in tokenized_articles:
        token_counter.update(tokens)
        total_tokens += len(tokens)

    # Select the most common tokens
    most_common_tokens = token_counter.most_common(vocab_size - 2)
    
    noise_dist = generate_noise_distribution(most_common_tokens)

    # Create the vocabulary with the most common tokens
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(most_common_tokens)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    word_prob = calculate_word_probabilities(token_counter, total_tokens)
    word_idx_prob = {vocab[word]: prob for word, prob in word_prob.items() if word in vocab}
    return tokenized_articles, vocab, noise_dist, word_idx_prob

def load_skipgram_data(vocab_size, context_size, amount_of_articles=None):
    tokenized_articles, vocab, noise_dist, word_prob = load_and_preprocess_data(vocab_size, amount_of_articles)
    return SkipGramDataset(tokenized_articles, vocab, context_size, word_prob), noise_dist


# Create data loaders
def create_data_loaders(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def generate_negative_samples(n_samples, noise_dist, batch_size):            
    noise_words = torch.multinomial(input = noise_dist, num_samples = batch_size * n_samples, replacement = True)
    return noise_words.view(batch_size, n_samples)