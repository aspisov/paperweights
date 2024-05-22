# load, process and prepare data for word2vec
from datasets import load_dataset
import nltk
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

class Word2VecDataset(Dataset):
    def __init__(self, tokenized_articles, vocab, context_size):
        self.vocab = vocab
        self.context = []
        self.center = []
        for tokens in tokenized_articles:
            token_ids = encode(tokens, vocab)
            current_context, current_center = self.generate_context_samples(token_ids, vocab, context_size)
            self.context.extend(current_context)
            self.center.extend(current_center)
            
    # Generate training samples with context windows
    def generate_context_samples(self, ids, vocab, context_size):
        context = []
        center = []
        # pad sequences
        padded_ids = ([vocab["<PAD>"]] * context_size + ids + [vocab["<PAD>"]] * context_size)

        for i in range(context_size, len(padded_ids) - context_size):
            current_context = (padded_ids[i - context_size : i]+ padded_ids[i + 1 : i + 1 + context_size])
            context.append(current_context)
            center.append(padded_ids[i])

        return context, center
    
    def __len__(self):
        return len(self.center)

# Dataset and Dataloader
class SkipGramDataset(Word2VecDataset):
    def __getitem__(self, idx):
        return torch.tensor(self.center[idx], dtype=torch.long), torch.tensor(self.context[idx], dtype=torch.long)

class CBOWDataset(Word2VecDataset): 
    def __getitem__(self, idx):
        return torch.tensor(self.context[idx], dtype=torch.long), torch.tensor(self.center[idx], dtype=torch.long)

# Convert tokens to ids
def encode(tokens, vocab):
    return [vocab[token] if token in vocab else vocab["<UNK>"] for token in tokens]


# Decode ids to tokens
def decode(ids, vocab):
    id_to_token = {v: k for k, v in vocab.items()}
    if ids.dim() == 0:
        return id_to_token[ids.item()]
    return [id_to_token[id.item()] for id in ids]


# Load and preprocess the dataset
def load_and_preprocess_data(vocab_size, context_size, amount_of_articles=None):
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
    for tokens in tokenized_articles:
        token_counter.update(tokens)

    # Select the most common tokens
    most_common_tokens = token_counter.most_common(vocab_size - 2)

    # Create the vocabulary with the most common tokens
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(most_common_tokens)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    return tokenized_articles, vocab, context_size

def load_skipgram_data(vocab_size, context_size, amount_of_articles=None):
    tokenized_articles, vocab, context_size = load_and_preprocess_data(vocab_size, context_size, amount_of_articles)
    return SkipGramDataset(tokenized_articles, vocab, context_size)

def load_cbow_data(vocab_size, context_size, amount_of_articles=None):
    tokenized_articles, vocab, context_size = load_and_preprocess_data(vocab_size, context_size, amount_of_articles)
    return CBOWDataset(tokenized_articles, vocab, context_size)

# Create data loaders
def create_data_loaders(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
