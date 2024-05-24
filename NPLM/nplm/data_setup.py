"""
data_setup.py is used to load and prepare data for the NPLM model. 
It loads brown corpus which was used in the original paper.
"""
import torch
from torch.utils.data import DataLoader, Dataset

import pickle
import os
import nltk
from nltk.corpus import brown


class BrownDataset(Dataset):
    def __init__(self, words, context_size, vocab):
        super().__init__()
        context = [0] * context_size

        self.vocab = vocab
        self.X, self.y = [], []

        for word in words:
            idx = self.vocab.get(word, self.vocab["<UNK>"])
            self.X.append(torch.tensor(context, dtype=torch.long))
            self.y.append(idx)
            context = context[1:] + [idx]

        # Convert lists to tensors
        self.X = torch.stack(self.X).to(torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Retrieve the (context, target) pair at the specified index."""
        return self.X[idx], self.y[idx]


def ensure_dir(file_path):
    """
    Ensure that a directory exists.
    :param file_path: The directory path to ensure.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_vocab(vocab, file_path):
    """
    Save the vocabulary to a file in the specified directory.
    :param vocab: Vocabulary dictionary to save.
    :param file_path: Path including directory to save the vocabulary pickle file.
    """
    ensure_dir(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(vocab, f)


def load_vocab(file_path):
    """
    Load the vocabulary from a file in a specified directory.
    :param file_path: Path including directory to the vocabulary file.
    :returns: Loaded vocabulary dictionary.
    """
    with open(file_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def create_vocab(file_path):
    """
    Create and save a vocabulary from the Brown corpus.
    :param file_path: Path including directory to save the vocabulary pickle file.
    """
    nltk.download("brown")
    frequency = nltk.FreqDist(
        [w.lower() for w in brown.words()]
    )  # Convert to lowercase
    vocab = [w for w in frequency if frequency[w] >= 5]
    vocab = {w: i for i, w in enumerate(vocab)}
    vocab["<UNK>"] = len(vocab)
    save_vocab(vocab, file_path)


def load_data(context_size, vocab_file):
    """
    Load data and prepare dataloaders for the Brown Corpus.
    
    Parameters:
        context_size (int): Size of the context window for the model.
        vocab_file (str): Path including directory to the vocabulary file.
        
    Returns:
        tuple: A tuple containing two DataLoaders: train_dataloader and dev_dataloader.
            train_dataloader (DataLoader): DataLoader for the training dataset.
            dev_dataloader (DataLoader): DataLoader for the development dataset.
    """    
    nltk.download("brown")

    dataset = [w.lower() for w in brown.words()[: len(brown.words())]]
    train_size = int(len(dataset) * 0.9)
    train = dataset[:train_size]
    dev = dataset[train_size:]
    print(f"Training set size: {len(train)}, dev set size: {len(dev)}")

    if not os.path.exists(vocab_file):
        create_vocab(vocab_file)
    vocab = load_vocab(vocab_file)

    print(f"Vocabulary size: {len(vocab)}")

    train_dataset = BrownDataset(train, context_size, vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_dataset = BrownDataset(dev, context_size, vocab)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    return train_dataloader, dev_dataloader
