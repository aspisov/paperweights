import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from tqdm import tqdm


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # central word embeddings
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim)
        # context word embeddings
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.5 / self.embedding_dim
        self.in_embedding.weight.data.uniform_(-initrange, initrange)
        self.out_embedding.weight.data.uniform_(-0, 0)

    def forward(self, target_word, context_words, negative_words):
        # (batch_size, embedding_dim)
        target_embeds = self.in_embedding(target_word)

        # (batch_size, num_context_words, embedding_dim)
        context_embeds = self.out_embedding(context_words)
        # (batch_size, num_negative_words, embedding_dim)
        negative_embeds = self.out_embedding(negative_words)

        context_score = torch.bmm(
            context_embeds, target_embeds.unsqueeze(2)
        ).squeeze(2)
        negative_score = torch.bmm(
            negative_embeds, target_embeds.unsqueeze(2)
        ).squeeze(2)

        return context_score, negative_score


class Word2VecDataset(Dataset):
    def __init__(self, data, word_freqs, window_size, num_negative):
        self.data = data
        self.word_freqs = word_freqs
        self.window_size = window_size
        self.num_negative = num_negative

        # unigram distribution
        self.word_list = list(self.word_freqs.keys())
        self.word_counts = np.array(
            [count for count in self.word_freqs.values()]
        )
        self.word_counts = np.power(self.word_counts, 0.75)
        self.word_counts = self.word_counts / np.sum(self.word_counts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center_word = self.data[idx]
        context_words = self.get_context_words(idx)
        negative_words = self.get_negative_words(context_words)

        return (
            torch.tensor(center_word),
            torch.tensor(context_words),
            torch.tensor(negative_words),
        )

    def get_context_words(self, idx):
        start = max(0, idx - self.window_size)
        end = min(len(self.data), idx + self.window_size)
        context_word = [self.data[i] for i in range(start, end) if i != idx]

        # pad
        if len(context_word) < 2 * self.window_size:
            context_word += [0] * (2 * self.window_size - len(context_word))
        return context_word

    def get_negative_words(self, context_words):
        negative_samples = []
        while len(negative_samples) < self.num_negative:
            neg = np.random.choice(self.word_list, p=self.word_counts)
            if neg in context_words or neg in negative_samples:
                negative_samples.append(neg)
        return negative_samples


class Word2Vec:
    def __init__(
        self,
        sentences,
        vector_size=32,
        window=5,
        min_count=5,
        negative=5,
        epochs=5,
        batch_size=32,
        learning_rate=0.025,
    ):
        # set hyperparameters
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # prepare data
        self.vocab, self.word_freqs = self._build_vocab(sentences)
        self.vocab_size = len(self.vocab)
        self.data = self._prepare_data(sentences)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Data size: {len(self.data)}")

        # build model and train model
        self.model = Word2VecModel(self.vocab_size, self.vector_size)
        self.model.to(self.device)
        self._train()

    def _build_vocab(self, sentences):
        # create word frequencies
        word_counts = Counter(
            [word for sentence in sentences for word in sentence]
        )

        # filter words
        filter_words = [
            word
            for word, count in word_counts.items()
            if count >= self.min_count
        ]
        vocab = {word: i + 1 for i, word in enumerate(filter_words)}
        vocab["<pad>"] = 0
        word_freqs = {
            vocab[word]: count
            for word, count in word_counts.items()
            if word in vocab
        }
        return vocab, word_freqs

    def _prepare_data(self, sentences):
        return [
            self.vocab[word]
            for sentence in sentences
            for word in sentence
            if word in self.vocab
        ]

    def _train(self):
        dataset = Word2VecDataset(
            self.data, self.word_freqs, self.window, self.negative
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.9999
        )

        for epoch in range(self.epochs):
            total_loss = 0
            for center_word, context_words, negative_words in tqdm(dataloader):
                # move to device
                center_word = center_word.to(self.device)
                context_words = context_words.to(self.device)
                negative_words = negative_words.to(self.device)

                optimizer.zero_grad()

                # forward
                context_score, negative_score = self.model(
                    center_word, context_words, negative_words
                )

                # compute loss
                context_loss = F.logsigmoid(context_score).mean()
                negative_loss = F.logsigmoid(-negative_score).mean()
                loss = -context_loss - negative_loss

                # backward
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}")

    @property
    def wv(self):
        return WordVectors(
            self.model.in_embedding.weight.detach().numpy(), self.vocab
        )


class WordVectors:
    def __init__(self, vectors, vocab):
        self.vectors = vectors
        self.vocab = vocab
        self.index_to_key = [
            word for word, _ in sorted(self.vocab.items(), key=lambda x: x[1])
        ]

    def get_vector(self, word):
        if word not in self.vocab:
            raise KeyError(f"Word {word} not in vocabulary")
        return self.vectors[self.vocab[word]]
