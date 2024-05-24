"""
Contains SkipGram model and Negative sampling loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------
# Base model
class Word2Vec(torch.nn.Module):
    def print_num_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n / 1e6:.2f}M")


# ----------------------------------
# SkipGram
class SkipGramNegativeSampling(Word2Vec):
    def __init__(self, vocab_size, embed_size, noise_dist=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.noise_dist = noise_dist

        self.input_embeddings = nn.Embedding(vocab_size, embed_size)
        self.output_embeddings = nn.Embedding(vocab_size, embed_size)

        # Initialize both embedding tables with uniform distribution
        self.input_embeddings.weight.data.uniform_(-1, 1)
        self.output_embeddings.weight.data.uniform_(-1, 1)

        self.print_num_params()

    def forward_input(self, center_words):
        return self.input_embeddings(center_words)

    def forward_output(self, context_words):
        return self.output_embeddings(context_words)

    def forward_noise(self, noise_words):
        return self.output_embeddings(noise_words)

# ----------------------------------
# Loss
class NegativeSampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_vectors, output_vectors, noise_vectors):
      
        batch_size, embed_size = input_vectors.shape
        
        input_vectors = input_vectors.view(batch_size, embed_size, 1)   # batch of column vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size) # batch of row vectors
        
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = torch.sum(noise_loss.squeeze(), dim=1)  # sum the losses over the sample of noise vectors

        return -torch.mean(out_loss + noise_loss)
