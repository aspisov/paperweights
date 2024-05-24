# CBOW and SkipGram models
import torch
import torch.nn as nn
import torch.nn.functional as F


# base model
# ----------------------------------
class Word2Vec(torch.nn.Module):
    def print_num_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n / 1e6:.2f}M")


# ----------------------------------
class CBOW(Word2Vec):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        self.print_num_params()

    def forward(self, context_words):
        context_embeddings = self.embeddings(context_words).mean(dim=1)
        scores = self.linear(context_embeddings)
        return scores


class SkipGramNeg(Word2Vec):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist

        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

        self.print_num_params()

    def forward_center(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors  # input vector embeddings

    def forward_context(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors  # output vector embeddings


class NegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):

        batch_size, embed_size = input_vectors.shape

        input_vectors = input_vectors.view(
            batch_size, embed_size, 1
        )  # batch of column vectors
        output_vectors = output_vectors.view(
            batch_size, 1, embed_size
        )  # batch of row vectors

        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()

        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(
            1
        )  # sum the losses over the sample of noise vectors

        return -(out_loss + noise_loss).mean()  # average batch loss


# Hierarchical Softmax
# ----------------------------------
class TreeNode:
    def __init__(self, word=None, left=None, right=None):
        self.word = word
        self.left = left
        self.right = right
        self.params = None  # Parameters for binary classification


def build_tree(vocab):
    # Example tree building logic
    nodes = [TreeNode(word=w) for w in vocab]
    while len(nodes) > 1:
        right = nodes.pop()
        left = nodes.pop()
        parent = TreeNode(left=left, right=right)
        nodes.append(parent)
    return nodes[0]


class HierarchicalSoftmax(nn.Module):
    def __init__(self, root, embedding_dim):
        super().__init__()
        self.root = root
        self.embedding_dim = embedding_dim
        self.build_params(self.root)

    def build_params(self, root):
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left and node.right:
                node.params = nn.Linear(self.embedding_dim, 1).to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                stack.append(node.left)
                stack.append(node.right)

    def forward(self, embeddings, word_idx):
        batch_size = embeddings.size(0)
        loss = 0
        for b in range(batch_size):
            node = self.root
            while node.word is None:
                score = node.params(embeddings[b])
                if self.is_in_subtree(word_idx[b], node.left):
                    loss += F.binary_cross_entropy_with_logits(
                        score, torch.ones_like(score)
                    )
                    node = node.left
                else:
                    loss += F.binary_cross_entropy_with_logits(
                        score, torch.zeros_like(score)
                    )
                    node = node.right
        return loss / batch_size

    def is_in_subtree(self, word_idx, node):
        if node.word is not None:
            return node.word == word_idx
        return self.is_in_subtree(word_idx, node.left) or self.is_in_subtree(
            word_idx, node.right
        )


# ----------------------------------


# Hierarchical Softmax with CBOW
class HS_CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, tree_root):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hs = HierarchicalSoftmax(tree_root, embedding_dim)

        # Report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def forward(self, context_words, target_word):
        context_embeddings = self.embeddings(context_words).mean(dim=0)
        loss = self.hs(context_embeddings, target_word)
        return loss

    def get_num_params(self):
        # Return the total number of parameters in the model
        return sum(p.numel() for p in self.parameters())
