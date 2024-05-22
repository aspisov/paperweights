# CBOW and SkipGram models
import torch
import torch.nn as nn
import torch.nn.functional as F    
    
# ----------------------------------
class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # Report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def forward(self, context_words):
        context_embeddings = self.embeddings(context_words).mean(dim=1)
        scores = self.linear(context_embeddings)
        return scores

    def get_num_params(self):
        # Return the total number of parameters in the model
        return sum(p.numel() for p in self.parameters())
    

# SkipGram
# ----------------------------------
class SkipGram(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # Report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def forward(self, x):
        embed = self.embeddings(x)
        return self.linear(embed)

    def get_num_params(self):
        # Return the total number of parameters in the model
        return sum(p.numel() for p in self.parameters())








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