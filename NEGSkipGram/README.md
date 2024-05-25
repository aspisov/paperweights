# Distributed Representations of Words and Phrases and Their Compositionality (reimplementation)

This project is a PyTorch reimplementation of the Skip Gram model with negative sampling and subsampling of frequent words, based on the [Distributed Representations of Words and Phrases and Their Compositionality](https://arxiv.org/pdf/1310.4546) paper.

## Structure

- `word2vec/`: Contains the main scripts and utilities for the model.
- `data/`: Contains word embeddings in txt format.
- `train.ipynb`: Notebook for training the model with desired hyperparameters.
- `demo.ipynb`: Notebook with demonstration and visualization of acquired word embeddings.

## Quick Start

To try out word embeddings, run `demo.ipynb`.

These are embeddings from SkipGram with a 20,000 vocabulary size, 300 embedding dimensions, a 5-word context window, and trained on 48M tokens with subsampling. 

## Train Your Own Skip Gram

**Setup and Train:**

Run `train.ipynb` with desired hyperparameters:

```python
# Hyperparameters 
vocab_size = 20000
embedding_dim = 300
context_size = 5

epochs = 3
batch_size = 128
```

Embeddings will be saved to the `data/` directory.

**Visualize Your Own Embeddings:**

1. Run `demo.ipynb`.

## References

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). "Distributed Representations of Words and Phrases and Their Compositionality." [Read more here](https://arxiv.org/abs/1310.4546).
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." [Read more here](https://arxiv.org/abs/1301.3781).