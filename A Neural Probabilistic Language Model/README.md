# Neural Probabilistic Language Model (NPLM) Reimplementation

This project is a reimplementation of the classic [Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) using PyTorch, aimed at exploring the foundational techniques in language modeling.

## Structure

- `nplm/`: Contains the main scripts and utilities for the model.
- `models/`: Directory where trained models are saved.
- `data/`: Used to store vocabulary and training data.
- `sample.ipynb`: Jupyter notebook to demonstrate sampling from trained models.

## Quick Start

To try out a pre-trained model with a custom prompt, run `sample.ipynb`:

```python
prompt = "Lisa is a very nice"
```

## Train Your Own NPLM

**Setup and Train:**

Navigate to the `nplm/` directory, adjust the hyperparameters in `trainer.py`, and execute the script:

```python
# Set hyperparameters
context_size = 3
hidden_size = 60
embed_size = 30
direct = False
epochs = 5
```

**Run training**
```bash
python trainer.py
```

Models will be saved in `models/` with names reflecting the chosen hyperparameters.

**Sample from your model:**

1. Set the model's hyperparameters in `sample.ipynb`.
2. Choose your prompt.
3. Run the notebook to see the generated text.

## References

- Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C. (2003). "A Neural Probabilistic Language Model." Journal of Machine Learning Research. [Read more here](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).