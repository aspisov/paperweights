# LSTM Language Model

This repository contains an implementation of a Long Short-Term Memory (LSTM) language model trained on [TinyStories](https://arxiv.org/abs/2305.07759). The project includes scripts for training the model and generating text using a trained model.

## Repository Structure

- `generate.ipynb`: Jupyter notebook for text generation using a trained model
- `train.ipynb`: Jupyter notebook for training the LSTM language model
- `model_checkpoints/`: Directory containing saved model checkpoints
  - `best_model.pth`: The best performing model saved during training
  - `vocabulary.json`: JSON file containing the vocabulary used by the model

## Usage

### Training the Model

To train a new LSTM language model:

1. Open `train.ipynb` in a Jupyter environment.
2. Run the cells to train the model.
3. The best model will be automatically saved in the `model_checkpoints/` directory.

### Generating Text

To generate text using a trained model:

1. Ensure you have a trained model saved in the `model_checkpoints/` directory.
2. Open `generate.ipynb` in a Jupyter environment.
3. Run the cells to load the trained model and generate text.

## References
[TinyStories dataset](https://arxiv.org/abs/2305.07759)