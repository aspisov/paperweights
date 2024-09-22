# BERT fine-tuning

This repo contains code for finetuning BERT for [emotion classification](https://huggingface.co/datasets/dair-ai/emotion).

## Repository Structure
- `train.ipynb`: Jupyter notebook for training
- `eval.ipynb`: Jupyter notebook for testing on your examples
- `checkpoints/`: model checkpoints

## Usage

### Training the Model

To fine-tune your own BERT:

1. Open `train.ipynb` in a Jupyter environment.
2. Run the cells to train the model.
3. Your model will be automatically saved in the `checkpoints/` directory.

### Generating Text

To classify emotions from your own text using your trained model: 

1. Ensure you have a trained model saved in the `checkpoints/` directory.
2. Open `eval.ipynb` in a Jupyter environment.
3. Run the cells to load the trained model and classify emotion.
